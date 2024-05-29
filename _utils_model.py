import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    nn = DLUtils.LazyImport("torch.nn")
    F = DLUtils.LazyImport("torch.nn.functional")
import math
from _utils_torch import (
    TorchModule, ModuleList, InitTorchModule,
    MLP,
    TorchModuleWrapper, TorchModule,
    LoadTorchModuleDict,
)
import DLUtils.network as network
class TransformerEncoder(ModuleList):
    # a stack of multiple multi-head self-attention layer
    def Init(self,
        LayerNum=5,
        TokenSize=512,
        QKSize=512,
        VSize=512,
        HeadNum=1,
        MLPNonLinear="ReLU",
        MLPMidLayerSize=512,
        DropOutRate=0.5
    ):
        config = self.config
        config.LayerNum = LayerNum
        config.TokenSize = TokenSize
        config.QKSize = QKSize
        config.VSize = VSize
        config.HeadNum = HeadNum
        config.MLPNonLinear = MLPNonLinear
        config.MLPMidLayerSize = MLPMidLayerSize
        config.DropOutRate = DropOutRate
        for Index in range(config.LayerNum):
                self.AddChild(
                "attention-%d"%Index,
                MultiheadSelfAttentionLayer().Init(
                    TokenSize = config.TokenSize,
                    QKSize = config.QKSize,
                    VSize = config.VSize,
                    HeadNum = config.HeadNum,
                    MLPMidLayerSize = config.MLPMidLayerSize,
                    MLPNonLinear = MLPNonLinear,
                    DropOutRate = config.DropOutRate
                )
            )
    def Build(self):
        config = self.config
        self.LayerNum = config.LayerNum
        self.TokenSize = config.TokenSize
        return super().Build()

class MultiheadSelfAttentionLayer(TorchModule):
    def Init(self,
        TokenSize=512,
        QKSize=512,
        VSize=512,
        HeadNum=1,
        MLPMidLayerSize=512,
        MLPNonLinear="ReLU",
        DropOutRate=0.5
    ):
        config = self.config
        config.TokenSize = TokenSize
        config.QKSize = QKSize
        config.VSize = VSize
        config.HeadNum = HeadNum
        config.MLPNonLinear = MLPNonLinear
        config.MLPMidLayerSize = MLPMidLayerSize
        config.DropOutRate = DropOutRate

        self.AddChildren(
            MultiHeadSelfAttention=MultiHeadSelfAttention(
                TokenSize=config.TokenSize,
                QKSize=config.QKSize,
                VSize=config.VSize,
                HeadNum=config.HeadNum,
            ),
            MLP=MLP(
                config.TokenSize,
                config.MLPMidLayerSize,
                config.TokenSize,
                NonLinear = config.MLPNonLinear
            ),
            LayerNorm1 = InitTorchModule(
                nn.LayerNorm, normalized_shape=(config.TokenSize)
            ),
            LayerNorm2 = InitTorchModule(
                nn.LayerNorm, normalized_shape=(config.TokenSize)
            ),
            DropOut = InitTorchModule(nn.Dropout, p=config.DropOutRate)
        )
        return self
    # multi-head self attention, with layer norm and 2-layer mlp
    def ReceiveNormTransformResidual(self, x):
        # LayerNorm --> MSA / MLP --> DropOut --> Residual
        # x: (BatchSize, TokenNumQ, TokenFeatureNum)
        y = self.LayerNorm1(x) # layer_norm
        y = self.MultiHeadSelfAttention(y) # multi-head attention
        y = self.DropOut(y) # dropout
        y = x + y # residual
        
        z = self.LayerNorm2(y) # layer_norm
        z = self.MLP(z) # multi-layer perceptron
        z = self.DropOut(z) # dropout
        z = y + z # residual
        return z
    def ReceiveTransformResidualNorm(self, x):
        # MSA / MLP --> DropOut --> Residual --> LayerNorm
        # x: (BatchSize, TokenNumQ, TokenFeatureNum)
        y = self.MultiHeadSelfAttention(x)
        y = x + y # residual
        y = self.LayerNorm1(y)
        
        z = self.MLP(y)
        z = y + z # residual
        z = self.LayerNorm2(z)
        return z
    def Build(self):
        self.forward = self.ReceiveTransformResidualNorm
        return super().Build()

class MultiHeadAttention(TorchModule):
    def Init(self,
            TokenSize,
            QKSize, # total size of all heads
            VSize, # total size of all heads
            HeadNum
        ):
        config = self.config
        config.VSize = VSize
        config.QKSize = QKSize
        config.TokenSize = TokenSize
        config.HeadNum = HeadNum
        self.AddChildren(
            LinearMapQ=InitTorchModule(nn.Linear,
                in_features=config.TokenSize,
                out_features=config.QKSize,
                bias=False
            ),
            LinearMapK=InitTorchModule(nn.Linear,
                in_features=config.TokenSize,
                out_features=config.QKSize,
                bias=False
            ),
            LinearMapV=InitTorchModule(nn.Linear,
                in_features=config.TokenSize,
                out_features=config.VSize,
                bias=False
            ),
            VToOut=InitTorchModule(nn.Linear,
                in_features=config.VSize,
                out_features=config.TokenSize,
                bias=False
            ),
        )
        return self
    def Build(self):
        config = self.config
        self.QKSize = config.QKSize
        self.VSize = config.VSize
        self.HeadNum = config.HeadNum
        assert self.QKSize % self.HeadNum == 0
        assert self.VSize % self.HeadNum == 0

        self.QKSizeHead = self.QKSize // self.HeadNum
        self.VSizeHead = self.VSize // self.HeadNum
        self.QKDotProductCoeff = 1.0 / self.QKSizeHead ** 0.5

        return super().Build()
    def forward(self,
            Q, # (BatchSize, TokenNumQ,  TokenSize)
            K, # (BatchSize, TokenNumKV, TokenSize)
            V # (BatchSize, TokenNumKV,  TokenSize)
        ):
        BatchSize = Q.size(0)
        TokenNumQ = Q.size(1)
        TokenNumKV = K.size(1)
        QKSize = Q.size(2)
        
        Q1 = self.LinearMapQ(Q)
            # (BatchSize, TokenNumQ, QKSizeTotal)
        K1 = self.LinearMapQ(K)
            # (BatchSize, TokenNumKV, QKSizeTotal)
        V1 = self.LinearMapQ(V)
            # (BatchSize, TokenNumKV, VSizeTotal)

        Q1 = Q1.view(BatchSize, TokenNumQ, self.HeadNum, self.QKSizeHead)
        K1 = K1.view(BatchSize, TokenNumKV, self.HeadNum, self.QKSizeHead)
        V1 = V1.view(BatchSize, TokenNumKV, self.HeadNum, self.VSizeHead)

        Q1 = Q1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumQ, QKSize)
        K1 = K1.permute(0, 2, 3, 1) # (BatchSize, HeadNum, QKSize, TokenNumKV)
        V1 = V1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumKV, VSize)

        AttentionCoeff = torch.matmul(Q1, K1) / self.QKDotProductCoeff
        # (BatchSize, HeadNum, TokenNumQ, TokenNumKV)
        AttentionCoeff = F.softmax(AttentionCoeff, 3)
        
        VAttention = torch.matmul(AttentionCoeff, V1)
        # (BatchSize, HeadNum, TokenNumQ, VSize)

        V2 = VAttention.permute(0, 2, 1, 3)
        V2 = V2.reshape(BatchSize, TokenNumQ, self.VSize)
        Out = self.VToOut(V2) # (BatchSize, TokenNumQ, OutSize)
        return Out


MA = MHA = MultiHeadAttention
AttentionMultiHead = MultiHeadAttention

class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self,
        x # (BatchSize, TokenNum, TokenSize)
    ):
        return super().forward(x, x, x)
MSA = MHSA = MultiHeadSelfAttention

def attention(
    Q, # Q: (BatchSize, TokenNumQ,   InSize)
    K, # K: (BatchSize, TokenNumKV,  InSize)
    V, # V: (BatchSize, TokenNumKV,  InSize)
    WQ, # WQ: (InSize, QKSize)
    WK, # WK: (InSize, QKSize)
    WV, # WV: (InSize, VSize)
    WO # WO: (VSize, OutSize)
):
    
    Q1 = torch.matmul(Q, WQ) # (BatchSize, TokenNumQ, QKSize)
    K1 = torch.matmul(K, WK) # (BatchSize, TokenNumKV, QKSize)
    V1 = torch.matmul(V, WV) # (BatchSize, TokenNumKV, VSize)
    
    QKSize = Q.size(2)
    AttentionCoeff = torch.matmul(
        Q1,                  # (BatchSize, TokenNumQ, QKSize)
        K1.permute(0, 2, 1)  # (BatchSize, QKSize, TokenNumKV)
    ) # (BatchSize, TokenNumQ, TokenNumKV)

    AttentionCoeff = AttentionCoeff / math.sqrt(QKSize)
    AttentionCoeff = F.softmax(AttentionCoeff,21) # (BatchSize, TokenNumQ, TokenNumKV)
    
    V2  = torch.matmul(AttentionCoeff, V1) # (BatchSize, TokenNumQ, VSize)
    V3 = torch.matmul(V2, WO)
    return V3

def attention_multi_head(Q, K, V, WQ, WK, WV, WO, HeadNum):
    # Q: (BatchSize, TokenNumQ,   QKSize)
    # K: (BatchSize, TokenNumKV,  QKSize)
    # V: (BatchSize, TokenNumKV,  VSize )

    BatchSize = Q.size(0)
    TokenNumQ = Q.size(1)
    QKSize = Q.size(2)
    TokenNumKV = K.size(1)
    VSize = WV.size(0)
    QKSizeHead = QKSize // HeadNum
    VSizeHead = VSize // HeadNum

    Q1 = torch.matmul(Q, WQ)
    K1 = torch.matmul(K, WK)
    V1 = torch.matmul(V, WV)
    
    Q1 = Q1.view(BatchSize, TokenNumQ, HeadNum, QKSizeHead)
    K1 = K1.view(BatchSize, TokenNumKV, HeadNum, QKSizeHead)
    V1 = V1.view(BatchSize, TokenNumKV, HeadNum, VSizeHead)
    
    Q1 = Q1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumQ, QKSizeHead)
    K1 = K1.permute(0, 2, 3, 1) # (BatchSize, HeadNum, QKSizeHead, TokenNumKV)
    V1 = V1.permute(0, 2, 1, 3) # (BatchSize, HeadNum, TokenNumKV, VSizeHead)

    AttentionCoeff = torch.matmul(Q1, K1) / QKSizeHead ** 0.5
    # (BatchSize, HeadNum, TokenNumQ, TokenNumKV)
    AttentionCoeff = F.softmax(AttentionCoeff)
    VAttention = torch.matmul(AttentionCoeff, V1)
    # (BatchSize, HeadNum, TokenNumQ, VSizeHead)
    
    V2 = VAttention.permute(0, 2, 1, 3) # (BatchSize, TokenNumQ, HeadNum, VSizeHead)
    V2 = V2.reshape(BatchSize, TokenNumQ, VSize) # (BatchSize, TokenNumQ, VSize)
    
    V3 = torch.matmul(V2, WO) # (BatchSize, TokenNumQ, OutSize)
    # output token num is same as token num of Q.
    return V2
attention_multihead = multihead_attetnion = multi_head_attention = attention_multi_head