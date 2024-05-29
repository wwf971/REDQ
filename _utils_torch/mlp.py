import torch.nn as nn
from .wrapper import TorchModuleWrapper, ModuleList

class NonLinearLayer(TorchModuleWrapper):
    def Init(self, InputSize, OutputSize, NonLinear="ReLU"):
        config = self.config
        config.NonLinear = NonLinear
        config.InputSize = InputSize
        config.OutputSize = OutputSize
        if NonLinear in ["ReLU"]:    
            self.AddChild("NonLinear", nn.ReLU())
        else:
            raise Exception()

        self.AddTorchModule(
            "LinearTransform",
            nn.Linear,
            in_features=config.InputSize,
            out_features=config.OutputSize,
            bias=True
        )
        return self
    def forward(self, x):
        return self.NonLinear(self.LinearTransform(x))

class MLP(ModuleList):
    def Init(self, *LayersSize, NonLinear="ReLU"):
        assert len(LayersSize) >= 2
        config = self.config
        config.LayersSize = LayersSize
        config.InputSize = LayersSize[0]
        config.HiddenSizes = LayersSize[1:-1]
        config.OutputSize = LayersSize[-1] # size of last layer
        
        InputSize = config.InputSize
        LayerIndex = 0
        for OutputSize in config.LayersSize[1:]:
            layer = NonLinearLayer().Init(InputSize, OutputSize, NonLinear)
            self.AddChild("layer-%d"%LayerIndex, layer)
            LayerIndex += 1
            InputSize = OutputSize
        config.LayerNum = len(LayersSize) - 1
        return self