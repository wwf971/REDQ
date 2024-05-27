import sqlite3
from _utils_import import DLUtils
import _utils
from typing import TYPE_CHECKING

def SetSeed(seed: int, Random=None, Numpy=None, PyTorch=None, ):
    import torch
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


    return _utils.Dict(
        
    )

def GetDataBaseConnection(FilePath):
    con = sqlite3.connect(FilePath)
    return con


import collections
class PerformanceAlongEpochBatchTrain:
    def __init__(self, BatchNumMax):
        # collections.deque: FIFO(first-in-first-out) queue with max length.
        self.PerformanceList = collections.deque([], maxlen=BatchNumMax)
        self.SampleNumList = collections.deque([], maxlen=BatchNumMax)
    def Append(self, Performance: float, SampleNum: int):
        self.PerformanceList.append(Performance)
        self.SampleNumList.append(SampleNum)
        return self
    def ReportAverage(self):
        # assert len(self.PerformanceList) == len(self.SampleNumList)
        SampleNumTotal = 0
        PerformanceTotal = 0.0
        for Index in range(len(self.PerformanceList)):
            PerformanceTotal += self.PerformanceList[Index]
            SampleNumTotal += self.SampleNumList[Index]
        return PerformanceTotal / SampleNumTotal

class AccuracyAlongEpochBatchTrain:
    def __init__(self, BatchNumMax):
        self.NumCorrectList = collections.deque([], maxlen=BatchNumMax)
        self.NumSampleList = collections.deque([], maxlen=BatchNumMax)
    def Append(self, NumCorrect: float, NumSample: int):
        self.NumCorrectList.append(NumCorrect)
        self.NumSampleList.append(NumSample)
        return self
    def ReportAverage(self):
        # assert len(self.NumCorrectList) == len(self.NumSampleList)
        NumCorrectTotal = 0
        NumSampleTotal = 0.0
        for Index in range(len(self.NumCorrectList)):
            self.NumSampleList[Index]
            NumCorrectTotal += self.NumCorrectList[Index]
            NumSampleTotal += self.NumSampleList[Index]
        return NumCorrectTotal / NumSampleTotal


class TriggerFuncAtEveryFixedInterval:
    def __init__(self, Interval, Func, *Args, **KwArgs):
        self.Args = Args
        self.KwArgs = KwArgs
        self.Reset()
        self.Func = Func
        self.Interval = Interval
    def Reset(self):
        self.Count = 0
    def Tick(self):
        self.Count += 1
        if self.Count >= self.Interval:
            Result = self.Func(*self.Args, **self.KwArgs)
            self.Reset()
            return Result
        else:
            Result = None
        return Result
