from __future__ import annotations
from _utils_import import DLUtils
import numpy as np
import torch
import torch.nn as nn

import os
import pathlib
from pathlib import Path

from _utils import (
    GetCurrentScriptDirPath,
    get_file_path_without_suffix,
)

from _utils_torch.wrapper import(
    TorchModuleWrapper,
    ModuleList, TorchModule,
    InitTorchModule,
    LoadTorchModuleFromFile,
    LoadTorchModuleFromDict,
    LoadTorchModuleBufferDict,
    LoadTorchModuleParamDict,
    LoadTorchModuleChildrenDict,
    BuildTorchModule,
    GetTorchModuleDict, Torchmodule_to_dict,
    GetTorchModuleParamDict,
    GetTorchModuleBufferDict,
    GetTorchModuleChildrenDict,
    GetTorchModuleConfig,
    PrintTorchModule,
)

from _utils_torch.mlp import (
    MLP,
)

if __name__=="__main__":
    # example usage:
    mlp = MLP().Init(10, 20, 30, 40).Build()
    print(mlp)
    CurrentDirPath = GetCurrentScriptDirPath(__file__)
    SaveFilePath = get_file_path_without_suffix(__file__) + "-model.dat"
    mlp_2 = mlp.ToFile(
        SaveFilePath
    ).FromFile(
        SaveFilePath
    )
    print(mlp_2)
    # mlp.register_parameter(
    #     "a.b", torch.nn.Parameter(torch.Tensor((10, 10)))
    # ) # => KeyError: 'parameter name can\'t contain "."'
    #     # use this rule to exclude parameters
    PrintTorchModule(mlp_2)
