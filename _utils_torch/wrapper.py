import DLUtils
import torch
import torch.nn as nn

import DLUtils
import torch
from _utils import (
    Dict,
    NpArrayToTorchTensor,
    TorchTensorToNpArray,
    GetScriptDirPath,
    GetCurrentScriptDirPath,
    get_file_path_without_suffix,
    class_path_from_class_instance,
    class_instance_from_class_path,
)

import numpy as np

class TorchModuleWrapper(nn.Module):
    def __init__(self, *Args, **KwArgs):
        super().__init__()
        self.config = Dict()
        if len(Args) + len(KwArgs) > 0:
            self.Init(*Args, **KwArgs)
    def Init(self):
        self._HasInit = True
        return self
    def Build(self):
        for Name, Child in dict(self.named_children()).items():
            if isinstance(Child, TorchModule):
                Child.Build()
            else:
                assert isinstance(Child, torch.nn.Module)
                BuildTorchModule(Child)
        return self
    def GetClassPath(self):
        return class_instance_from_class_path(self)
    def AddParam(self, Name, Param, **ParamDict):
        if isinstance(Param, np.ndarray):
            Param = torch.from_numpy(Param).float() # dtype=float32
        self.register_parameter(
            Name, torch.nn.Parameter(Param)
        )
        if len(ParamDict) > 0:
            self.AddParams(ParamDict)
        return self
    def AddParams(self, **ParamDict):
        """add many pairs of params to torch module"""
        for Name, Param in ParamDict.items():
            self.AddParam(Name, Param)
    def AddBuffer(self, Name, Value: torch.Tensor, **BufferDict):
        self.register_buffer(Name, Value)
        if len(BufferDict) > 0:
            self.AddBuffers(**BufferDict)
        return self
    def AddBuffers(self, **BufferDict):
        for Name, Param in BufferDict.items():
            self.AddBuffer(Name, Param)
        return self
    def AddSubModule(self, Name=None, Module=None, **SubModuleDict):
        if len(SubModuleDict) > 0:
            assert Name is None and Module is None
            for SubModuleName, SubModule in SubModuleDict.items():
                self.AddSubModule(SubModuleName, SubModule)
            return
        # self.add_module(Name, Module)
        if isinstance(Module, TorchModule):
            self.add_module(Name, Module)
        else:
            assert isinstance(Module, torch.nn.Module)
            self.add_module(Name, Module)

        # setattr(self, Name, Module)
            # torch will check if Module is subclass of torch.nn.Module
        return self
    AddModule = AddChild = AddSubModule
    def AddSubModules(self, **ModuleDict):
        for Name, Child in ModuleDict.items():
            self.AddChild(Name, Child)
        return self
    AddModules = AddChildren = AddSubModules
    def GetSubModule(self, SubModuleName):
        assert SubModuleName in self.GetChildrenNameList()
        return getattr(self, SubModuleName)
    def AddTorchModule(self, Name, ModuleClass, **KwArgs):
        Module = ModuleClass(**KwArgs)
        Module.config = {
            "init_args": KwArgs
        }
        self.add_module(Name, Module)
        return self
    def AddModuleList(self, Name, *ModulesList, **ModulesDict):
        if len(ModulesList) > 0:
            # ModuleName will be like "1", "2", ...
            assert len(ModulesList) == 0
            self.AddChild(
                Name,
                ModuleList().Init(
                    *ModulesList
                )
            )
        elif len(ModulesDict) > 0:
            assert len(ModulesList) == 0
            self.AddChild(
                Name,
                ModuleList().Init(
                    **ModulesDict
                )
            )
        else:
            raise Exception()
        return self
    def GetParamDict(self):
        ParamDict = GetTorchModuleParamDict(self)
        return ParamDict
    def LoadParamDict(self, ParamDict):
        LoadTorchModuleParamDict(self, ParamDict)
    def GetBufferDict(self):
        BufferDict = GetTorchModuleBufferDict(self)
        return BufferDict
    def LoadBufferDict(self, BufferDict):
        LoadTorchModuleBufferDict(self, BufferDict)
    def GetChildrenNameList(self):
        ChildNameList = []
        for Name, SubModule in dict(self.named_children()).items():
            if "." in Name:
                continue
            else:
                ChildNameList.append(Name)
        return ChildNameList
    def GetChildrenDict(self):
        # get a dict of submodules
        ChildrenDict = {}
        for Name, Child in dict(self.named_children()).items():
            if "." in Name:
                continue
            if isinstance(Child, TorchModule):
                ChildrenDict[Name] = Child.GetModuleDict()
            else:
                assert isinstance(Child, torch.nn.Module)
                ChildrenDict[Name] = GetTorchModuleDict(Child)
        return ChildrenDict
    def LoadChildrenDict(self, ChildrenDict: dict):
        # load submodules
        ChildrenDict = Dict(ChildrenDict)
        for Name, ChildDict in ChildrenDict.items():
            ChildDict = Dict(ChildDict)
            ChildClassPath = ChildDict._class_path
            if "init_args" in ChildDict.config:
                InitArgs = ChildDict.config["init_args"]
            else:
                InitArgs = {}
            Child = class_path_from_class_instance(
                ChildClassPath,
                **InitArgs
            )
            if isinstance(Child, TorchModule):
                Child.LoadModuleDict(ChildDict)
            else:
                assert isinstance(Child, torch.nn.Module)
                LoadTorchModuleDict(Child, ChildDict)
            self.AddChild(Name, Child)
        return self
    def GetModuleDict(self):
        self.config._class_path = self.GetClassPath() # for debug
        return {
            "config": self.config.to_dict(),    
            "param": self.GetParamDict(),
            "buffer": self.GetBufferDict(),
            "children": self.GetChildrenDict(), # submodules
            "_class_path": self.GetClassPath()
        }
    def LoadFromDict(self, ModuleDict: dict, LoadChildrenDict=True):
        ModuleDict = Dict(ModuleDict)
        self.config = Dict(ModuleDict.config)
        self.LoadParamDict(ModuleDict.param)
        self.LoadBufferDict(ModuleDict.buffer)
        if LoadChildrenDict:
            self.LoadChildrenDict(ModuleDict.children)
        return self
    LoadModuleDict = FromDict = LoadFromDict
    def ToFile(self, FilePath):
        FilePath = DLUtils.file.EnsureFileDir(FilePath)
        ModuleDict = self.GetModuleDict()
        DLUtils.file.ObjToBinaryFile(ModuleDict, FilePath)
        return self
    def FromFile(self, FilePath):
        if hasattr(self, "config"):
            delattr(self, "config")
        FilePath = DLUtils.file.CheckFileExists(FilePath)
        ModuleDict = DLUtils.file.BinaryFileToObj(FilePath)
        self.LoadModuleDict(ModuleDict, LoadChildrenDict=True)
        return self
    def Clear(self):
        if hasattr(self, "config"):
            delattr(self, "config")
        if hasattr(self, ""):
            delattr(self, "TorchModule")
        if hasattr(self, "_HasBuild"):
            delattr(self, "_HasBuild")
        if hasattr(self, "_IsLoad"):
            delattr(self, "_IsLoad")
        return self
    # def __repr__(self):
    #     return PrintTorchModule(self)
TorchModule = TorchModuleWrapper

def PrintModuleParam(model: torch.nn.Module, OutPipe=None):
    # print(model)
    if OutPipe is None:
        OutPipe = DLUtils.GetLibOutPipeWriter()
    ParamDict = dict(model.named_parameters())
    TrainParamList = []
    for name, value in ParamDict.items():
        TrainParamList.append(name)
        OutPipe.print("Param: %s.\t%s."%(name, value.shape))
    return TrainParamList

def PrintTorchModule(model: torch.nn.Module, OutPipe=None):
    if OutPipe is None:
        OutPipe = DLUtils.GetLibOutPipeWriter()
    ParamDict = dict(model.named_parameters())
    for Name, Param in ParamDict.items():
        if "." in Name: # Param belongs to one of child modules.
            continue
        OutPipe.print("param: %s.\t%s."%(Name, list(Param.shape)))
    SubModuleDict = dict(model.named_children()) # list direct submodule of the module
    for Name, SubModule in SubModuleDict.items():
        OutPipe.print("SubModule: %s. class: %s"%(
            Name,
            SubModule._get_name() # torch.nn.Linear ==> name
        ))
        with OutPipe.IncreasedIndent():
            PrintTorchModule(SubModule, OutPipe=OutPipe)

def ListSubModule(model: torch.nn.Module, OutPipe=None):
    if OutPipe is None:
        OutPipe = DLUtils.GetLibOutPipeWriter()
    SubModuleDict = dict(model.named_children()) # list direct submodule of the module
    for Name, SubModule in SubModuleDict.items():
        OutPipe.print("%s %s"%(Name, SubModule))    
    return

def TorchModuleParamNum(Module: torch.nn.Module):
    ParamNum = sum(p.numel() for p in Module.parameters())
    # print(f"Total number of parameters: {total_params}")
    return ParamNum

def GetTorchModuleConfig(Module: torch.nn.Module):
    if hasattr(Module, "config"):
        config = Module.config
        if isinstance(config, Dict):
            return config.to_dict()
        else:
            return config
    else:
        config = Dict()
        # if isinstance(Module, torch.nn.Linear):
        #     config.init_args.in_features = Module.in_features
        #     config.init_args.out_features = Module.out_features
        #     config.init_args.bias = Module

        # if len(config) == 0:
        #     return None
        # else:
        #     return config
        return config.to_dict()


def LoadTorchModuleParamDict(Module: torch.nn.Module, ParamDict):
    for Name, NpArray in ParamDict.items():
        assert isinstance(NpArray, np.ndarray)
        Tensor = NpArrayToTorchTensor(NpArray)
        Module.register_parameter(Name, nn.Parameter(Tensor))

def GetTorchModuleBufferDict(Module: torch.nn.Module):
    BufferDict = {}
    for Name, Tensor in dict(Module.named_buffers()).items():
        assert isinstance(Tensor, torch.Tensor)
        if "." in Name:
            continue
        BufferDict[Name] = Tensor.cpu().detach().numpy()
    return BufferDict

def LoadTorchModuleBufferDict(Module: torch.nn.Module, BufferDict):
    for Name, NpArray in BufferDict.items():
        assert isinstance(NpArray, np.ndarray)
        Tensor = NpArrayToTorchTensor(NpArray)
        Module.register_buffer(Name, Tensor)

def GetTorchModuleChildrenDict(Module: torch.nn.Module):
    ChildrenDict = {}
    for Name, Child in dict(Module.named_children()).items():
        if "." in Name:
            continue
        ChildrenDict[Name] = GetTorchModuleDict(Child)
    return ChildrenDict

def Torchmodule_to_dict(Module: torch.nn.Module):
    ModuleDict = {
        "config": GetTorchModuleConfig(Module),
        "param": GetTorchModuleParamDict(Module),
        "buffer": GetTorchModuleBufferDict(Module),
        "children": GetTorchModuleChildrenDict(Module),
        "_class_path": class_instance_from_class_path(Module)
    }
    return ModuleDict
GetTorchModuleDict = Torchmodule_to_dict

def BuildTorchModule(Module: torch.nn.Module, OutPipe=None):
    for Name, Child in dict(Module.named_children()).items():
        if "." in Name:
            continue
        if isinstance(Child, TorchModuleWrapper):
            Child.Build()
        else:
            BuildTorchModule(Child)

def TorchModuleToFile(Module: torch.nn.Module, FilePath, OutPipe=None):
    DLUtils.file.ObjToBinaryFile(GetTorchModuleDict(Module), FilePath)
    return

def InitTorchModule(ModuleClass, **KwArgs):
    Module = ModuleClass(**KwArgs)
    Module.config = {
        "init_args": KwArgs
    }
    return Module

def LoadTorchModuleChildrenDict(Module: torch.nn.Module, ChildrenDict):
    # load submodules
    for Name, ChildDict in ChildrenDict.items():
        ChildClassPath = ChildDict._class_path
        if "init_args" in ChildDict.config:
            InitArgs = ChildDict.config["init_args"]
        else:
            InitArgs = {}
        Child = class_path_from_class_instance(ChildClassPath, **InitArgs)
        LoadTorchModuleDict(ChildDict)
        Module.add_module(Name, Child)
    return Module

def LoadTorchModuleDict(Module, ModuleDict: dict, LoadChildrenDict=True):
    ModuleDict = Dict(ModuleDict)
    Module.config = Dict(ModuleDict.config)
    LoadTorchModuleParamDict(Module, ModuleDict.param)
    LoadTorchModuleBufferDict(Module, ModuleDict.buffer)
    if LoadChildrenDict:
        LoadTorchModuleChildrenDict(Module, ModuleDict.children)
    return Module

def GetTorchModuleParamDict(Module: torch.nn.Module):
    ParamDict = {}
    for Name, Tensor in dict(Module.named_parameters()).items():
        if "." in Name:
            continue
        ParamDict[Name] = TorchTensorToNpArray(Tensor)
    return ParamDict

def LoadTorchModuleFromDict(ModuleDict: dict):
    assert isinstance(ModuleDict, dict)
    ModuleDict = Dict(ModuleDict)
    ClassPath = ModuleDict._class_path

    Module = class_path_from_class_instance(ClassPath)
    if isinstance(Module, TorchModuleWrapper):
        Module.LoadModuleDict(ModuleDict)
    elif isinstance(Module, torch.nn.Module):
        LoadTorchModuleDict(Module, ModuleDict)
    else:
        raise Exception()
    # ClassObj = Class()
    # assert isinstance(ClassObj, DLUtils.torch.TorchModuleWrapper)
    return Module

def LoadTorchModuleFromFile(FilePath):
    FilePath = DLUtils.CheckFileExists(FilePath)
    ModuleDict = DLUtils.file.BinaryFileToObj(FilePath)
    return LoadTorchModuleFromDict(ModuleDict)

class ModuleList(TorchModuleWrapper):
    def Init(self, *ModulesList, **ModulesDict):
        if len(ModulesList) > 0:
            assert len(ModulesDict) == 0
            for Index, Module in enumerate(ModulesList):
                self.AddChild(
                    "%d"%Index, Module
                )
        elif len(ModulesDict) > 0:
            assert len(ModulesList) == 0
            for Name, Module in ModulesDict.items():
                self.AddChild(
                    Name, Module
                )
        return self
    def Build(self):
        self.ModuleList = []
        for Name, Child in self.named_children():
            self.ModuleList.append(Child)
        return super().Build()
    def forward(self, x):
        y = x
        for Child in self.ModuleList:
            y = Child(y)
        return y