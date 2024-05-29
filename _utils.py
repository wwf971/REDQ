from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from _utils_import import DLUtils

import os
from pathlib import Path
def GetScriptDirPath(__File__=None, EndWithSlash=True):
    if __File__ is None:
        __File__ = __file__
    # Using os.path
    script_dir_os = os.path.dirname(os.path.abspath(__File__))
    print(f"Using os.path: {script_dir_os}")

    # Using pathlib
    script_dir_pathlib = Path(__File__).resolve().parent.__str__()
    print(f"Using pathlib: {script_dir_pathlib}")
    
    if EndWithSlash:
        script_dir_pathlib += "/"
    return script_dir_pathlib
GetCurrentScriptDirPath = GetScriptDirPath

def GetCurrentFilePathWithoutSuffix(__File__=None, EndWithSlash=True):
    if __File__ is None:
        __File__ = __file__
    Name, Suffix = DLUtils.file.SeparateFileNameAndSuffix(__File__)
    return Name

def GetClassPathFromClassInstance(Instance):
    cls = Instance.__class__
    module = cls.__module__
    qualname = cls.__qualname__
    return f"{module}.{qualname}"

def GetClassInstanceFromClassPath(ClassPath: str, **KwArgs):
    import importlib
    # Split the class_path into module path and class name
    ModulePath, ClassName = ClassPath.rsplit('.', 1)
    # Import the module
    module = importlib.import_module(ModulePath)
    # Get the class
    cls = getattr(module, ClassName)
    # Create an instance of the class
    instance = cls(**KwArgs)
    return instance

def NpArrayToTorchTensor(NpArray: torch.Tensor):
    return torch.from_numpy(NpArray)

def TorchTensorToNpArray(Tensor: np.ndarray):
    return Tensor.cpu().detach().numpy()

class Dict(dict):
    """
    A subclass of dict that allows attribute-style access.
    """
    def __init__(self,
            _dict: dict = None,
            # allow_missing_attr=False,
            **Dict
        ):
        # self.allow_missing_attr = allow_missing_attr
        """
            If allow_missing_attr == True, empty Dict object will be created and returned
                when trying to get a non-existent attribute
        """
        if _dict is not None:
            assert len(Dict) == 0
            if not isinstance(_dict, dict):
                raise TypeError("Expect a dictionary")
            self.from_dict(_dict)
        
        if len(Dict) > 0:
            assert _dict is None
            self.from_dict(Dict)
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
    def __setattr__(self, key, value):
        """
        will be called when setting attribtue in this way: DictObj.a = b
        """
        self[key] = value
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
    def test(self):
        # Example usage:
        d = Dict()
        d.a = 10
        print(d.a)  # Output: 10
        print(d['a'])  # Output: 10

        d['b'] = 20
        print(d.b)  # Output: 20

        del d.a
        # print(d.a)  # Raises AttributeError
    def from_dict(self, _dict: dict):
        if not isinstance(_dict, dict):
            raise TypeError("Expect a dictionary")
        for Key, Value in _dict.items():
            if isinstance(Value, dict):
                self[Key] = Dict(Value)
            else:
                self[Key] = Value
        return self
    def to_dict(self):        
        _dict = dict()
        for key, value in self.items():
            if isinstance(value, Dict):
                _dict[key] = value.to_dict()
            else:
                _dict[key] = value
        return _dict

class DefaultDict(Dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            # if self.allow_missing_attr:
            child = Dict()
            setattr(self, key, child)
            return child
            # else:
            #     raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

def LoadModuleFromDict(ModuleDict):
    assert isinstance(ModuleDict, dict)
    ModuleDict = Dict(ModuleDict)
    ClassPath = ModuleDict._class_path
    module = GetClassInstanceFromClassPath(ClassPath)
    assert isinstance(module, Module)
    module.FromDict(ModuleDict)
    return module

def LoadModuleFromFile(FilePath):
    FilePath = DLUtils.file.CheckFileExists(FilePath)
    ModuleDict = DLUtils.file.BinaryFileToObj(FilePath)
    assert isinstance(ModuleDict, dict)
    return LoadModuleFromDict(ModuleDict)

def ModuleToDict(module: Module):
    return module.ToDict()

class Module():
    def __init__(self, *Args, **KwArgs):
        self.params = Dict()
        self.config = Dict()
        self.submodules = Dict()
        if len(Args) + len(KwArgs) > 0:
            self.Init(*Args, **KwArgs)
    def AddSubModule(self, **SubModuleDict):
        for Name, SubModule in SubModuleDict.items():
            self.submodules[Name] = SubModule
        setattr(self, Name, SubModule)
    def GetSubModuleDict(self):
        SubModuleDict = {}
        for Name in self.submodules.keys():
            SubModule = getattr(self, Name)
            assert isinstance(SubModule, Module)
            SubModuleDict[Name] = SubModule.ToDict()
        return SubModuleDict
    def AddParam(self, Name=None, Value=None, **ParamDict):
        if len(ParamDict) > 0:
            assert Name is None and Value is None
            for Name, Value in ParamDict.items():
                self.AddParam(Name, Value)
            return
        self.params[Name] = Value
        setattr(self, Name, Value)
        return self
    def FromFile(self, FilePath):
        FilePath = DLUtils.file.CheckFileExists(FilePath)
        ModuleDict = DLUtils.file.BinaryFileToObj(ModuleDict)

    def FromDict(self, ModuleDict: dict):
        self.config = ModuleDict["config"]
        self.params = ModuleDict["params"]
        for Name, SubModuleDict in ModuleDict["submodules"].items():
            self.AddSubModule(
                self, Name, LoadModuleFromDict(SubModuleDict)
            )
        for Name, Value in self.params.items():
            setattr(self, Name, Value)
        return self

    def ToDict(self):
        for Name in self.params.keys():
            # collect params
            self.params[Name] = getattr(self, Name)
        return {
            "config": self.config,
            "param": self.params,
            "submodules": self.GetSubModuleDict(),
            "_class_path": self.GetClassPath()
        }
    def ToFile(self, FilePath):
        FilePath = DLUtils.EnsureFileDir(FilePath)
        ModuleDict = self.ToDict()
        DLUtils.file.ObjToBinaryFile(ModuleDict, FilePath)
        return self
    def GetClassPath(self):
        return GetClassPathFromClassInstance(self)
    def Build(self):
        for SubModule in self.submodules.values():
            SubModule.Build()
        return self