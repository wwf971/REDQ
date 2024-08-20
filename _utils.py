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
    def hasattr(self, key):
        return key in self
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
        for key, value in _dict.items():
            if isinstance(value, dict):
                self[key] = Dict(value)
            else:
                self[key] = value
        return self
    def to_dict(self):        
        _dict = dict()
        for key, value in self.items():
            if isinstance(value, Dict):
                _dict[key] = value.to_dict()
            else:
                _dict[key] = value
        return _dict
    def update(self, dict_external:Dict):
        for key, value in dict_external.items():
            if isinstance(value, dict) and not isinstance(value, Dict):
                value = Dict(value)
            if not hasattr(self, key):
                self[key] = value
                continue

            value_old = self[key]
            if isinstance(value_old, dict):
                if not isinstance(value_old, Dict):
                    value_old = Dict(value_old)
                value_old.update(value)
            else:
                self[key] = value # overwrite
        return self
    def create_if_non_exist(self, key) -> Dict:
        if key in self:
            return self[key]
        else:
            value = Dict()
            self[key] = value
            return value
    def check_key_exist(self, key):
        assert key in self
        return self
    def set_if_non_exist(self, **_dict):
        for key, value in _dict.items():
            if key in self:
                continue
            else:
                self[key] = value
        return self

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
        self.param = Dict()
        self.config = Dict()
        self.submodules = Dict()
        if len(Args) + len(KwArgs) > 0:
            self.Init(*Args, **KwArgs)
    def AddSubModule(self, Name=None, SubModule=None, **SubModuleDict):
        if len(SubModuleDict) > 0:
            for _Name, _SubModule in SubModuleDict.items():
                self.AddSubModule(_Name, _SubModule)
            assert Name is None and SubModule is None
        else:
            self.submodules[Name] = SubModule
            setattr(self, Name, SubModule)
        return self
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
        self.param[Name] = Value
        setattr(self, Name, Value)
        return self
    def FromFile(self, FilePath):
        FilePath = DLUtils.file.CheckFileExists(FilePath)
        ModuleDict = DLUtils.file.BinaryFileToObj(ModuleDict)

    def FromDict(self, ModuleDict: dict):
        self.config = ModuleDict["config"]
        self.param = ModuleDict["param"]
        for Name, SubModuleDict in ModuleDict["submodules"].items():
            self.AddSubModule(
                Name, LoadModuleFromDict(SubModuleDict)
            )
        for Name, Value in self.param.items():
            setattr(self, Name, Value) # mount param to self
        return self

    def ToDict(self):
        for Name in self.param.keys():
            self.param[Name] = getattr(self, Name) # collect param from self
        return {
            "config": self.config,
            "param": self.param,
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