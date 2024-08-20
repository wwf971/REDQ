from __future__ import annotations
from _utils_import import DLUtils
from _utils import (
    Dict,
    class_instance_from_class_path,
    class_path_from_class_instance
)

# a lightweight wrapper
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
    def get_submodule_dict(self):
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
    def ToFile(self, FilePath):
        FilePath = DLUtils.EnsureFileDir(FilePath)
        ModuleDict = self.ToDict()
        DLUtils.file.ObjToBinaryFile(ModuleDict, FilePath)
        return self
    def FromDict(self, ModuleDict: dict):
        self.config = ModuleDict["config"]
        self.param = ModuleDict["param"]
        for Name, SubModuleDict in ModuleDict["submodules"].items():
            self.AddSubModule(
                Name, dict_to_module(SubModuleDict)
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
            "submodules": self.get_submodule_dict(),
            "_class_path": self.get_class_path()
        }
    def get_class_path(self):
        return class_instance_from_class_path(self)
    def Build(self):
        for SubModule in self.submodules.values():
            SubModule.Build()
        return self

def dict_to_module(ModuleDict):
    assert isinstance(ModuleDict, dict)
    ModuleDict = Dict(ModuleDict)
    ClassPath = ModuleDict._class_path
    module = class_path_from_class_instance(ClassPath)
    assert isinstance(module, Module)
    module.FromDict(ModuleDict)
    return module

def file_to_module(FilePath):
    import _utils_file
    FilePath = _utils_file.file_exist(FilePath)
    ModuleDict = DLUtils.file.BinaryFileToObj(FilePath)
    assert isinstance(ModuleDict, dict)
    return dict_to_module(ModuleDict)

def module_to_dict(module: Module):
    return module.ToDict()
