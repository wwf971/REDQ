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

def class_instance_from_class_path(Instance):
    cls = Instance.__class__
    module = cls.__module__
    qualname = cls.__qualname__
    return f"{module}.{qualname}"

def class_path_from_class_instance(ClassPath: str, **KwArgs):
    import importlib
    # split the class_path into module path and class name
    ModulePath, ClassName = ClassPath.rsplit('.', 1)
    # import the module
    module = importlib.import_module(ModulePath)
    # get the class
    cls = getattr(module, ClassName)
    # create an instance of the class
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
