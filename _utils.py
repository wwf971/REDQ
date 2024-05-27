
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
    def __init__(self, _dict: dict = None, **Dict):
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
            child = Dict()
            setattr(self, key, child)
            return child
            # raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
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


