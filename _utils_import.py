"""
check import:
    python -X importtime _lib_import.py
"""

import sys, os, pathlib
dir_path_current = os.path.dirname(os.path.realpath(__file__)) + "/"
dir_path_parent = pathlib.Path(dir_path_current).parent.absolute().__str__() + "/"
dir_path_grandparent = pathlib.Path(dir_path_parent).parent.absolute().__str__() + "/"
sys.path += [
    dir_path_current,
    dir_path_current + "/utils-python-global",
]

LAZY_IMPORT = True

import time
import importlib
def ImportModule(SubModuleStr: str):

    SubModule = importlib.import_module(SubModuleStr)
    return SubModule
def ImportSubModuleFromPathList(SubModulePathList: list):
    assert len(SubModulePathList) > 1
    SubModule = importlib.import_module(".".join(SubModulePathList))
    return SubModule
def LazyImport(ModuleStr, FuncAfterImport=None):
    """
    import a => LazyImport("a") # lazy import module
    import a.b => LazyImport("a.b") # lazy import submodule
    module, submodule
    """
    return _LazyImport(ModuleStr, ImportMode="Module", FuncAfterImport=FuncAfterImport)
def FromImport(ModuleStr: str, VarStr: str):
    """
    from a.b import c as d => d = FromImport("a.b", "c")
    """
    SubModule = ImportModule(ModuleStr)
    Var = getattr(SubModule, VarStr)
    return Var
def LazyFromImport(ModuleStr: str, VarStr: str):
    """
    from a.b import c as d => d = LazyFromImport("a.b", "c")
    """
    return _LazyImport(ModuleStr, VarStr, ImportMode="From")

class _LazyImport(object):
    def __init__(self,
        ModuleName: str,
        VarName: str = None, 
        RaiseOnImportFailure: bool=True,
        ImportMode="Import", # "Import", "FromImport"
        FuncAfterImport=None
    ):
        self.ModuleName = ModuleName
        self.RaiseOnImportFailure = RaiseOnImportFailure
        self.IsModuleImported = False
        
        self.SubModulePathList = ModuleName.split(".")
        if len(self.SubModulePathList) == 1:
            self.ImportModule = self._ImportModule
        elif len(self.SubModulePathList) > 1:
            self.ImportModule = self._ImportSubModule
        else:
            raise Exception()
        if ImportMode in ["Module", "SubModule", "Import"]:
            self.ImportMode = "Import"
            self.Import = self._ImportModule
            self.IsFromImport = False
        elif ImportMode in ["FromModuleImport", "FromSubModuleImport", "From", "FromImport"]:
            self.ImportMode = "FromImport"
            self.VarName = VarName
            self.Import = self._FromImport
            self.IsFromImport = True
        else:
            raise Exception()
        self.FuncAfterImport = FuncAfterImport
    def _ImportModule(self):
        # assert not self.IsModuleImported
        import importlib
        try:
            Module = importlib.import_module(self.ModuleName)
        except Exception:
            if self.RaiseOnImportFailure:
                raise Exception(f"Failed to import module: %s"%{self.ModuleName})
            else:
                self.IsModuleImported = False
                Module = "DLUtils.LazyImport: ImportFailure"
                return None
        else:
            self.IsModuleImported = True
        self.Module = Module
        if self.FuncAfterImport is not None:
            self.FuncAfterImport(Module)
        return Module
    def _ImportSubModule(self):
        Module = ImportSubModuleFromPathList(self.SubModulePathList)
        self.Module = Module
        if self.FuncAfterImport is not None:
            self.FuncAfterImport(Module)
        return Module
    def _FromImport(self):
        Module = self.ImportModule()
        Var = getattr(Module, self.VarName)
        self.Var = Var
        return Var
    def __call__(self, *List, **Dict):
        """
        self.Import = _FromImport
        """
        # assert self.ImportMode == "FromImport"
        if not self.IsModuleImported:
            self.Import()    
        return self.Var(*List, **Dict)
    def __getattr__(self, Name):
        # print(f"LazyImport:{name}")
        if not self.IsModuleImported:
            self.Import()
        if self.IsFromImport:
            return getattr(self.Var, Name)
        else:
            Var = getattr(self.Module, Name) # submodule, method, variable etc
            setattr(self, Name, Var)
            return Var
    def GetModuleAndIsModuleImported(self):
        return self.Module, self.IsModuleImported

"""import DLUtils module"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import DLUtils
else:
    if LAZY_IMPORT:
        # lazy import
        print("Import DLUtils(Lazy).", end=" ")
        TimeStart = time.time()
        DLUtils = LazyImport("DLUtils")
        TimeEnd = time.time()
        TimeImport = TimeEnd - TimeStart
        print("Finished. Time: %.3fs"%(TimeImport))
    else:
        # direct import
        print("Import DLUtils.", end=" ")
        TimeStart = time.time()
        import DLUtils
        TimeEnd = time.time()
        TimeImport = TimeEnd - TimeStart
        print("Finished. Time: %.3fs"%(TimeImport))

import time
print("Import DLUtils.", end=" ")
TimeStart = time.time()
import DLUtils
TimeEnd = time.time()
print("Finished. Time: %.3fs"%(TimeEnd-TimeStart))

LazyNumpy = LazyImport("numpy")
def GetLazyNumpy():
    return LazyNumpy
LazyScipy = LazyImport("scipy")
def GetLazyScipy():
    return LazyScipy
LazyTorch = LazyImport("torch")
def GetLazyTorch():
    return LazyTorch
LazyPsUtil = LazyImport("psutil")
def GetLazyPsUtil():
    return LazyPsUtil
LazyMatplotlib = LazyImport("matplotlib")
def GetLazyMatplotlib():
    return LazyMatplotlib
LazyPlt = LazyFromImport("matplotlib", "plt")
def GetLazyPlt():
    return LazyPlt
LazyPILImage = LazyImport("PIL.Image")
def GetLazyPILImage():
    return LazyPILImage