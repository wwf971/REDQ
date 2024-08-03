# import mujoco_py as soon as possible. mujoco_py is very buggy.
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import mujoco_py

import sys, os, pathlib
DirPathCurrent = os.path.dirname(os.path.realpath(__file__)) + "/"
DirPathParent = pathlib.Path(DirPathCurrent).parent.absolute().__str__() + "/"
DirPathGrandParent = pathlib.Path(DirPathParent).parent.absolute().__str__() + "/"
sys.path += [
    DirPathCurrent, DirPathParent, DirPathGrandParent
]
DirPathProject = DirPathCurrent

def ListAllFileName(DirPath):
    # assert ExistsDir(DirPath), "Non-existing DirPath: %s"%DirPath
    assert os.path.isdir(DirPath), "Not a Dir: %s"%DirPath
    Items = os.listdir(DirPath)
    Files = []

    for Item in Items:
        if os.path.isfile(os.path.join(DirPath, Item)):
            Files.append(Item)
    return Files

def ListAllFilePath(DirPath):
    if not DirPath.endswith("/") or DirPath.endswith("\\"):
        DirPath += "/"
    FileNameList = ListAllFileName(DirPath)
    return [DirPath + FileName for FileName in FileNameList]