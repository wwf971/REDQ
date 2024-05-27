from _utils_import import DLUtils

def CleanEmptySubDir(BaseDirPath, OutPipe=None):
    """
    
    """
    BaseDirPath = DLUtils.file.CheckDirExists(BaseDirPath)
    DirNameList = DLUtils.file.ListAllDirNames(BaseDirPath)
    for DirName in DirNameList:
        DirPath = BaseDirPath + DirName
        if DLUtils.file.IsEmptyDir(DirPath):
            DLUtils.file.RemoveDir(DirPath)
            assert not DLUtils.file.ExistsDir(DirPath)
            if OutPipe is not None:
                OutPipe.print("Removed empty dir.")
                with OutPipe.IncreasedIndent():
                    OutPipe.print("DirPath: %s"%DirPath)
    return

from DLUtils.file import (
    GetDirPathWithSameDirNameAsFile,
    GetFilePathWithoutSuffix,
    DeleteFileIfExists
)

def GetProjectDirPath(FilePath):
    """
    
    """
    FilePathNoSuffix, Suffix = DLUtils.file.SeparateFileNameSuffix(FilePath)
    BaseDirPath = FilePathNoSuffix + "/"
    BaseDirPath = DLUtils.EnsureDir(BaseDirPath)
    return BaseDirPath

def GetExperimentDirPath(BaseDirPath):
    """
    path of a folder that stores data of one experiment instance
    """
    from datetime import datetime
    ExperimentDirPath = GetDirPathWithSameDirNameAsFile(BaseDirPath) + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    return ExperimentDirPath

