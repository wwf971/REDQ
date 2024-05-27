import sys, os, pathlib
CurrentDirPath = os.path.dirname(os.path.realpath(__file__)) +"/"

ImportDirPath = [
    CurrentDirPath,
    CurrentDirPath + "/utils-python-global",
]
sys.path += ImportDirPath

# import DLUtils module
import time
print("Import DLUtils.", end=" ")
TimeStart = time.time()
import DLUtils
TimeEnd = time.time()
print("Finished. Time: %.3fs"%(TimeEnd-TimeStart))