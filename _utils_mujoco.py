from __future__ import annotations
import xml
from _utils import Dict


def SetMujocoRenderBackend(Backend="egl"):
    import os
    # os.environ["MJKEY_PATH"] = "/home/wwf/.mujoco/mjkey.txt"
    if Backend in ["egl", "EGL"]:
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    elif Backend in ["osmesa", "OSMESA"]:
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    elif Backend in ["glfw", "GLFW", "gl", "GL"]:
        os.environ["MUJOCO_GL"] = "glfw"
        os.environ["PYOPENGL_PLATFORM"] = "glfw"
    elif Backend in ["wgl", "WGL"]:
        os.environ["MUJOCO_GL"] = "wgl"
        os.environ["PYOPENGL_PLATFORM"] = "wgl"
    else:
        raise Exception()
    return

def _GenerateCustomXMLFile(xml_file, xml_file_custom, time_step):
    import xml.etree.ElementTree as ET
    XMLObj = ET.parse(xml_file)    
    root = XMLObj.getroot()
    option = root.find("option")
    if not ("." in time_step): # integer millisecond
        time_step_ms = int(time_step)
        option.set("timestep", "%.3f"%(time_step_ms/1000.0))
    else:
        raise NotImplementedError
    XMLObj.write(xml_file_custom)

def GenerateCustomXMLFile(xml_file, xml_file_custom, time_step, frame_skip):
    if isinstance(time_step, int):
        time_step = str(time_step)
    if isinstance(frame_skip, int):
        frame_skip = str(frame_skip)

    import DLUtils
    xml_file = DLUtils.CheckFileExists(xml_file)
    xml_file_custom = DLUtils.EnsureFileDir(xml_file_custom)
    assert isinstance(time_step, str)
    assert isinstance(frame_skip, str)

    _GenerateCustomXMLFile(xml_file, xml_file_custom, time_step)
    frame_skip_int = int(frame_skip)
    return frame_skip_int

if __name__ == "__main__":
    import sys, os, pathlib
    DirPathCurrent = os.path.dirname(os.path.realpath(__file__)) + "/"
    DirPathParent = pathlib.Path(DirPathCurrent).parent.absolute().__str__() + "/"
    DirPathGrandParent = pathlib.Path(DirPathParent).parent.absolute().__str__() + "/"
    DirPathGreatGrandParent = pathlib.Path(DirPathGrandParent).parent.absolute().__str__() + "/"
    sys.path += [
        DirPathCurrent, DirPathParent, DirPathGrandParent, DirPathGreatGrandParent
    ]
    from _utils_import import DLUtils
    xml_file_path = "/home/wwf/Project/paper/RFC-origin/khrylib/assets/mujoco_models/mocap_v2.xml"
    import xml.etree.ElementTree as ET
    XMLObj = ET.parse(xml_file_path)    
    root = XMLObj.getroot()
    option = root.find("option")
    
    a = 1
# gym/gymnasium wrapping of mujoco
# from _utils_gym import VideoRecorder