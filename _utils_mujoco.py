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


# gym/gymnasium wrapping of mujoco
# from _utils_gym import VideoRecorder