

from .gym_video_recorder import VideoRecorder

def GetEnv(EnvName, Seed=None, RenderMode="rgb_array"):
    """
    RenderMode: 决定了env.render()的返回值.
        "rgb_array": 返回值
    
    """
    import gym
    Env = gym.make(EnvName, render_mode=RenderMode)
    
    if Seed is not None:
        SetSeedForEnv(Env, Seed)
    return Env
GetSingleEnv = GetEnv

def GetVectorEnv(EnvName, Num: int, Seed, ):
    """
    parallel env.
    """


    assert len(Seed) == Num


    return Env


def SetSeedForEnv(Env, Seed):
    return

def GetActionSpaceSize(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


def GetStateSpaceSize():
    """
    observation is a flattened 1-d vector.
    """

    
    return

from _utils_mujoco import SetMujocoRenderBackend