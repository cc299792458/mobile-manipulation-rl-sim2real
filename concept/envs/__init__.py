from .mujoco import *

def make(env_name, *args, **kwargs):
    return eval(env_name)(*args, **kwargs)
