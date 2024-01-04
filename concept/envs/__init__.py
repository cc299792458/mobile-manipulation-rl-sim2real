from .mujoco import *
from .sapien import *

def make(env_name, *args, **kwargs):
    return eval(env_name)(*args, **kwargs)
