import gym
import numpy as np
import torch
from concept.envs.vec_env.vec_env import SubprocVectorEnv


class TorchSubProcVecEnv:

    def __init__(self, env_maker, obs_shape, act_dim, batch_size, episode_length):
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.episode_length = episode_length
        self.vec_env = SubprocVectorEnv([env_maker for _ in range(batch_size)])

    def reset(self):
        obs_list = self.vec_env.reset()
        if isinstance(obs_list[0], dict):
            obs = {}
            for k in obs_list[0].keys():
                obs[k] = torch.from_numpy(np.stack([obs[k] for obs in obs_list])).float()
        else:
            obs = torch.from_numpy(np.stack(obs_list)).float()
        return obs
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs_list, reward, done, info = self.vec_env.step(action)
        if isinstance(obs_list[0], dict):
            obs = {}
            for k in obs_list[0].keys():
                obs[k] = torch.from_numpy(np.stack([obs[k] for obs in obs_list])).float()
        else:
            obs = torch.from_numpy(np.stack(obs_list)).float()
        return (
            obs,
            torch.from_numpy(reward).float().unsqueeze(-1),
            torch.zeros(done.shape).float().unsqueeze(-1),  # halfcheetah v2 is not episodic
            info  # list of dict
        )

    def render(self, mode="rgb_array"):
        return self.vec_env.render(mode=mode)


class TorchSingleProcVecEnv:

    def __init__(self, env_maker, obs_shape, obs_dtype, act_dim, batch_size, episode_length):
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.act_dim = act_dim
        self.episode_length = episode_length
        self.envs = [env_maker() for _ in range(batch_size)]

    def reset(self):
        obs_list = [env.reset() for env in self.envs]
        if isinstance(obs_list[0], dict):
            obs = {}
            for k in obs_list[0].keys():
                obs[k] = torch.from_numpy(np.stack([obs[k] for obs in obs_list])).to(dtype=self.obs_dtype)
        else:
            obs = torch.from_numpy(np.stack(obs_list)).to(dtype=self.obs_dtype)
        return obs
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(action[i])
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        
        if isinstance(obs_list[0], dict):
            obs = {}
            for k in obs_list[0].keys():
                obs[k] = torch.from_numpy(np.stack([obs[k] for obs in obs_list])).to(dtype=self.obs_dtype)
        else:
            obs = torch.from_numpy(np.stack(obs_list)).to(dtype=self.obs_dtype)

        reward = np.stack(reward_list)
        done = np.stack(done_list)
        info = info_list
        return (
            obs,
            torch.from_numpy(reward).float().unsqueeze(-1),
            torch.zeros(done.shape).float().unsqueeze(-1),  # halfcheetah v2 is not episodic
            info  # list of dict
        )

    def render(self, mode="rgb_array"):
        imgs = []
        for env in self.envs:
            imgs.append(env.render(mode=mode))
        return np.stack(imgs)