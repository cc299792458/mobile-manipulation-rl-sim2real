import gym
import numpy as np
import torch
from concept.envs.vec_env.vec_env import SubprocVectorEnv


class HalfCheetahV2:

    def __init__(self, batch_size) -> None:
        
        def env_maker():
            env = gym.make("HalfCheetah-v2")
            print("made HalfCheetah-v2")
            return env
        
        self.dummy_env = env_maker()
        self.obs_shape = self.dummy_env.observation_space.shape
        self.act_dim = self.dummy_env.action_space.shape[0]
        self.dummy_env.close()

        self.vec_env = SubprocVectorEnv([env_maker for _ in range(batch_size)])
        self.episode_length = 1000

    def reset(self):
        obs = self.vec_env.reset()
        return torch.from_numpy(obs).float()
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        obs, reward, done, info = self.vec_env.step(action)
        return (
            torch.from_numpy(obs).float(),
            torch.from_numpy(reward).float().unsqueeze(-1),
            torch.zeros(done.shape).float().unsqueeze(-1),  # halfcheetah v2 is not episodic
            info  # list of dict
        )
    
    def render(self, mode="rgb_array"):
        return self.vec_env.render(mode=mode)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    batch_size = 2
    env = HalfCheetahV2(batch_size=batch_size)
    obs = env.reset()
    print(obs.shape, obs.dtype)
    action = torch.randn(batch_size, env.act_dim)
    obs, reward, done, info = env.step(action)

    print(obs.shape, obs.dtype)
    print(action.shape)
          
    print(reward, done, info)
    rendered_img = env.render()

    from concept.tools import animate
    # render the first image
    plt.imshow(rendered_img[0]) 
    plt.savefig("test_halfcheetah.png")

    renders = []
    obs = env.reset()
    for t in range(20):
        action = torch.randn(batch_size, env.act_dim)
        obs, reward, done, info = env.step(action)
        print(t, info)
        renders.append(env.render()[0])
    animate(renders, "test_halfcheetah.mp4", fps=30)

