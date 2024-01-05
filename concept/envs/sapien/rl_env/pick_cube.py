import gym
from gym import spaces

import sapien.core as sapien
from sapien.core import Pose

import numpy as np
from concept.envs.sapien.simulation_env import SimEnv

import torch
from concept.envs.vec_env.vec_env import SubprocVectorEnv


class PickCube:
    def __init__(self, batch_size, version='v0') -> None:
        
        def env_maker():
            if version == 'v0':
                env = PickCube_v0()
                print("made PickCube-v0")
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
        # return self.dummy_env.render(mode=mode)
        return self.vec_env.render(mode=mode)
    

# class PickCubeV0:

#     def __init__(self, batch_size) -> None:
#         assert batch_size ==1
#         self.dummy_env = PickCube()
#         self.obs_shape = self.dummy_env.observation_space.shape
#         self.act_dim = self.dummy_env.action_space.shape[0]
#         self.episode_length = 1000

#     def reset(self):
#         obs = self.dummy_env.reset()
#         return obs[None, ...]
    
#     def step(self, action):
#         if isinstance(action, torch.Tensor):
#             action = action.cpu().numpy()
#         obs, reward, done, info = self.dummy_env.step(action)
#         return (
#             torch.from_numpy(obs).float(),
#             torch.from_numpy(reward).float().unsqueeze(-1),
#             torch.zeros(done.shape).float().unsqueeze(-1),  # halfcheetah v2 is not episodic
#             info  # list of dict
#         )
    
#     def render(self, mode="rgb_array"):
#         return self.dummy_env.render(mode=mode)

class PickCube_v0(SimEnv):
    def __init__(self, 
                 **kwargs):
        self.cube_half_size = 0.015
        self.goal_thresh = 0.025
        SimEnv.__init__(self, **kwargs)

    def initialize_task(self):
        self.cube = self._build_cube(np.array([self.cube_half_size, self.cube_half_size, self.cube_half_size]))
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def reset_task(self):
        self.cube.set_pose(Pose(p=np.array([0.3, 0.0, self.cube_half_size])))
        self.goal_site.set_pose(Pose(p=np.array([0.3, 0.0, 0.1])))

    def get_obs_task(self):
        obs_task = np.hstack([self._vectorize_pose(self.cube.pose), self._vectorize_pose(self.goal_site.pose)])
        return obs_task

    def evaluate(self, **kwargs):
        is_grasped = self.check_grasp(self.cube)
        # is_robot_static = self.check_robot_static()
        reach_pos = self.cube.get_pose().p - self.goal_site.get_pose().p
        reach_dis = np.linalg.norm(reach_pos, ord=2)
        return dict(
            is_grasped=is_grasped,
            success=bool(reach_dis <= 0.025),
        )

    def get_reward(self, obs, action, info):
        reward = 0
        ##### Reach reward #####
        reach_pos = self.tcp_link.get_pose().p - self.cube.get_pose().p
        reach_dis = np.linalg.norm(reach_pos, ord=2)
        reach_reward = 1 - np.tanh(5 * reach_dis)
        reward += reach_reward

        ##### Grasp reward #####
        is_grasped = self.check_grasp(self.cube)
        if is_grasped:
            reward += 1
            ##### Place reward #####
            reach_pos = self.cube.get_pose().p - self.goal_site.get_pose().p
            reach_dis = np.linalg.norm(reach_pos, ord=2)
            reach_reward = 1 - np.tanh(5 * reach_dis)
            reward += reach_reward
        
        reward = reward/3 - 1   # use negtive reward

        return reward

    def get_done(self, obs, info):
        return bool(info["success"])
    
    def render(self, mode="console"):
        if mode in ['human', 'rgb_array']:
            self.goal_site.unhide_visual()
            ret = super().render(mode=mode)
            self.goal_site.hide_visual()
        else:
            ret = super().render(mode=mode)
        return ret

    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        if render_material is None:
            render_material = self.renderer.create_material()
            render_material.set_base_color(np.hstack([color, 1.0]))
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size, density=100)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE: Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere

    def _vectorize_pose(self, pose: sapien.Pose):
        return np.hstack([pose.p, pose.q])

if __name__ == '__main__':
    env = PickCube(render_mode='human')
    while True:
        env.step(np.array([0, 0, 0, 0, 0, 0, 0]))