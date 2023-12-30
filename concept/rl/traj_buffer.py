import torch
import torch.nn as nn


class TrajBuffer:

    def __init__(self, episode_length, max_size, obs_shape, act_dim) -> None:
        self.buffer_s = torch.zeros(episode_length+1, max_size, *obs_shape)
        self.buffer_a = torch.zeros(episode_length, max_size, act_dim)
        self.buffer_r = torch.zeros(episode_length, max_size, 1)
        self.buffer_d = torch.zeros(episode_length, max_size, 1)
        self.episode_length = episode_length
        self.traj_count = 0
        self.timestep_count = 0
        self.max_size = max_size

    def __repr__(self) -> str:
        return f'TrajBuffer(traj_count={self.traj_count}, max_size={self.max_size})'

    def add(self, s, a, r, d):
        traj_idx = self.traj_count % self.max_size
        self.buffer_s[self.timestep_count, traj_idx] = s
        self.buffer_a[self.timestep_count, traj_idx] = a
        self.buffer_r[self.timestep_count, traj_idx] = r
        self.buffer_d[self.timestep_count, traj_idx] = d
        self.timestep_count += 1

    def add_batch(self, s, a, r, d):
        batch_size = s.shape[0]
        traj_idxs = torch.arange(self.traj_count, self.traj_count + batch_size) % self.max_size
        self.buffer_s[self.timestep_count, traj_idxs] = s
        self.buffer_a[self.timestep_count, traj_idxs] = a
        self.buffer_r[self.timestep_count, traj_idxs] = r
        self.buffer_d[self.timestep_count, traj_idxs] = d
        self.timestep_count += 1

    def add_last(self, s):
        traj_idx = self.traj_count % self.max_size
        self.buffer_s[self.timestep_count, traj_idx] = s
        self.traj_count += 1
        self.timestep_count = 0

    def add_batch_last(self, s):
        batch_size = s.shape[0]
        traj_idxs = torch.arange(self.traj_count, self.traj_count + batch_size) % self.max_size
        self.buffer_s[self.timestep_count, traj_idxs] = s
        self.traj_count += batch_size
        self.timestep_count = 0

    def sample_traj_batch(self, batch_size):
        idxs = torch.randint(0, min(self.traj_count, self.max_size), size=(min(self.traj_count, batch_size),))
        return (
            self.buffer_s[:, idxs, ...], 
            self.buffer_a[:, idxs, ...], 
            self.buffer_r[:, idxs, ...], 
            self.buffer_d[:, idxs, ...]
        )
    
    def sample_transition_batch(self, batch_size):
        timestep_idxs = torch.randint(0, self.episode_length, size=(batch_size,))
        idxs = torch.randint(0, min(self.traj_count, self.max_size), size=(batch_size,))
        return (
            self.buffer_s[timestep_idxs, idxs, ...], 
            self.buffer_a[timestep_idxs, idxs, ...], 
            self.buffer_r[timestep_idxs, idxs, ...], 
            self.buffer_d[timestep_idxs, idxs, ...],
            self.buffer_s[timestep_idxs+1, idxs, ...]
        )


if __name__ == "__main__":
    buffer = TrajBuffer(10, 10, (2,), 2)
    for i in range(20):
        print(i)
        for t in range(10):
            buffer.add_batch(torch.ones(10, 2) * i, torch.ones(10, 2) * i, torch.ones(10, 1) * i, torch.ones(10, 1) * i)
        buffer.add_batch_last(torch.ones(10, 2) * i)

    print(buffer.sample_transition_batch(10)[0].shape)