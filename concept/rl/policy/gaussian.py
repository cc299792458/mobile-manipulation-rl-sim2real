import torch
import torch.nn as nn
import torch.nn.functional as thfunc
import torch.distributions as thdist
import math
from concept.rl.policy.std_manager import Std


class GaussianDist(thdist.Normal):

    def log_prob(self, value):
        return super().log_prob(value).sum(-1, keepdim=True)


class GaussianPolicy(nn.Module):

    def __init__(self, action_dim, std_manager:Std=None):
        super().__init__()
        self.std_manager = std_manager or Std(out_dim=action_dim)
        self.action_dim = action_dim
        self.in_dim = action_dim + self.std_manager.in_dim

    def extra_repr(self):
        return f"action_dim={self.action_dim}"
    
    def forward(self, x):
        assert x.shape[-1] == self.in_dim
        mu, std = x[..., :self.action_dim], self.std_manager(x[..., self.action_dim:])
        # sample and log_prob
        dist = GaussianDist(loc=mu, scale=std)
        u = dist.rsample()
        log_prob = dist.log_prob(u) - (2 * (math.log(2) - u - thfunc.softplus(-2 * u))).sum(-1, keepdim=True)
        return dict(dist=dist, sample=torch.tanh(u), log_prob=log_prob)


if __name__ == "__main__":
    policy = GaussianPolicy(action_dim=3)
    print(policy)

    x = torch.zeros(4, 6)
    out = policy(x)
    print(out)

    from concept.tools import unpack
    a, log_prob_a = unpack(out, "sample", "log_prob")
    print(a.shape, log_prob_a.shape)