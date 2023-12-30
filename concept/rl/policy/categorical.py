import torch
import torch.nn as nn
import torch.distributions as thdist
import torch.nn.functional as thfunc


class CategoricalDist(thdist.Categorical):

    """ random variable is one-hot encoded """

    def log_prob(self, value):
        value = value.argmax(-1)
        return super().log_prob(value).unsqueeze(-1)
    
    def sample(self, sample_shape=torch.Size()):
        n_classes = self.logits.shape[-1]
        classes = super().sample(sample_shape)
        return thfunc.one_hot(classes, n_classes).float()


class CategoricalPolicy(nn.Module):

    def __init__(self, action_dim) -> None:
        super().__init__()
        self.action_dim = action_dim
    
    def extra_repr(self):
        return f"action_dim={self.action_dim}"

    def forward(self, logits):
        dist = CategoricalDist(logits=logits)
        sample = dist.sample()
        return dict(
            dist=dist,
            sample=sample,
            log_prob=dist.log_prob(sample)
        )



if __name__ == "__main__":
    policy = CategoricalPolicy(action_dim=3)
    
    logits = torch.zeros(4, 4)
    dist = policy(logits)
    sample = torch.eye(4)

    print(policy)
    print(dist)
