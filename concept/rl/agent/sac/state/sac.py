import torch
import torch.nn as nn
from concept.tools import unpack
from concept.rl.policy import GaussianPolicy
from concept.rl.traj_buffer import TrajBuffer
from concept.rl.utils import ema


class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, init_log_alpha=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim * 2)
        )
        self.action_head = GaussianPolicy(act_dim)
        self.log_alpha = torch.nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.float32))

    def alpha(self):
        return self.log_alpha.exp()
    
    def forward(self, obs):
        x = self.net(obs)
        return self.action_head(x)


class Critic(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class SAC:

    def __init__(
            self, 
            obs_dim, act_dim, 
            episode_length, buffer_size, device,
            gamma=0.99
        ) -> None:
        self.device = device
        self.gamma = gamma
        # actor
        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.target_entropy = -act_dim
        # critics
        self.critic1 = Critic(obs_dim, act_dim).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim).to(self.device)
        # target networks
        self.target1 = Critic(obs_dim, act_dim).to(self.device)
        self.target2 = Critic(obs_dim, act_dim).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam([*self.critic1.parameters(), *self.critic2.parameters()], lr=3e-4)
        self.replay_buffer = TrajBuffer(episode_length, buffer_size, obs_shape=(obs_dim, ), act_dim=act_dim)

    @torch.no_grad()
    def act(self, obs, mode):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        dist, sample = unpack(self.actor(obs), "dist", "sample")
        if mode == "train":
            return sample
        elif mode == "eval": 
            return torch.tanh(dist.mean)
        
    def store_transition(self, obs, a, r, done):
        self.replay_buffer.add_batch(obs, a, r, done)

    def store_last_obs(self, obs):
        self.replay_buffer.add_batch_last(obs)
    
    def sample_batch(self, batch_size):
        s, a, r, d, s_next = self.replay_buffer.sample_transition_batch(batch_size)
        return s.to(self.device), a.to(self.device), r.to(self.device), d.to(self.device), s_next.to(self.device)

    def update_actor(self, batch_size=512):
        # sample batch
        s, _, r, d, s_next = self.sample_batch(batch_size)
        # actor loss
        dist, a, log_prob_a = unpack(self.actor(s), "dist", "sample", "log_prob")
        q1, q2 = self.critic1(s, a), self.critic2(s, a)
        v_elbo = torch.min(q1, q2) - self.actor.alpha().detach() * log_prob_a
        policy_entropy = -log_prob_a.detach().mean()
        alpha_loss = self.actor.alpha() * (policy_entropy - self.target_entropy)
        actor_loss = -v_elbo.mean() + alpha_loss
        # update
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return dict(
            actor_loss=actor_loss.item(),
            actor_v_elbo=v_elbo,
            actor_mu=dist.loc, 
            actor_std=dist.scale, 
            actor_entropy=policy_entropy.item(),
            actor_alpha=self.actor.alpha().item()
        )

    def update_critic(self, batch_size=512):
        # sample batch
        s, a, r, d, s_next = self.sample_batch(batch_size)
        # critic loss
        with torch.no_grad():
            a_next, log_prob_a_next = unpack(self.actor(s_next), "sample", "log_prob")
            q1_next, q2_next = self.target1(s_next, a_next), self.target2(s_next, a_next)
            q_next = torch.min(q1_next, q2_next) - self.actor.alpha().detach() * log_prob_a_next
            q_target = r + self.gamma * (1 - d) * q_next
        q1, q2 = self.critic1(s, a), self.critic2(s, a)
        critic_loss = (q1 - q_target).pow(2).mean() + (q2 - q_target).pow(2).mean()
        # update
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return dict(
            critic_loss=critic_loss.item(),
            critic_q1=q1,
            critic_q2=q2,
        )

    def update(self, batch_size=512):
        actor_update_info = self.update_critic(batch_size)
        critic_update_info = self.update_actor(batch_size)
        ema(self.critic1, self.target1, 0.005)
        ema(self.critic2, self.target2, 0.005)
        return dict(**actor_update_info, **critic_update_info)