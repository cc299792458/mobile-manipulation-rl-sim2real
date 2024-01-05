import tqdm
from concept.tools import logger, animate
from concept.rl.agent.sac.state.sac import SAC
from concept.rl.agent.trainer_base import TrainerBase
from concept.envs import make


class Trainer(TrainerBase):

    def __init__(
            self, env_name, n_env, seed, batch_size=512,
            buffer_size=int(8e5), eval_every=10000, warm_steps=10000, 
            device="cuda", log_dir="./workspace", exp_name="sac/state"
        ) -> None:
        super().__init__()
        # basics
        self.log_dir, self.exp_name = log_dir, exp_name
        self.seed = seed
        self.device = device
        # make env
        self.env_name = env_name
        self.env = make(self.env_name, batch_size=n_env)
        self.n_env = n_env
        # agent
        self.obs_dim, self.act_dim = self.env.obs_shape[0], self.env.act_dim
        self.agent = SAC(
            self.obs_dim, self.act_dim, 
            episode_length=self.env.episode_length, 
            buffer_size=buffer_size // self.env.episode_length, 
            device=device)
        self.batch_size = batch_size
        self.warm_steps = warm_steps
        # episode counter
        self.ep_timestep = 0
        self.ep_return = 0
        # global counter
        self.epoch_id = 0
        self.global_step = 0
        self.global_n_samples = 0
        self.eval_every = eval_every
        self.obs = self.env.reset()

    def train(self):
        obs = self.obs
        action = self.agent.act(obs, mode="train").cpu()
        next_obs, reward, done, info = self.env.step(action)
        self.agent.store_transition(obs, action, reward, done)

        self.ep_timestep += 1
        self.ep_return += reward
        self.global_step += 1
        self.global_n_samples += self.n_env
        
        # episode end
        if self.ep_timestep == self.env.episode_length:
            self.agent.store_last_obs(next_obs)
            logger.logkv("train_ep_return", self.ep_return)
            # reset
            self.obs = self.env.reset()
            self.ep_timestep = 0
            self.ep_return = 0
        else:
            self.obs = next_obs
        
        # update agent
        if self.agent.replay_buffer.traj_count >= 10:
            agent_update_info = self.agent.update(self.batch_size)
            logger.logkvs_min_avg_max(agent_update_info)

    def evaluate(self):
        obs = self.env.reset()
        ep_return = 0
        video_frames = []
        for ep_timestep in range(self.env.episode_length):
            action = self.agent.act(obs, mode="eval").cpu()
            obs, reward, done, info = self.env.step(action)
            ep_return += reward
            # img = self.env.render(mode="rgb_array")
            # if len(img.shape) > 3:
            #     img = img[0]
            # video_frames.append(img)
        logger.logkv("eval_epoch", self.epoch_id)
        logger.logkv("eval_ep_return", ep_return)
        # logger.animate(video_frames, f="eval.mp4", fps=30)
        self.obs = self.env.reset()


if __name__ == "__main__":
    trainer = Trainer("HalfCheetahV2", n_env=1, seed=0)
    trainer.start(steps=int(1e7))
