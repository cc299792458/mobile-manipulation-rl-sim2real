import tqdm
import datetime
from concept.tools import logger


class TrainerBase:

    def __init__(self, use_wandb=False, project_name="mobile_manipulation") -> None:
        self.use_wandb = use_wandb
        self.project_name = project_name

    def setup_logger(self):
        timenow = datetime.datetime.now().strftime("%y%m%d_%H%M%S_%f")
        logger_dir = self.log_dir
        exp_name = self.exp_name
        env_name = self.env_name
        seed = self.seed
        dir = f"{logger_dir}/{exp_name}/{env_name}/{seed}/{timenow}"
        format_strs = [
            "stdout", "log", "csv"
        ]
        if self.use_wandb:
            format_strs.append("wandb")
        logger.configure(
            dir=dir,
            project=self.project_name,
            config=dict(
                env_name=env_name,
                exp_name=exp_name,
                seed=seed
            ),
            format_strs=format_strs,
            group=f"{exp_name}",
            name=f"{env_name}/{seed}"
        )
        # with open(f"{logger.get_dir()}/config.yml", "w") as f:
        #     f.write(trainer_cfg.dump())
        logger.log("logger initialized")

    def start(self, steps):
        self.setup_logger()
        logger.log(f"[{datetime.datetime.now()}] trainer start")
        import traceback
        try:
            self.run(steps)
        except KeyboardInterrupt as e:
            logger.log("KeyboardInterrupt at main.py")
            traceback.print_exc()
        except Exception as e:
            logger.log("Catch exception at main.py")
            traceback.print_exc()
        finally:
            # logger.torch_save(self, "error.pt")
            # logger.log("Saved trainer to error.pt")
            pass

    def run(self, steps):

        n_epochs = steps // (self.env.episode_length)
        start, end = self.epoch_id, self.epoch_id + n_epochs

        for _ in range(start, end):

            if self.global_step % self.eval_every == 0 and self.global_step > self.warm_steps:
                self.evaluate()

            for _ in tqdm.trange(self.env.episode_length):
                self.train()
            
            logger.log(f"epoch {self.epoch_id}")
            logger.logkv("epoch", self.epoch_id)
            logger.logkv("global_steps", self.global_step)
            logger.logkv("total_n_traj", self.agent.replay_buffer.traj_count)
            
            logger.dumpkvs()
            self.epoch_id += 1
