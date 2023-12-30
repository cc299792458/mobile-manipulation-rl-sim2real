from concept.rl.agent.sac.state.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer("HalfCheetahV2", n_env=1, seed=0)
    trainer.start(steps=int(1e7))