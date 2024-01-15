from tqdm import tqdm
import gymnasium as gym
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from concept.envs import PickCube_v0


env = TimeLimit(PickCube_v0(only_arm=True), max_episode_steps=200)

# 初始化 PPO 模型
model = PPO("MlpPolicy", env, verbose=1)

model.load("ppo_pickcube")
eval_env = TimeLimit(PickCube_v0(only_arm=True, render_mode='human'))
obs = eval_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    # env.render()
    if dones:
        obs = env.reset()

total_timesteps = 1000000  # 总训练步数
eval_steps = 100000  # 每隔多少步进行评估
n_eval_episodes = 10  # 评估时运行多少个回合
model_save_path = "./models/ppo_pickcube"  # 模型保存路径

# 创建一个 tqdm 进度条
with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
    for step in range(0, total_timesteps, eval_steps):
        # 训练模型
        model.learn(total_timesteps=eval_steps, reset_num_timesteps=False)
        pbar.update(eval_steps)

        # 评估模型
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
        print(f"Steps: {step + eval_steps}, Mean Reward: {mean_reward}, Std Reward: {std_reward}")

        # 保存模型
        model.save(f"{model_save_path}_{step + eval_steps}")

# 关闭环境
env.close()
eval_env.close()

