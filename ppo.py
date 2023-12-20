import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gridworld import CustomGridWorld
from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
import time

CONFIGS = {
    "tiny": {"size": 4, "slip_prob": 0.1, "max_len": 50},
    "small": {"size": 8, "slip_prob": 0.1, "max_len": 80},
    "medium": {"size": 10, "slip_prob": 0.1, "max_len": 100},
    "medium20": {"size": 20, "slip_prob": 0.1, "max_len": 150},
    "large": {"size": 50, "slip_prob": 0.1, "max_len": 200},
}

class MyEvalCB(BaseCallback):
    def __init__(self, verbose=0, n_steps=100, env_maker: Callable=Callable, seed=0):
        super().__init__(verbose)
        self.n_steps = n_steps
        self.last_time_trigger = 0
        self.eval_env = make_vec_env(env_maker, n_envs=1)
        self.accumulator = []
        self.seed = seed
        self.start = time.time()

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            mean_reward, _ = evaluate_policy(self.model, self.eval_env)
            self.accumulator.append(mean_reward)
            print(
                f"Seed: {self.seed}, time: {time.time() - self.start}, Step {self.num_timesteps}, Average Reward: {mean_reward}"
            )
        return True
    
def run_ppo(env_maker: Callable, train_steps:int, seed=0):
    vec_env = make_vec_env(env_maker, n_envs=4)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    callback = MyEvalCB(n_steps=1000, env_maker=env_maker, seed=seed)
    model.learn(
        total_timesteps=train_steps,
        log_interval=train_steps,
        callback=EveryNTimesteps(
            100,
            callback=callback,
        )
    )
    return model.policy, callback.accumulator