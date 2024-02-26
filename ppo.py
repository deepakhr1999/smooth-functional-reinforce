"""Code for Proximal Policy Evaluation"""
from typing import Callable
import time
import torch
import numpy as np
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Convert one-hot feature vector for stable-baselines"""
    def __init__(self, observation_space: Discrete, features_dim: int):
        super().__init__(observation_space, features_dim)
        self.grid_size = int(pow(observation_space.n, 0.5))
        print("initializing custom extractor", self.grid_size)

    def forward(self, state: torch.Tensor):
        """Forward pass"""
        state = state.argmax(-1)
        alpha, beta = state // self.grid_size, state % self.grid_size
        features = [alpha, alpha**2, beta, beta**2, (alpha + beta) % 2, state]
        return torch.stack(features, dim=-1).float()

class MyEvalCB(BaseCallback):
    """Callback for evaulating and accumulating total reward of iterates"""
    def __init__(self, verbose=0, n_steps=100, env_maker: Callable = Callable, seed=0):
        super().__init__(verbose)
        self.n_steps = n_steps
        self.last_time_trigger = 0
        self.eval_env = env_maker()
        self.accumulator = []
        self.seed = seed
        self.start = time.time()
        self.gamma = 0.99
        with open("ppo.txt", "+a") as file:
            print(f"Seed,Time,Episode,Reward", file=file)

    def evaluate(self) -> float:
        state, _ = self.eval_env.reset()
        done = False
        tot = 0
        for tdx in range(100_000_000):
            action = self.model.predict(np.array([state]))[0][0]
            state, reward, done, *_ = self.eval_env.step(action)
            tot += (self.gamma**tdx) * reward
            if done:
                break
        return tot

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps

            mean_reward = self.evaluate()
            self.accumulator.append(mean_reward)
            print(
                f"Seed: {self.seed}, time: {time.time() - self.start}, "
                f"Step {self.num_timesteps}, Average Reward: {mean_reward}"
            )
            with open("ppo.txt", "+a") as file:
                print(f"{self.seed},{time.time() - self.start},{self.num_timesteps},{mean_reward}", file=file)
        return True

def run_ppo(algo:str, env_maker: Callable, train_steps: int, seed=0):
    """Driver code for PPO"""
    vec_env = make_vec_env(env_maker, n_envs=4)
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {'features_dim': 6},
    }
    module = PPO if algo == 'ppo' else A2C
    model = module("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs, device="cpu")

    callback = MyEvalCB(n_steps=100, env_maker=env_maker, seed=seed)
    print("Running for", train_steps, "steps")
    model.learn(
        total_timesteps=train_steps,
        log_interval=train_steps,
        callback=callback
    )
    return model.policy, callback.accumulator
