"""Code for Proximal Policy Evaluation"""
from typing import Callable
import time
import torch
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
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
                f"Seed: {self.seed}, time: {time.time() - self.start}, "
                f"Step {self.num_timesteps}, Average Reward: {mean_reward}"
            )
        return True


def run_ppo(env_maker: Callable, train_steps: int, seed=0):
    """Driver code for PPO"""
    vec_env = make_vec_env(env_maker, n_envs=4)
    policy_kwargs = {
        "features_extractor_class": CustomFeatureExtractor,
        "features_extractor_kwargs": {'features_dim': 6},
    }
    model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)
    # model = PPO("MlpPolicy", vec_env, verbose=1)
    callback = MyEvalCB(n_steps=1000, env_maker=env_maker, seed=seed)
    model.learn(
        total_timesteps=train_steps,
        log_interval=train_steps,
        callback=EveryNTimesteps(
            100,
            callback=callback,
        ),
    )
    return model.policy, callback.accumulator
