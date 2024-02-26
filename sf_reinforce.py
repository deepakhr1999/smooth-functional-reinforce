"""Smooth Functional REINFORCE algorithm"""

import time
from argparse import Namespace
from typing import NamedTuple
import gym
import torch
from torch.nn import Module


class SimulationResult(NamedTuple):
    """Simulation returns these values together"""

    avg_reward: float
    update_factor: float
    perts: list[torch.Tensor]


def get_delta(episode: int, delta_pow: float, const_delta: None | float) -> float:
    """Pertubation schedule according to cmd args"""
    if const_delta is None:
        # run with values such as .15, .25, .35, .45
        return (2e-5 / (1 + episode * 2e-5)) ** delta_pow
    return const_delta


def get_alpha(episode: int, start: float = 2e-6) -> float:
    """Learning rate schedule"""
    return start / (1 + episode * 2e-5)


class SFPolicy:
    """Smooth Functional REINFORCE"""

    def __init__(self, env: gym.Env, policy: Module, args: Namespace):
        self.env = env
        self.policy = policy
        self.args = args
        self.old_params = [t.clone() for t in policy.parameters()]

    def perturb_policy(self, episode: int) -> list:
        """Perturbs the weights of policy with a normal distribution"""
        delta = get_delta(episode, self.args.delta_pow, self.args.const_delta)
        perts = [torch.randn_like(t.data) for t in self.policy.parameters()]
        self.old_params = [t.clone() for t in self.policy.parameters()]
        for t, d in zip(self.policy.parameters(), perts):
            t.data += delta * d.data
        return perts

    def rollout(self, gamma: float = 0.99) -> float:
        """Evaluates the policy by running an agent in the environment"""
        state, _ = self.env.reset()
        tdx = 0
        returns = 0
        while True:
            probs = self.policy(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            state, reward, done, *_ = self.env.step(action.item())
            returns += gamma**tdx * reward
            tdx += 1
            if done:
                break
        return returns

    def revert_weights(self):
        """Revert weights of policy to old parameters"""
        for t, old_t in zip(self.policy.parameters(), self.old_params):
            t.data = old_t.data.clone()

    def perturb_for_two_sided(self, episode: int, perts: list[torch.Tensor]):
        """For two-sided SFR, get the old - pert policy"""
        delta = get_delta(episode, self.args.delta_pow, self.args.const_delta)
        for new_param, pert in zip(self.policy.parameters(), perts):
            new_param.data -= 2 * delta * pert.data

    def simulate(self, episode: int, num_trials: int):
        """Perturb and simulate policy in the environment"""
        perts = self.perturb_policy(episode)

        # simulate for num_trials
        avg_reward_plus = sum(self.rollout() for _ in range(num_trials)) / num_trials

        if not self.args.algo.startswith("two_sided"):
            self.revert_weights()
            return SimulationResult(
                avg_reward=avg_reward_plus,
                update_factor=avg_reward_plus,
                perts=perts,
            )

        # perturb for minus and get other measurement
        self.perturb_for_two_sided(episode, perts)
        avg_reward_minus = sum(self.rollout() for _ in range(num_trials)) / num_trials

        self.revert_weights()
        return SimulationResult(
            avg_reward=(avg_reward_plus + avg_reward_minus) / 2,
            update_factor=(avg_reward_plus - avg_reward_minus) / 2,
            perts=perts,
        )

    def update_weights(
        self,
        episode: int,
        avg_reward: float,
        perts: list[torch.Tensor],
    ):
        """Gradient update according to the paper"""
        norm = None
        if self.args.grad_norm < 1e9:
            norm = 0
            for pert in perts:
                update_factor = avg_reward * pert
                norm += (update_factor**2).sum().cpu().item()
            norm = norm**0.5

        for t, pert in zip(self.policy.parameters(), perts):
            update_factor = avg_reward * pert
            if self.args.sign:
                # signed udpate
                sign = 2 * (update_factor > 0) - 1
                t += get_alpha(episode, self.args.alpha) * sign
            else:
                # clamp, norm and update
                update = (
                    update_factor
                    / get_delta(episode, self.args.delta_pow, self.args.const_delta)
                ).clamp(-self.args.grad_bound, self.args.grad_bound)

                if norm is not None and norm > self.args.grad_norm:
                    update *= self.args.grad_norm / norm

                t += get_alpha(episode, self.args.alpha) * update
        return norm


@torch.no_grad
def sf_reinforce(env, policy, seed, args, num_trials=10):
    """Driver code for SF-REINFOCE algorithm"""
    start = time.time()
    results = []
    norms = []
    two_sided = args.algo.startswith("two_sided")
    filename = "sf_reinforce_logs.txt"
    if two_sided:
        num_trials = num_trials // 2
        filename = "two_sided_sf_reinforce.txt"

    model = SFPolicy(env, policy, args)

    with open(filename, "+a") as file:
        print(f"Seed,Time,Episode,Reward", file=file)
    for episode in range(args.iterations):
        result = model.simulate(episode, num_trials)
        # update weights according to the paper
        norm = model.update_weights(episode, result.update_factor, result.perts)

        if norm is None:
            norm = 0
        norms.append(norm)
        results.append(result.avg_reward)

        if episode % 1000 == 0:
            avg = sum(results[-1000:]) / min(len(results), 1000)
            norm_avg = sum(norms[-1000:]) / min(len(norms), 1000)
            print(
                f"Seed: {seed}, time: {time.time() - start}, Episode {episode},"
                f"Average Reward: {avg:.3f}, delta_pow: {args.delta_pow}, "
                f"const_delta: {args.const_delta}, signed={args.sign}, norm: {norm_avg:.3f}",
            )
            with open(filename, "+a") as file:
                print(f"{seed},{time.time() - start},{episode},{avg}", file=file)
    return results
