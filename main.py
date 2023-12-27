import sys
from reinforce import reinforce
from network import PolicyNetwork
from spsa import sf_reinforce
from ppo import run_ppo
import pickle
import torch
from gridworld import CustomGridWorld
from multiprocessing import Pool
import os
import time
import argparse

CONFIGS = {
    "tiny": {"size": 4, "slip_prob": 0.1, "max_len": 50},
    "small": {"size": 8, "slip_prob": 0.1, "max_len": 80},
    "medium": {"size": 10, "slip_prob": 0.1, "max_len": 100},
    "medium20": {"size": 20, "slip_prob": 0.1, "max_len": 150},
    "large": {"size": 50, "slip_prob": 0.1, "max_len": 200},
}


def get_dirname(args):
    delta_str = "signed_" if args.sign else ""
    if args.delta_pow is None:
        delta_str += f"const_delta={args.const_delta}"
    else:
        delta_str += str(args.delta_pow)
    if args.alpha != 2e-6:
        delta_str += f"_start_alpha={args.alpha}"

    if args.grad_bound < 1e5:
        delta_str += f"_grad_bound_{args.grad_bound}"
    return f"saves/{args.algo}_{delta_str}/{args.config_name}"


def read_delta_pow(algo):
    delta_pow = algo.split("_")[-1]
    const_delta = None
    if delta_pow == "reinforce":
        delta_pow = 0.25
    elif "const" in algo.split("_"):
        return None, float(delta_pow)
    return float(delta_pow), const_delta


def main(seed: int, args):
    torch.manual_seed(seed)
    dirname = get_dirname(args)
    print("saving to", dirname)

    cfg = CONFIGS[args.config_name]
    env_maker = lambda: CustomGridWorld(**cfg)

    if args.algo == "ppo":
        policy, results = run_ppo(env_maker, args.iterations, seed)
    elif args.algo.startswith("reinforce"):
        env = env_maker()
        policy = PolicyNetwork(env.n_actions, grid_size=cfg["size"])
        results = reinforce(env, policy, seed, args.iterations)
    else:
        env = env_maker()
        policy = PolicyNetwork(env.n_actions, grid_size=cfg["size"])
        results = sf_reinforce(env, policy, seed, args)

    os.makedirs(dirname, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(dirname, f"weights.{seed}.pth"))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train agents using various RL algorithms."
    )
    parser.add_argument(
        "algo",
        choices=["reinforce", "sf_reinforce", "two_sided_sf_reinforce", "ppo"],
        help="RL algorithm",
    )
    parser.add_argument(
        "config_name", choices=CONFIGS.keys(), help="Configuration name"
    )
    parser.add_argument("iterations", type=int, help="Number of iterations")
    parser.add_argument("--sign", action="store_true", default=False)
    parser.add_argument("--delta_pow", type=float, default=None)
    parser.add_argument("--const_delta", type=float, default=0.175)
    parser.add_argument("--alpha", type=float, default=2e-6)
    parser.add_argument("--grad_bound", type=float, default=1e5)
    args = parser.parse_args()

    with Pool(processes=10) as pool:
        results = pool.starmap(main, [(seed, args) for seed in range(10)])

    filename = "_".join(map(str, [args.iterations, int(time.time()), "results.pkl"]))
    dirname = get_dirname(args)
    with open(os.path.join(dirname, filename), "wb") as file:
        pickle.dump(results, file)
