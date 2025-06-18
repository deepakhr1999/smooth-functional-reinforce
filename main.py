"""Driver Code"""
import os
import time
import argparse
from multiprocessing import Pool
import pickle
import torch
from reinforce import reinforce
from network import PolicyNetwork
from sf_reinforce import sf_reinforce
from ppo import run_ppo
from gridworld import CustomGridWorld

CONFIGS = {
    "tiny": {"size": 4, "slip_prob": 0.1, "max_len": 50},
    "small": {"size": 8, "slip_prob": 0.1, "max_len": 80},
    "medium": {"size": 10, "slip_prob": 0.1, "max_len": 100},
    "medium20": {"size": 20, "slip_prob": 0.1, "max_len": 150},
    "large": {"size": 50, "slip_prob": 0.1, "max_len": 200},
}


def get_dirname(args: argparse.Namespace) -> str:
    """Save directory from config"""
    delta_str = "_signed_" if args.sign else "_"
    if args.delta_pow is None:
        if args.const_delta != 0.175 or True:
            delta_str += f"const_delta={args.const_delta}"
    else:
        delta_str += str(args.delta_pow)
    if args.alpha != 2e-6:
        delta_str += f"_start_alpha={args.alpha}"

    if args.grad_bound < 1e5:
        delta_str += f"_grad_bound={args.grad_bound}"

    if args.grad_norm < 1e9:
        delta_str += f"_grad_norm={args.grad_norm}"
    if delta_str == "_":
        delta_str = ""
    return f"saves/{args.algo}{delta_str}/{args.config_name}"


def run_for_seed(seed: int, args: argparse.Namespace):
    """Run the algorithm for one seed using config from cmd args"""
    torch.manual_seed(seed)
    dirname = get_dirname(args)
    print("saving to", dirname)

    cfg = CONFIGS[args.config_name]
    def env_maker():
        return CustomGridWorld(**cfg)

    if args.algo in ("ppo", "a2c", "trpo"):
        policy, results = run_ppo(args.algo, env_maker, args.iterations, seed)
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

def main(args: argparse.Namespace):
    """Driver code"""
    with Pool(processes=3) as pool:
        results = pool.starmap(run_for_seed, [(seed, args) for seed in range(10)])


    filename = "_".join(map(str, [args.iterations, int(time.time()), "results.pkl"]))
    dirname = get_dirname(args)
    with open(os.path.join(dirname, filename), "wb") as file:
        pickle.dump(results, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train agents using various RL algorithms."
    )
    parser.add_argument(
        "algo",
        choices=["reinforce", "sf_reinforce", "two_sided_sf_reinforce", "ppo", "a2c", "trpo"],
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
    parser.add_argument("--grad_norm", type=float, default=1e9)
    brgs = parser.parse_args()

    main(brgs)
