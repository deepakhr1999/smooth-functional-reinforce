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

CONFIGS = {
    "tiny": {"size": 4, "slip_prob": 0.1, "max_len": 50},
    "small": {"size": 8, "slip_prob": 0.1, "max_len": 80},
    "medium": {"size": 10, "slip_prob": 0.1, "max_len": 100},
    "medium20": {"size": 20, "slip_prob": 0.1, "max_len": 150},
    "large": {"size": 50, "slip_prob": 0.1, "max_len": 200},
}


def read_delta_pow(algo):
    delta_pow = algo.split("_")[-1]
    const_delta = None
    if delta_pow == "reinforce":
        delta_pow = 0.25
    elif "const" in algo.split("_"):
        return None, float(delta_pow)
    return float(delta_pow), const_delta


def run_with_seed_sf_reinforce(seed, config_name="tiny", iterations=50000):
    torch.manual_seed(seed)
    cfg = CONFIGS[config_name]
    env = CustomGridWorld(**cfg)
    policy = PolicyNetwork(env.n_actions, grid_size=cfg["size"])
    results = sf_reinforce(env, policy, seed, iterations)

    dirname = f"saves/spsa/{config_name}"
    os.makedirs(dirname, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(dirname, "weights.{seed}.pth"))
    return results


def run_with_seed_reinforce(seed, config_name="tiny", iterations=50000):
    torch.manual_seed(seed)
    cfg = CONFIGS[config_name]
    env = CustomGridWorld(**cfg)

    policy = PolicyNetwork(env.n_actions, grid_size=cfg["size"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    results = reinforce(env, policy, optimizer, seed, iterations)

    dirname = f"saves/reinforce/{config_name}"
    os.makedirs(dirname, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(dirname, "weights.{seed}.pth"))
    return results


def main(algo: str, seed: int, config_name: str, iterations: int):
    torch.manual_seed(seed)
    cfg = CONFIGS[config_name]
    env_maker = lambda: CustomGridWorld(**cfg)
    
    if algo == "ppo":
        policy, results = run_ppo(env_maker, iterations, seed)
    elif algo.startswith("reinforce"):
        env = env_maker()
        policy = PolicyNetwork(env.n_actions, grid_size=cfg["size"])
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        results = reinforce(env, policy, optimizer, seed, iterations)
    else:
        env = env_maker()
        policy = PolicyNetwork(env.n_actions, grid_size=cfg["size"])
        delta_pow, const_delta = read_delta_pow(algo)
        results = sf_reinforce(
            env,
            policy,
            seed,
            delta_pow,
            const_delta,
            iterations,
            two_sided=algo.startswith("two_sided"),
            signed=("sign" in algo)
        )

    dirname = f"saves/{algo}/{config_name}"
    os.makedirs(dirname, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(dirname, f"weights.{seed}.pth"))
    return results


if __name__ == "__main__":
    assert (
        sys.argv[1].startswith("reinforce")
        or sys.argv[1].startswith("sf_reinforce")
        or sys.argv[1].startswith("two_sided_sf_reinforce")
        or sys.argv[1].startswith("ppo")
    ), "Wrong algorithm chosen"
    algo = sys.argv[1]
    config_name = sys.argv[2]
    iterations = int(sys.argv[3])
    with Pool(processes=10) as pool:
        results = pool.starmap(
            main, [(algo, seed, config_name, iterations) for seed in range(10)]
        )

    filename = "_".join(map(str, [iterations, int(time.time()), "results.pkl"]))
    with open(os.path.join("saves", algo, config_name, filename), "wb") as file:
        pickle.dump(results, file)
