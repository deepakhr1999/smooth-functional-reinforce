import sys
from reinforce import reinforce
from network import PolicyNetwork
from spsa import spsa
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


def run_with_seed_spsa(seed, config_name="tiny", iterations=50000):
    torch.manual_seed(seed)
    cfg = CONFIGS[config_name]
    env = CustomGridWorld(**cfg)
    policy = PolicyNetwork(env.n_actions, grid_size=cfg["size"])
    results = spsa(env, policy, seed, iterations)

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


def main(algo, seed, config_name, iterations):
    torch.manual_seed(seed)
    cfg = CONFIGS[config_name]
    env = CustomGridWorld(**cfg)
    policy = PolicyNetwork(env.n_actions, grid_size=cfg["size"])

    if algo.startswith("reinforce"):
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        results = reinforce(env, policy, optimizer, seed, iterations)
    else:
        delta_pow = algo.split("_")[-1]
        results = spsa(env, policy, seed, float(delta_pow), iterations)

    dirname = f"saves/{algo}/{config_name}"
    os.makedirs(dirname, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(dirname, f"weights.{seed}.pth"))
    return results


if __name__ == "__main__":
    assert sys.argv[1].startswith("reinforce") or sys.argv[1].startswith(
        "sf_reinforce"
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
