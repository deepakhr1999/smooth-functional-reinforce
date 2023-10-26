import sys
from reinforce_lake import PolicyNetwork, reinforce
from spsa_lake import spsa
import pickle
import torch
from gridworld import CustomGridWorld
from multiprocessing import Pool


def run_with_seed_spsa(seed):
    torch.manual_seed(seed)
    env = CustomGridWorld(4, 0.1, 50)
    n_actions = 4
    n_features = 6  # Based on our extract_features function

    policy = PolicyNetwork(n_features, n_actions)

    # results = spsa_x(env, policy, 10000, num_trials=10, num_perts=5, x=1)
    # optimizer = torch.optim.Adam(policy.parameters(), lr=3e-5)
    num_trials = 10
    iterations = 50000
    results = spsa(env, policy, seed, iterations, num_trials=num_trials)
    torch.save(
        policy.state_dict(),
        f"saves/spsa/weights.seed={seed}.num_trails={num_trials}.pth",
    )
    return results


def run_with_seed_reinforce(seed):
    torch.manual_seed(seed)
    env = CustomGridWorld(4, 0.1, 50)
    n_actions = 4
    n_features = 6  # Based on our extract_features function

    policy = PolicyNetwork(n_features, n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    results = reinforce(env, policy, optimizer, seed, 50000, 0.99, 1)
    torch.save(policy.state_dict(), f"saves/reinforce/weights.{seed}.pth")
    return results


if __name__ == "__main__":
    if sys.argv[1] == "reinforce":
        with Pool(processes=10) as pool:
            results = pool.map(run_with_seed_reinforce, list(range(10)))

        with open("reinforce_results.pkl", "wb") as file:
            pickle.dump(results, file)
    else:
        with Pool(processes=10) as pool:
            results = pool.map(run_with_seed_spsa, list(range(10)))

        with open("spsa_results.pkl", "wb") as file:
            pickle.dump(results, file)
