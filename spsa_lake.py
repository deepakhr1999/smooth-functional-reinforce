import torch
import time
from collections import deque


def top_x_grad(env, policy, gamma, alpha, delta, avg, num_trials, num_perts, x):
    return_pert_pairs = [
        f(env, policy, gamma, delta, num_trials) for _ in range(num_perts)
    ]
    return_pert_pairs.sort(reverse=True)

    log_G = 0
    for idx in range(x):
        G, perts = return_pert_pairs[idx]
        log_G += G
        for t, pert in zip(policy.parameters(), perts):
            t += alpha * (G - avg) * pert / delta
    return log_G / x


def f(env, policy, gamma, delta, num_trials):
    perts = [torch.randn_like(t.data) for t in policy.parameters()]
    old_params = [t.clone() for t in policy.parameters()]
    for t, d in zip(policy.parameters(), perts):
        t.data += delta * d.data
    G = []
    for _ in range(num_trials):
        state, _ = env.reset()
        tdx = 0
        G_new = 0
        while True:
            probs = policy(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            state, reward, term, trunc, _ = env.step(action.item())
            done = term or trunc
            # if reward == 0:
            #     reward = -1
            # elif reward == 1:
            #     reward = 10
            G_new += gamma**tdx * reward
            tdx += 1
            if done:
                break
        G.append(G_new)

    for t, old_t in zip(policy.parameters(), old_params):
        t.data = old_t.data.clone()
    return sum(G) / len(G), perts


def get_delta(episode):
    return (2e-5 / (1 + episode * 2e-5)) ** 0.25


def get_alpha(episode):
    return 2e-6 / (1 + episode * 2e-5)


def spsa(env, policy, seed, num_episodes=20000, gamma=0.99, verbose=1, num_trials=100):
    start = time.time()
    rolling_window = deque(maxlen=1000)
    results = []
    delta = 0.1
    for episode in range(num_episodes):
        with torch.no_grad():
            G, perts = f(
                env, policy, gamma, delta=get_delta(episode), num_trials=num_trials
            )
            avg = 0
            if len(rolling_window) > 0:
                avg = sum(rolling_window) / len(rolling_window)

            for t, pert in zip(policy.parameters(), perts):
                t += get_alpha(episode) * G * pert / get_delta(episode)

        # optimizer.step()
        rolling_window.append(G)
        results.append(G)

        if episode % 1000 == 0 and verbose != 0:
            print(
                f"Seed: {seed}, time: {time.time() - start}, Episode {episode}, Average Reward: {avg}"
            )

    return results


def spsa_x(
    env,
    policy,
    num_episodes=20000,
    gamma=0.99,
    verbose=1,
    num_trials=100,
    num_perts=10,
    x=3,
):
    rolling_window = deque(maxlen=2000)
    results = []
    delta = 0.1
    for episode in range(num_episodes):
        with torch.no_grad():
            avg = 0
            if len(rolling_window) > 0:
                avg = sum(rolling_window) / len(rolling_window)
            # Update parameters using the gradient estimate

            G = top_x_grad(
                env,
                policy,
                gamma,
                get_alpha(episode),
                get_delta(episode),
                avg,
                num_trials,
                num_perts,
                x,
            )

        rolling_window.append(G)
        results.append(G)

        if episode % 100 == 0 and verbose != 0:
            print(f"Episode {episode}, Average Reward: {avg}, G={G}")

    return results


if __name__ == "__main__":
    import gym
    from network import PolicyNetwork

    env = gym.make(
        "FrozenLake-v1", is_slippery=True, desc=["SFFH", "FFFF", "FFFF", "FFFG"]
    )
    n_actions = 4
    n_features = 6

    policy = PolicyNetwork(n_features, n_actions)
    results = spsa(env, policy)
    print(list(policy.parameters()))
