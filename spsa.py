import torch
import time
from collections import deque
from torch.nn import Module
import gym


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


def perturb_policy(policy: Module, delta: float) -> tuple[Module, list, list]:
    perts = [torch.randn_like(t.data) for t in policy.parameters()]
    old_params = [t.clone() for t in policy.parameters()]
    for t, d in zip(policy.parameters(), perts):
        t.data += delta * d.data
    return policy, old_params, perts


def simulate(policy: Module, env: gym.Env, gamma: float) -> float:
    state, _ = env.reset()
    tdx = 0
    G_new = 0
    while True:
        probs = policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        state, reward, term, trunc, _ = env.step(action.item())
        done = term or trunc
        G_new += gamma**tdx * reward
        tdx += 1
        if done:
            break
    return G_new


def revert_weights(policy, old_params):
    for t, old_t in zip(policy.parameters(), old_params):
        t.data = old_t.data.clone()
    return policy


def update_weights(policy, avg_reward, perts, episode, delta_pow):
    for t, pert in zip(policy.parameters(), perts):
        t += get_alpha(episode) * avg_reward * pert / get_delta(episode, delta_pow)
    return policy


def get_delta(episode, delta_pow):
    # run with values such as .15, .25, .35, .45
    if not delta_pow:
        return 0.005
    return (2e-5 / (1 + episode * 2e-5)) ** delta_pow


def get_alpha(episode):
    return 2e-6 / (1 + episode * 2e-5)


# def sf_reinforce(
#     env, policy, seed, delta_pow, num_episodes=20000, gamma=0.99, num_trials=10
# ):
#     start = time.time()
#     results = []
#     for episode in range(num_episodes):
#         with torch.no_grad():
#             # sample perturbations
#             perturbed_policy, old_params, perts = perturb_policy(
#                 policy, delta=get_delta(episode, delta_pow)
#             )

#             # simulate for num_trials
#             rewards = []
#             for _ in range(num_trials):
#                 rewards.append(simulate(perturbed_policy, env, gamma))

#             # revert weights of the policy
#             policy = revert_weights(perturbed_policy, old_params)

#             # update weights according to the paper
#             avg_reward = sum(rewards) / len(rewards)
#             policy = update_weights(policy, avg_reward, perts, episode, delta_pow)

#         results.append(avg_reward)

#         if episode % 1000 == 0:
#             avg = sum(results[-1000:]) / min(len(results), 1000)
#             print(
#                 f"Seed: {seed}, time: {time.time() - start}, Episode {episode}, Average Reward: {avg}, delta_pow: {delta_pow}",
#             )

#     return results


def sf_reinforce(
    env, policy, seed, delta_pow, num_episodes=20000, gamma=0.99, num_trials=10, two_sided=False,
):
    start = time.time()
    results = []
    for episode in range(num_episodes):
        with torch.no_grad():
            # sample perturbations
            perturbed_policy, old_params, perts = perturb_policy(
                policy, delta=get_delta(episode, delta_pow)
            )

            # simulate for num_trials
            rewards_plus = []
            for _ in range(num_trials):
                rewards_plus.append(simulate(perturbed_policy, env, gamma))
            avg_reward_plus = sum(rewards_plus) / len(rewards_plus)

            # perturb policy for -
            if two_sided:
                delta = get_delta(episode, delta_pow)
                for new_param, pert in zip(perturbed_policy.parameters(), perts):
                    new_param.data -= 2 * delta * pert
                rewards_minus = []
                for _ in range(num_trials):
                    rewards_minus.append(simulate(perturbed_policy, env, gamma))
                avg_reward_minus = sum(rewards_minus) / len(rewards_minus)
                avg_reward = (avg_reward_plus - avg_reward_minus) / 2
            else:
                avg_reward = avg_reward_plus
            
            # revert weights of the policy
            policy = revert_weights(perturbed_policy, old_params)

            # update weights according to the paper
            policy = update_weights(policy, avg_reward, perts, episode, delta_pow)

        results.append(avg_reward)

        if episode % 1000 == 0:
            avg = sum(results[-1000:]) / min(len(results), 1000)
            print(
                f"Seed: {seed}, time: {time.time() - start}, Episode {episode}, Average Reward: {avg}, delta_pow: {delta_pow}",
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
