import torch
import time
from torch.nn import Module
import gym


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


def update_weights(
    policy,
    avg_reward,
    perts,
    episode,
    args,
):
    for t, pert in zip(policy.parameters(), perts):
        update_factor = avg_reward * pert
        if not args.sign:
            update = (
                update_factor / get_delta(episode, args.delta_pow, args.const_delta)
            ).clamp(-args.grad_bound, args.grad_bound)
            t += get_alpha(episode, args.alpha) * update
        else:
            sign = 2 * (update_factor > 0) - 1
            t += get_alpha(episode, args.alpha) * sign
    return policy


def get_delta(episode, delta_pow, const_delta):
    if const_delta is None:
        # run with values such as .15, .25, .35, .45
        return (2e-5 / (1 + episode * 2e-5)) ** delta_pow
    return const_delta


def get_alpha(episode, start=2e-6):
    return start / (1 + episode * 2e-5)


def sf_reinforce(
    env,
    policy,
    seed,
    args,
    num_trials=10,
    gamma=0.99,
):
    start = time.time()
    results = []
    two_sided = args.algo.startswith("two_sided")
    if two_sided:
        num_trials = num_trials // 2

    for episode in range(args.iterations):
        with torch.no_grad():
            # sample perturbations
            perturbed_policy, old_params, perts = perturb_policy(
                policy, delta=get_delta(episode, args.delta_pow, args.const_delta)
            )

            # simulate for num_trials
            rewards_plus = []
            for _ in range(num_trials):
                rewards_plus.append(simulate(perturbed_policy, env, gamma))
            avg_reward_plus = sum(rewards_plus) / len(rewards_plus)

            # perturb policy for -
            if two_sided:
                delta = get_delta(episode, args.delta_pow, args.const_delta)
                for new_param, pert in zip(perturbed_policy.parameters(), perts):
                    new_param.data -= 2 * delta * pert.data
                rewards_minus = []
                for _ in range(num_trials):
                    rewards_minus.append(simulate(perturbed_policy, env, gamma))
                avg_reward_minus = sum(rewards_minus) / len(rewards_minus)
                avg_reward = (avg_reward_plus + avg_reward_minus) / 2
                update_factor = (avg_reward_plus - avg_reward_minus) / 2
            else:
                avg_reward = avg_reward_plus
                update_factor = avg_reward_plus

            # revert weights of the policy
            policy = revert_weights(perturbed_policy, old_params)

            # update weights according to the paper
            policy = update_weights(
                policy,
                update_factor,
                perts,
                episode,
                args,
            )

        results.append(avg_reward)

        if episode % 1000 == 0:
            avg = sum(results[-1000:]) / min(len(results), 1000)
            print(
                f"Seed: {seed}, time: {time.time() - start}, Episode {episode}, Average Reward: {avg}, delta_pow: {args.delta_pow}, const_delta: {args.const_delta}, signed={args.sign}",
            )
            # print(f"Seed: {seed}: ", pd.Series(all_updates).describe())
    return results
