"""REINFORCE algorithm"""
import time
from collections import deque
import torch

def loss_fn(policy, env, gamma=0.99) -> tuple[torch.Tensor, float]:
    """Return policy loss and discounted return"""
    state, _ = env.reset()
    rewards = []
    log_probs = []

    while True:
        probs = policy(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        state, reward, done, *_ = env.step(action.item())

        rewards.append(reward)
        log_probs.append(log_prob)

        if done:
            break

    total_return = 0.0
    policy_loss = torch.tensor(0.0)

    for idx in reversed(range(len(rewards))):
        total_return = rewards[idx] + gamma * total_return
        policy_loss += -log_probs[idx] * total_return
    return policy_loss, total_return


def reinforce(env, policy, seed, num_episodes=1000) -> list:
    """Driver for REINFORCE algorithm"""
    start = time.time()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rolling_window = deque(maxlen=100)
    results = []
    for episode in range(num_episodes):
        policy_loss, total_return = loss_fn(policy, env)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        rolling_window.append(total_return)
        results.append(total_return)
        if episode % 1000 == 0:
            avg = sum(rolling_window) / len(rolling_window)
            print(
                f"Seed: {seed}, time: {time.time() - start},"
                f"Episode {episode}, Average Reward: {avg}"
            )
    return results
