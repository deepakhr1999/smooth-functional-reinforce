import torch
from collections import deque
import time


def reinforce(env, policy, optimizer, seed, num_episodes=1000, gamma=0.99):
    start = time.time()
    rolling_window = deque(maxlen=100)
    results = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        rewards = []
        log_probs = []

        while True:
            probs = policy(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            state, reward, term, trunc, _ = env.step(action.item())
            done = term or trunc

            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break

        G = 0
        policy_loss = 0

        for i in reversed(range(len(rewards))):
            G = rewards[i] + gamma * G
            policy_loss += -log_probs[i] * G

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        rolling_window.append(G)
        results.append(G)
        if episode % 1000 == 0:
            avg = sum(rolling_window) / len(rolling_window)
            print(
                f"Seed: {seed}, time: {time.time() - start}, Episode {episode}, Average Reward: {avg}"
            )
    return results
