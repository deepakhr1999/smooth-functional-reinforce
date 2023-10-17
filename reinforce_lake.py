import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class PolicyNetwork(nn.Module):
    def __init__(self, n_features, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(n_features, n_actions)

    def forward(self, state):
        features = extract_features(state)
        return nn.functional.softmax(self.fc(features), dim=0)


def extract_features(state):
    """Extract polynomial and modulo features from a scalar state."""
    alpha, beta = state // 4, state % 4
    features = [alpha, alpha**2, beta, beta**2, (alpha + beta) % 2, state]
    return torch.tensor(features, dtype=torch.float32)


def reinforce(env, policy, optimizer, num_episodes=1000, gamma=0.99, verbose=1):
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
            if reward == 0:
                reward = -1
            elif reward == 1:
                reward = 10
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
        if episode % 1000 == 0 and verbose != 0:
            avg = sum(rolling_window) / len(rolling_window)
            print(f"Episode {episode}, Loss: {policy_loss.item()}, G={G}")
    return results
