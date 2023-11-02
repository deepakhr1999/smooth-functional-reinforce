import torch
import torch.nn as nn
import math


class PolicyNetwork(nn.Module):
    def __init__(self, n_actions, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.fc = nn.Linear(6, n_actions)

    def forward(self, state):
        features = extract_features(state, self.grid_size)
        return nn.functional.softmax(self.fc(features), dim=0)


def extract_features(state, grid_size):
    """Extract polynomial and modulo features from a scalar state."""
    alpha, beta = state // grid_size, state % grid_size
    features = [alpha, alpha**2, beta, beta**2, (alpha + beta) % 2, state]
    return torch.tensor(features, dtype=torch.float32)


class ContinuousPolicyNetwork(nn.Module):
    def __init__(self):
        super(ContinuousPolicyNetwork, self).__init__()

        self.fc = nn.Linear(8, action_dim * 2)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = 2
        self.min_action = -2
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0

    def forward(self, x):
        mu, log_std = torch.chunk(self.fc(x), 2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        features = extract_features(state)
        mean, log_std = self.forward(features)
        std = torch.exp(log_std)
        reparameter = torch.distributions.Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(
            torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True
        )

        return action, log_prob


def extract_features_pendulum(state):
    x, y, omega = state
    omega /= 8
    theta = math.asin(y) / math.pi
    return torch.tensor([x, x**2, y, y**2, omega, omega**2, theta, theta**2])
