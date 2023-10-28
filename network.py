import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, n_actions, grid_size):
        super(PolicyNetwork, self).__init__()
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
