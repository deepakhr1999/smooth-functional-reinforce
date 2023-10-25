import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, n_features, n_actions, n_hidden=10):
        super(PolicyNetwork, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_features, n_actions),
            # nn.Tanh(),
            # nn.Linear(n_hidden, n_hidden),
            # nn.Tanh(),
            # nn.Linear(n_hidden, n_actions),
            nn.Softmax(),
        )

    def forward(self, state):
        features = extract_features(state)
        return self.nn(features)


def extract_features(state):
    """Extract polynomial and modulo features from a scalar state."""
    alpha, beta = state // 4, state % 4
    features = [alpha, alpha**2, beta, beta**2, (alpha + beta) % 2, state]
    return torch.tensor(features, dtype=torch.float32)
