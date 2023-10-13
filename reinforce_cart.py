import os

# Set JAX to use CPU backend
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import jax
import jax.numpy as jnp
import flax.linen as nn
import gym
import numpy as np

class Policy(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Assuming x is the state
        # x = nn.Dense(32, kernel_init=nn.initializers.xavier_uniform())(x)
        # x = jax.nn.tanh(x)
        # x = nn.Dense(8, kernel_init=nn.initializers.xavier_uniform())(x)
        # x = jax.nn.relu(x)
        # x = nn.Dense(8, kernel_init=nn.initializers.xavier_uniform())(x)
        # x = jax.nn.relu(x)
        x = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x) # mean of the action distribution
        x = jax.nn.sigmoid(x)
        # sigma = jnp.exp(nn.Dense(1)(x))  # standard deviation (log-space to keep it positive)
        if np.random.random() < x[0]:
            return 1
        return 0


def sample_episode(env, policy_fn, params, gamma=0.99):
    state, _ = env.reset()
    done = False
    total_reward = 0
    for tdx in range(500):
        state = jnp.array(state)
        action = policy_fn.apply(params, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += gamma ** tdx * reward
        state = next_state
        if terminated or truncated:
            break
    return total_reward


def alpha(idx):
    return 5e-6 * (1000 / (idx + 1000)) ** -1.5


def delta(idx):
    return 1e-4 * 1000 / (idx + 1000)


def train_episode(env, model, params, idx, num_episodes):
    # Perturb the policy parameters and store the perturbation as pert
    perts = jax.tree_map(
        lambda x: np.random.normal(size=x.shape), params
    )
    perturbed_params = jax.tree_map(lambda x, d: x + delta(idx) * d, params, perts)
    perturbed_params_m = jax.tree_map(lambda x, d: x - delta(idx) * d, params, perts)

    # Sample an episode using the perturbed policy
    total_reward = np.mean([
        sample_episode(env, model, perturbed_params)
        for _ in range(num_episodes)
    ])

    total_reward_m = np.mean([
        sample_episode(env, model, perturbed_params_m)
        for _ in range(num_episodes)
    ])

    # Update the original parameter by delta * total_reward
    factor = alpha(idx) * (total_reward - total_reward_m) / (2 * delta(idx))
    updated_params = jax.tree_map(
        lambda theta, pert: theta + factor / pert,
        params,
        perts,
    )

    print(f"Episode: {idx}, Reward: {total_reward}, {total_reward_m}, Factor: {factor}")
    return updated_params, total_reward


def main():
    # Create the Pendulum environment
    env = gym.make("CartPole-v1")

    # Initialize the policy model
    model = Policy()
    params = model.init(jax.random.PRNGKey(0), jnp.ones([4]))

    for idx in range(1000):
        params, total_reward = train_episode(env, model, params, idx, num_episodes=50)



if __name__ == "__main__":
    main()
