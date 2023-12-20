import numpy as np
import gymnasium as gym

class CustomGridWorld(gym.Env):
    def __init__(self, size=4, slip_prob=0.2, max_len=100, **kwargs):
        """
        size: Dimension of the square grid.
        slip_prob: Probability of moving in a perpendicular direction.
        """
        self.size = size
        self.n_actions = 4
        self.slip_prob = slip_prob
        self.state = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.steps = 0
        self.max_len = max_len
        self.observation_space = gym.spaces.Discrete(self.size**2)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *args, **kwargs):
        self.steps = 0
        self.state = (0, 0)
        return 0, None

    def step(self, action):
        # Action mappings: 0=up, 1=right, 2=down, 3=left
        self.steps += 1

        # If the slip happens, choose a perpendicular direction
        if np.random.rand() < self.slip_prob:
            action_offsets = [[1, 3], [0, 2], [1, 3], [0, 2]]  # Perpendicular offsets
            action = np.random.choice(action_offsets[action])

        if action == 0:  # up
            new_state = (max(self.state[0] - 1, 0), self.state[1])
        elif action == 1:  # right
            new_state = (self.state[0], min(self.state[1] + 1, self.size - 1))
        elif action == 2:  # down
            new_state = (min(self.state[0] + 1, self.size - 1), self.state[1])
        elif action == 3:  # left
            new_state = (self.state[0], max(self.state[1] - 1, 0))

        self.state = new_state

        done = self.state == self.goal
        reward = (
            self.size**2
            if done
            else -(sum(self.goal) - sum(self.state)) / (5 * self.size)
        )

        if self.steps == self.max_len:
            done = True
        return new_state[0] * self.size + new_state[1], reward, done, self.steps >= self.max_len, {}

    def render(self):
        grid = [["-" for _ in range(self.size)] for _ in range(self.size)]
        grid[self.state[0]][self.state[1]] = "A"
        grid[self.goal[0]][self.goal[1]] = "G"
        for row in grid:
            print(" ".join(row))
        print()


# Example usage:
if __name__ == "__main__":
    env = CustomGridWorld(size=4, slip_prob=0.2)
    state, _ = env.reset()
    done = False
    while not done:
        action = np.random.choice([0, 1, 2, 3])  # Take random actions for demonstration
        state, reward, done, *_ = env.step(action)
        env.render()
