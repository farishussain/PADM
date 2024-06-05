import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class SnakesLaddersEnv(gym.Env):
    """
    Custom Environment for a Snake and Ladders game.
    """

    def __init__(self, grid_size=10):
        """
        Initialize the environment.
        """
        super().__init__()
        self.grid_size = grid_size
        self.grid = np.arange(1, grid_size**2 + 1).reshape((grid_size, grid_size))
        self.snakes = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}
        self.ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
        self.position = 1  # Starting position
        self.action_space = spaces.Discrete(6)  # Simulating dice rolls from 1 to 6
        self.observation_space = spaces.Discrete(grid_size**2)
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.position = 1
        return self.position

    def step(self, action):
        """
        Execute actions and return the new state, reward, done, and info.
        """
        done = False
        reward = 0
        self.position += action + 1  # Dice roll from 1 to 6

        # Check for snakes and ladders
        if self.position in self.snakes:
            self.position = self.snakes[self.position]
        elif self.position in self.ladders:
            self.position = self.ladders[self.position]

        if self.position >= self.grid_size**2:
            self.position = self.grid_size**2
            done = True
            reward = 10  # Reward for winning the game

        return self.position, reward, done, {}

    def render(self):
        """
        Render the current state of the environment.
        """
        self.ax.clear()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_number = self.grid[i, j]
                self.ax.text(j, i, str(cell_number), ha='center', va='center')

        self.ax.plot(self.position % self.grid_size, self.position // self.grid_size, 'ro')  # Player's position
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")
        plt.pause(0.1)

    def close(self):
        """
        Close the rendering window.
        """
        plt.close()

if __name__ == "__main__":
    env = SnakesLaddersEnv(grid_size=10)  # Initialize with custom grid size
    state = env.reset()
    for _ in range(100):
        action = env.action_space.sample()  # Sample random action (dice roll)
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("Reached the end!")
            break
    env.close()
