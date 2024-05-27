import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class FastFuriousEnv(gym.Env):
    def __init__(self, grid_size = 7):
        super().__init__()
        self.grid_size = grid_size
        self.agent_states = {
            "Dom": np.array([0, 0]),
            "Brian": np.array([6,6])
        } #Initial Positions for Two Agents
        self.goal_state = np.array([3, 3]) #Single Goal Position
        self.obstacles = [np.array([3, 4]), np.array([4, 3]), np.array([2, 2])] #Obstacles
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.space.Box(low=0, high=self.grid_size - 1, shape=(2,))
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self):
        self.agent_states = {
            "Dom": np.array([0, 0]),
            "Brian": np.array([6, 6])
        }   #Reset initial positions for two agents
        return self.agent_states
    
    def step(self, actions):
        rewards = {}
        infos = {}
        dones = {}

        for agent, action in actions.items():
