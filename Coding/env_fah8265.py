import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class FastFuriousEnv(gym.Env):
    """
    Custom Environment for a Fast and Furious inspired game.

    Agent Dom must navigate a grid to reach a goal while avoiding buildings and collecting power-ups.
    """

    def __init__(self, grid_size=7, level=1):
        """
        Initialize the environment.

        Parameters:
        grid_size (int): The size of the grid.
        level (int): The level of the game.
        """
        super().__init__()
        self.grid_size = grid_size
        self.level = level
        self.agent_states = {
            "Dom": np.array([0, 0])
        }  # Initial position for the agent
        self.goal_state = np.array([grid_size // 2, grid_size // 2])  # Goal position changes with level
        self.buildings = [
            {"position": np.array([3, 4]), "image": 'building1.png'},
            {"position": np.array([4, 3]), "image": 'building2.png'},
            {"position": np.array([2, 2]), "image": 'building3.png'}
        ]  # Static buildings with different images
        self.power_ups = [np.array([2, 5]), np.array([5, 2])]  # Positions for power-ups
        self.action_space = gym.spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int64)
        self.fig, self.ax = plt.subplots()
        self.car_images = {
            "Dom": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/dom_car.png')
        }
        self.building_images = {
            "building1": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/building1.png'),
            "building2": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/building2.png'),
            "building3": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/building3.png'),
        }
        self.power_up_image = mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/power_up.png')
        self.finish_line_image = mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/finish_line.png')  # Load finish line image
        plt.show(block=False)

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        dict: The initial states of the agent.
        """
        self.agent_states = {
            "Dom": np.array([0, 0])
        }  # Reset initial position for the agent
        self.power_ups = [np.array([2, 5]), np.array([5, 2])]  # Reset power-ups
        return self.agent_states

    def step(self, action):
        """
        Execute actions and return the new state, reward, done, and info.

        Parameters:
        action (int): The action for the agent.

        Returns:
        dict: The new states of the agent.
        dict: The rewards for the agent.
        bool: Whether the episode is done.
        dict: Additional information.
        """
        reward = 0
        done = False
        info = {}

        # Define the penalties and rewards
        building_penalty = -3
        power_up_reward = 5

        # Process agent's action
        new_state = self.agent_states["Dom"].copy()
        if action == 0 and new_state[1] < self.grid_size - 1:  # up
            new_state[1] += 1
        elif action == 1 and new_state[1] > 0:  # down
            new_state[1] -= 1
        elif action == 2 and new_state[0] > 0:  # left
            new_state[0] -= 1
        elif action == 3 and new_state[0] < self.grid_size - 1:  # right
            new_state[0] += 1

        # Check for collisions with buildings and power-ups
        if any(np.array_equal(new_state, bldg["position"]) for bldg in self.buildings):
            reward = building_penalty
            done = False
            info = {"Accident with Building"}
        elif any(np.array_equal(new_state, pu) for pu in self.power_ups):
            reward = power_up_reward
            self.power_ups = [pu for pu in self.power_ups if not np.array_equal(new_state, pu)]  # Remove collected power-up
            done = False
            info = {"Powers":"Collected Power-Up"}
            distance_to_goal = np.linalg.norm(self.goal_state - new_state)
            info.update({"Distance to Goal": distance_to_goal})
        else:
            # Check if the new state is within the grid boundaries
            if (0 <= new_state[0] < self.grid_size) and (0 <= new_state[1] < self.grid_size):
                self.agent_states["Dom"] = new_state
                reward = 0
                # Check if all power-ups are collected and Dom has reached the goal
                if np.array_equal(new_state, self.goal_state) and len(self.power_ups) == 0:
                    reward = 10
                    done = True
                    info = {"Goal Reached": True}
                elif np.array_equal(new_state, self.goal_state):
                    reward = 0
                    done = False
                    info = {"Goal Reached without collecting all Power-Ups": False}
                distance_to_goal = np.linalg.norm(self.goal_state - new_state)
                info.update({"Distance to Goal": distance_to_goal})
            else:
                # Agent attempted to move out of bounds
                reward = -5  # Penalty for attempting to move out of bounds
                done = False
                info = {"Attempted to move out of bounds"}

        return self.agent_states["Dom"], reward, done, info

    def render(self):
        """
        Render the current state of the environment.
        """
        self.ax.clear()
        for agent, state in self.agent_states.items():
            car_image = self.car_images[agent]
            self.ax.imshow(car_image, extent=(state[0] - 0.5, state[0] + 0.5, state[1] - 0.5, state[1] + 0.5))
        self.ax.imshow(self.finish_line_image, extent=(self.goal_state[0] - 0.5, self.goal_state[0] + 0.5, self.goal_state[1] - 0.5, self.goal_state[1] + 0.5))  # Finish Line
        for bldg in self.buildings:
            building_image = self.building_images[bldg["image"].split('.')[0]]
            self.ax.imshow(building_image, extent=(bldg["position"][0] - 0.5, bldg["position"][0] + 0.5, bldg["position"][1] - 0.5, bldg["position"][1] + 0.5))  # Buildings
        for pu in self.power_ups:
            self.ax.imshow(self.power_up_image, extent=(pu[0] - 0.5, pu[0] + 0.5, pu[1] - 0.5, pu[1] + 0.5))  # Power-Ups

        # Turn off the axis
        self.ax.axis('off')

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
    env = FastFuriousEnv(grid_size=7)  # Initialize with custom grid size
    state = env.reset()
    for _ in range(500):
        action = env.action_space.sample()  # Sample random action for the agent
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("Dom wins!")
            break
    env.close()
