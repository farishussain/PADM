import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

class FastFuriousEnv(gym.Env):
    """
    Custom Environment for a Fast and Furious inspired game.

    Agents Dom and Brian must navigate a grid to reach a goal while avoiding buildings and police cars.
    """

    def __init__(self, grid_size=7, num_police=2):
        """
        Initialize the environment.

        Parameters:
        grid_size (int): The size of the grid.
        num_police (int): The number of police cars in the environment.
        """
        super().__init__()
        self.grid_size = grid_size
        self.num_police = num_police
        self.agent_states = {
            "Dom": np.array([0, 0]),
            "Brian": np.array([6, 6])
        }  # Initial positions for two agents
        self.goal_state = np.array([3, 3])  # Single goal position
        self.buildings = [
            {"position": np.array([3, 4]), "image": 'building1.png'},
            {"position": np.array([4, 3]), "image": 'building2.png'},
            {"position": np.array([2, 2]), "image": 'building3.png'}
        ]  # Static buildings with different images
        self.police_states = [np.array([3, 0]), np.array([0, 3])]  # Initial positions for police
        self.action_space = gym.spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int64)
        self.fig, self.ax = plt.subplots()
        self.car_images = {
            "Dom": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/dom_car.png'),
            "Brian": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/brian_car.png'),
            "Police": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/police_car.png'),
        }
        self.building_images = {
            "building1": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/building1.png'),
            "building2": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/building2.png'),
            "building3": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/building3.png'),
        }
        self.finish_line_image = mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Coding/finish_line.png')  # Load finish line image
        plt.show(block=False)

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        dict: The initial states of the agents.
        """
        self.agent_states = {
            "Dom": np.array([0, 0]),
            "Brian": np.array([6, 6])
        }  # Reset initial positions for two agents
        self.police_states = [np.array([3, 0]), np.array([0, 3])]  # Reset police positions
        return self.agent_states

    def step(self, actions):
        """
        Execute actions and return the new state, reward, done, and info.

        Parameters:
        actions (dict): A dictionary with agents' actions.

        Returns:
        dict: The new states of the agents.
        dict: The rewards for each agent.
        dict: Whether the episode is done for each agent.
        dict: Additional information.
        """
        rewards = {}
        infos = {}
        dones = {}

        # Define the penalties
        building_penalty = -3
        police_penalty = -2
        agent_collision_penalty = -1

        # Move police randomly
        for i, police_state in enumerate(self.police_states):
            direction = np.random.choice(['up', 'down', 'left', 'right'])
            if direction == 'up' and police_state[1] < self.grid_size - 1:
                self.police_states[i][1] += 1
            elif direction == 'down' and police_state[1] > 0:
                self.police_states[i][1] -= 1
            elif direction == 'left' and police_state[0] > 0:
                self.police_states[i][0] -= 1
            elif direction == 'right' and police_state[0] < self.grid_size - 1:
                self.police_states[i][0] += 1

        # Process agents' actions
        for agent, action in actions.items():
            new_state = self.agent_states[agent].copy()
            if action == 0 and new_state[1] < self.grid_size - 1:  # up
                new_state[1] += 1
            elif action == 1 and new_state[1] > 0:  # down
                new_state[1] -= 1
            elif action == 2 and new_state[0] > 0:  # left
                new_state[0] -= 1
            elif action == 3 and new_state[0] < self.grid_size - 1:  # right
                new_state[0] += 1

            # Check for collisions
            if any(np.array_equal(new_state, bldg["position"]) for bldg in self.buildings):
                reward = building_penalty
                done = False
                info = {"Accident with Building"}
            elif any(np.array_equal(new_state, police_state) for police_state in self.police_states):
                reward = police_penalty
                done = True
                info = {"Arrested by Police rest Positon"}
                # Reset the agent's position
                self.agent_states[agent] = np.array([0, 0]) if agent == "Dom" else np.array([6, 6])
            elif any(np.array_equal(new_state, other_state) for other_agent, other_state in self.agent_states.items() if other_agent != agent):
                reward = agent_collision_penalty
                done = False
                info = {"Accident with other Agent"}
            else:
                self.agent_states[agent] = new_state
                reward = 0
                done = np.array_equal(new_state, self.goal_state)
                if done:
                    reward = 10
                distance_to_goal = np.linalg.norm(self.goal_state - new_state)
                info = {"Distance to Goal": distance_to_goal}

            rewards[agent] = reward
            dones[agent] = done
            infos[agent] = info

        return self.agent_states, rewards, dones, infos

    def render(self):
        """
        Render the current state of the environment.
        """
        self.ax.clear()
        for agent, state in self.agent_states.items():
            car_image = self.car_images[agent]
            self.ax.imshow(car_image, extent=(state[0] - 0.5, state[0] + 0.5, state[1] - 0.5, state[1] + 0.5))
        for police_state in self.police_states:
            police_image = self.car_images["Police"]
            self.ax.imshow(police_image, extent=(police_state[0] - 0.5, police_state[0] + 0.5, police_state[1] - 0.5, police_state[1] + 0.5))
        self.ax.imshow(self.finish_line_image, extent=(self.goal_state[0] - 0.5, self.goal_state[0] + 0.5, self.goal_state[1] - 0.5, self.goal_state[1] + 0.5))  # Finish Line
        for bldg in self.buildings:
            building_image = self.building_images[bldg["image"].split('.')[0]]
            self.ax.imshow(building_image, extent=(bldg["position"][0] - 0.5, bldg["position"][0] + 0.5, bldg["position"][1] - 0.5, bldg["position"][1] + 0.5))  # Buildings

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
    states = env.reset()
    for _ in range(500):
        actions = {
            "Dom": env.action_space.sample(),
            "Brian": env.action_space.sample()
        }  # Sample random actions for both agents
        states, rewards, dones, infos = env.step(actions)
        env.render()
        print(f"States: {states}, Rewards: {rewards}, Done: {any(dones.values())}, Infos: {infos}")
        if any(dones.values()):
            if dones["Dom"]:
                print("Dom wins, Brian loses!")
            elif dones["Brian"]:
                print("Brian wins, Dom loses!")
            break
    env.close()
