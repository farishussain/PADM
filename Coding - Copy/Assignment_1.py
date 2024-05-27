import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class FastFuriousEnv(gym.Env):
    def __init__(self, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        self.agent_states = {
            "Dom": np.array([0, 0]),
            "Brian": np.array([6, 6])
        }  # Initial positions for two agents
        self.goal_state = np.array([3, 3])  # Single goal position
        self.obstacles = [np.array([3, 4]), np.array([4, 3]), np.array([2, 2])]  # Obstacles
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size - 1, shape=(2,))
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self):
        self.agent_states = {
            "Dom": np.array([0, 0]),
            "Brian": np.array([6, 6])
        }  # Reset initial positions for two agents
        return self.agent_states

    def step(self, actions):
        rewards = {}
        infos = {}
        dones = {}

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

            # Check for obstacle collision or collision with another agent
            if any(np.array_equal(new_state, obs) for obs in self.obstacles) or any(np.array_equal(new_state, other_state) for other_agent, other_state in self.agent_states.items() if other_agent != agent):
                reward = -1
                done = False
                info = {"Collision": True}
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
        self.ax.clear()
        for agent, state in self.agent_states.items():
            self.ax.plot(state[0], state[1], "ro", label=agent)  # agents
        self.ax.plot(self.goal_state[0], self.goal_state[1], "g+", label="Goal")  # goal
        for obs in self.obstacles:
            self.ax.plot(obs[0], obs[1], "ks")  # obstacles
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.legend()
        plt.pause(0.1)

    def close(self):
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
