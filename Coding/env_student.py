import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gymnasium import spaces

class PADMEnv(gym.Env):
    """
    Custom Environment for a PADM course completion simulation.
    The student must attend lectures and labs, complete assignments, presentations, and give an exam while avoiding blockers.
    """

    def __init__(self, grid_size=7):
        """
        Initialize the environment.
        Parameters:
        grid_size (int): The size of the grid.
        """
        super(PADMEnv, self).__init__()
        self.grid_size = grid_size
        self.student_state = np.array([0, 0])  # Initial position for the student
        self.goal_state = np.array([grid_size // 2, grid_size // 2])  # Goal position
        self.blockers = [
            {"position": np.array([3, 4]), "image": 'Cancel'}
        ]  # Static blockers
        self.tasks = [
            {"position": np.array([2, 5]), "reward": 20, "name": "Assignment 1"},
            {"position": np.array([5, 2]), "reward": 10, "name": "Assignment 2"},
            {"position": np.array([4, 3]), "reward": 30, "name": "Presentation"}
        ]  # Tasks with different rewards
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int64)
        
        # Load images
        self.student_image = mpimg.imread('Coding/student.png')
        self.blocker_images = {
            "Cancel": mpimg.imread('Coding/Cancel.png')
        }
        self.task_image = mpimg.imread('Coding/assignment.png')
        self.presentation_image = mpimg.imread('Coding/presentation.png')
        self.finish_line_image = mpimg.imread('Coding/exam.png')

        # Initialize plot
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self):
        """
        Reset the environment to its initial state.
        Returns:
        np.array: The initial state of the student.
        """
        self.student_state = np.array([0, 0])  # Reset initial position for the student
        self.tasks = [
            {"position": np.array([2, 5]), "reward": 20, "name": "Assignment 1"},
            {"position": np.array([5, 2]), "reward": 10, "name": "Assignment 2"},
            {"position": np.array([4, 3]), "reward": 30, "name": "Presentation"}
        ]  # Reset tasks
        return self.student_state

    def step(self, action):
        """
        Execute actions and return the new state, reward, done, and info.
        Parameters:
        action (int): The action for the student.
        Returns:
        np.array: The new state of the student.
        float: The reward for the student.
        bool: Whether the episode is done.
        dict: Additional information.
        """
        reward = 0
        done = False
        days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
        info = {"Days to Exam": days_to_exam}

        # Define the penalties and rewards
        blocker_penalty = -3

        # Process student's action
        new_state = self.student_state.copy()
        if action == 0 and new_state[1] < self.grid_size - 1:  # up
            new_state[1] += 1
        elif action == 1 and new_state[1] > 0:  # down
            new_state[1] -= 1
        elif action == 2 and new_state[0] > 0:  # left
            new_state[0] -= 1
        elif action == 3 and new_state[0] < self.grid_size - 1:  # right
            new_state[0] += 1

        # Check for collisions with blockers
        if any(np.array_equal(new_state, blk["position"]) for blk in self.blockers):
            reward = blocker_penalty
            done = False
            days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
            info = {"Blocked": "Deregistered from Exam", "Days to Exam": days_to_exam}
        else:
            # Check if the new state is within the grid boundaries
            if (0 <= new_state[0] < self.grid_size) and (0 <= new_state[1] < self.grid_size):
                self.student_state = new_state
                reward = 0
                # Check for task completion
                completed_task_index = None
                for idx, task in enumerate(self.tasks):
                    if np.array_equal(new_state, task["position"]):
                        reward += task["reward"]
                        completed_task_index = idx
                        days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
                        info = {"Task Completed": task["name"], "Days to Exam": days_to_exam}
                        break
                if completed_task_index is not None:
                    del self.tasks[completed_task_index]
                # Check if all tasks are completed and student has reached the examination
                if np.array_equal(new_state, self.goal_state) and len(self.tasks) == 0:
                    reward += 50
                    done = True
                    days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
                    info = {"Course Completed": True, "Days to Exam": days_to_exam}
                elif np.array_equal(new_state, self.goal_state):
                    done = False
                    days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
                    info = {"Examination without completing all Tasks": False, "Days to Exam": days_to_exam}
            else:
                # Student attempted to move out of bounds
                reward = -5  # Penalty for attempting to move out of bounds
                done = False
                info = {"Attempted to move out of bounds"}

        return self.student_state, reward, done, info

    def render(self):
        """
        Render the current state of the environment.
        """
        self.ax.clear()
        self.ax.imshow(self.student_image, extent=(self.student_state[0] - 0.5, self.student_state[0] + 0.5, self.student_state[1] - 0.5, self.student_state[1] + 0.5))
        self.ax.imshow(self.finish_line_image, extent=(self.goal_state[0] - 0.5, self.goal_state[0] + 0.5, self.goal_state[1] - 0.5, self.goal_state[1] + 0.5))  # Examination
        for blk in self.blockers:
            blocker_image = self.blocker_images[blk["image"]]
            self.ax.imshow(blocker_image, extent=(blk["position"][0] - 0.5, blk["position"][0] + 0.5, blk["position"][1] - 0.5, blk["position"][1] + 0.5))  # Blockers
        for task in self.tasks:
            if task["name"] == "Presentation":
                self.ax.imshow(self.presentation_image, extent=(task["position"][0] - 0.5, task["position"][0] + 0.5, task["position"][1] - 0.5, task["position"][1] + 0.5))  # Presentation
            else:
                self.ax.imshow(self.task_image, extent=(task["position"][0] - 0.5, task["position"][0] + 0.5, task["position"][1] - 0.5, task["position"][1] + 0.5))  # Tasks

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
    env = PADMEnv(grid_size=7)  # Initialize with custom grid size
    state = env.reset()
    for _ in range(500):
        action = env.action_space.sample()  # Sample random action for the student
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("I reached the goal")
            break
    env.close()
