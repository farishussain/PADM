import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gymnasium import spaces

class PadmEnv(gym.Env):
    """
    Custom Environment for a PADM course completion simulation.
    The student must attend lectures (states), complete assignments, give presentation, and give an exam while avoiding deregistration (blocker).
    """

    def __init__(self, grid_size=7, goal_coordinates=(3,3)):
        """
        Initialize the environment.
        Parameters:
        grid_size (int): The size of the grid.
        """
        super(PadmEnv, self).__init__()
        self.grid_size = grid_size  # Size of the grid
        self.student_state = np.array([0, 0])  # Initial position of the student
        self.goal_state = np.array(goal_coordinates)  # Position of the exam (goal)
        
        # Define the blockers in the environment
        self.blockers = [
            {"position": np.array([4, 4]), "image": 'Cancel'}
        ]
        
        # Define tasks with their positions and rewards
        self.tasks = [
            {"position": np.array([2, 5]), "reward": 0, "name": "Assignment 1"},
            {"position": np.array([5, 2]), "reward": 0, "name": "Assignment 2"},
            {"position": np.array([4, 3]), "reward": 0, "name": "Presentation"}
        ]
        
        # Define the action and observation spaces
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
        
        # Load images for the student, blockers, tasks, presentation, and exam
        self.student_image = mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Assignment_3/Coding/Assignment_1/Images/student.png')
        self.blocker_images = {
            "Cancel": mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Assignment_3/Coding/Assignment_1/Images/Cancel.png')
        }
        self.task_image = mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Assignment_3/Coding/Assignment_1/Images/assignment.png')
        self.presentation_image = mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Assignment_3/Coding/Assignment_1/Images/presentation.png')
        self.finish_line_image = mpimg.imread('C:/Users/FarisHussain/OneDrive/Doc/Ingolstadt/THI/PADM/Assignment_3/Coding/Assignment_1/Images/exam.png')

        # Initialize plot
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self):
        """
        Reset the environment to its initial state.
        Returns:
        np.array: The initial state of the student.
        """
        self.student_state = np.array([0, 0])  # Reset initial position of the student
        
        # Reset tasks
        self.tasks = [
            {"position": np.array([2, 5]), "reward": 0, "name": "Assignment 1"},
            {"position": np.array([5, 2]), "reward": 0, "name": "Assignment 2"},
            {"position": np.array([4, 3]), "reward": 0, "name": "Presentation"}
        ]
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
        days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))  # Calculate days to exam based on distance
        info = {"days_to_exam": days_to_exam}

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
            reward = blocker_penalty  # Penalty for hitting a blocker
            done = False
            days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
            info = {"blocked": "deregistered from exam", "days_to_exam": days_to_exam}
        else:
            # Check if the new state is within the grid boundaries
            if (0 <= new_state[0] < self.grid_size) and (0 <= new_state[1] < self.grid_size):
                self.student_state = new_state
                reward = 0
                # Check for task completion
                completed_task_index = None
                for idx, task in enumerate(self.tasks):
                    if np.array_equal(new_state, task["position"]):
                        reward += task["reward"]  # Reward for completing the task
                        completed_task_index = idx
                        days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
                        info = {"task_completed": task["name"], "days_to_exam": days_to_exam}
                        break
                if completed_task_index is not None:
                    del self.tasks[completed_task_index]
                # Check if all tasks are completed and student has reached the examination
                if np.array_equal(new_state, self.goal_state) and len(self.tasks) == 0:
                    reward += 100  # Bonus reward for completing all tasks and reaching the goal
                    done = True
                    days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
                    info = {"course_completed": True, "days_to_exam": days_to_exam}
                elif np.array_equal(new_state, self.goal_state):
                    done = False
                    days_to_exam = int(np.linalg.norm(self.student_state - self.goal_state))
                    info = {"examination_without_completing_all_tasks": False, "days_to_exam": days_to_exam}
            else:
                # Student attempted to move out of bounds
                reward = -1000  # Penalty for attempting to move out of bounds
                done = False
                info = {"attempted_to_move_out_of_bounds": True}

        return self.student_state, reward, done, info

    def render(self):
        """
        Render the current state of the environment.
        """
        self.ax.clear()
        self.ax.imshow(self.student_image, extent=(self.student_state[1] - 0.5, self.student_state[1] + 0.5, self.student_state[0] - 0.5, self.student_state[0] + 0.5))
        self.ax.imshow(self.finish_line_image, extent=(self.goal_state[1] - 0.5, self.goal_state[1] + 0.5, self.goal_state[0] - 0.5, self.goal_state[0] + 0.5))  # Examination
        for blk in self.blockers:
            blocker_image = self.blocker_images[blk["image"]]
            self.ax.imshow(blocker_image, extent=(blk["position"][1] - 0.5, blk["position"][1] + 0.5, blk["position"][0] - 0.5, blk["position"][0] + 0.5))  # Blockers
        for task in self.tasks:
            if task["name"] == "Presentation":
                self.ax.imshow(self.presentation_image, extent=(task["position"][1] - 0.5, task["position"][1] + 0.5, task["position"][0] - 0.5, task["position"][0] + 0.5))  # Presentation
            else:
                self.ax.imshow(self.task_image, extent=(task["position"][1] - 0.5, task["position"][1] + 0.5, task["position"][0] - 0.5, task["position"][0] + 0.5))  # Tasks

        # Turn off the axis
        self.ax.axis('on')

        # Set axis limits
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")

        plt.pause(0.1)

    def close(self):
        """
        Close the rendering window.
        """
        plt.close()

    def add_hell_states(self, hell_state_coordinates):
        """
        Add blockers dynamically to the environment.
        Parameters:
        hell_state_coordinates (tuple): The coordinates of the new blocker.
        """
        self.blockers.append({"position": np.array(hell_state_coordinates), "image": "Cancel"})

### Main loop to run the environment

if __name__ == "__main__":
    env = PadmEnv()  # Initialize with custom grid size
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

# Function 1: Create an instance of the environment
# -----------
def create_env(goal_coordinates, hell_state_coordinates):
    # Create the environment:
    # -----------------------
    env = PadmEnv(goal_coordinates=goal_coordinates)

    for i in range(len(hell_state_coordinates)):
        env.add_hell_states(hell_state_coordinates=hell_state_coordinates[i])

    return env