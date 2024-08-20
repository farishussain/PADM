# Imports:
# --------
from padm_env import PadmEnv
from q_learning import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True
visualize_results = True

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1000  # Number of episodes

# Execute:
# --------
if train:
    env = PadmEnv()  # Create an instance of the environment

    goal_coordinates = tuple(env.goal_state)
    hell_state_coordinates = [tuple(blocker["position"]) for blocker in env.blockers]

    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                      goal_coordinates=goal_coordinates,
                      q_values_path="q_table.npy")