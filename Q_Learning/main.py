from padm_env import create_env
from q_learning import train_q_learning, visualize_q_table


learning_rate = 0.01
gamma  = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
no_epsiodes = 1000

goal_coordinates = (4,4)
hell_state_coordinates = [(2,1), (0,4)]


env = create_env(goal_coordinates=goal_coordinates,
                 hell_state_coordinates=hell_state_coordinates)

train_q_learning(env=env, no_episodes=no_epsiodes,
                 epsilon=epsilon,
                 epsilon_min=epsilon_min,
                 epsilon_decay=epsilon_decay,
                 alpha=learning_rate,
                 gamma=gamma)

visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                  goal_coordinates=goal_coordinates)