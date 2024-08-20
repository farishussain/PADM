# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gymnasium as gym

# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Print the initialized Q-table
    print("Initialized Q-table:")
    print(q_table)

    # Q-learning algorithm:
    # ---------------------
    for episode in range(no_episodes):
        state = env.reset()

        state = tuple(state)
        total_reward = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, info = env.step(action)
            # env.render()

            next_state = tuple(next_state)
            total_reward += reward

            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    print("Training finished.\n")

    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")

# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(hell_state_coordinates=[(4, 4)],
                      goal_coordinates=(3, 3),
                      actions=["Up", "Down", "Left", "Right"],
                      q_values_path="q_table.npy"):
    try:
        q_table = np.load(q_values_path)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask out goal and blocker states
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True
            for hell_coord in hell_state_coordinates:
                mask[hell_coord] = True

            # Invert the y-axis by setting origin to 'upper'
            sns.heatmap(heatmap_data.T, annot=True, fmt=".2f", cmap="viridis",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9},
                        square=True, xticklabels=True, yticklabels=True)

            # Add labels for goal and blockers
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            for hell_coord in hell_state_coordinates:
                ax.text(hell_coord[1] + 0.5, hell_coord[0] + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')

            # Invert y-axis to start from the top
            ax.invert_yaxis()

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
