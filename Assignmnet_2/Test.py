import numpy as np

# Define the grid-world parameters
size = 4
gamma = 0.9
states = [(i, j) for i in range(1, size + 1) for j in range(1, size + 1)]
actions = ['Up', 'Down', 'Left', 'Right']
transition_prob = {
    'intended': 0.7,
    'unintended': 0.1
}
danger_states = [(3, 2), (3, 4)]
goal_state = (4, 4)
living_reward = -0.1

# Initialize value function
V = {state: 0 for state in states}
V[goal_state] = 10
for danger_state in danger_states:
    V[danger_state] = -1

def get_next_state(state, action):
    i, j = state
    if action == 'Up':
        return (max(i-1, 1), j)
    elif action == 'Down':
        return (min(i+1, size), j)
    elif action == 'Left':
        return (i, max(j-1, 1))
    elif action == 'Right':
        return (i, min(j+1, size))

def get_transition_states(state, action):
    intended = get_next_state(state, action)
    if action == 'Up':
        unintended = [get_next_state(state, 'Left'), get_next_state(state, 'Right'), get_next_state(state, 'Down')]
    elif action == 'Down':
        unintended = [get_next_state(state, 'Left'), get_next_state(state, 'Right'), get_next_state(state, 'Up')]
    elif action == 'Left':
        unintended = [get_next_state(state, 'Up'), get_next_state(state, 'Down'), get_next_state(state, 'Right')]
    elif action == 'Right':
        unintended = [get_next_state(state, 'Up'), get_next_state(state, 'Down'), get_next_state(state, 'Left')]
    return intended, unintended

def reward(state):
    if state in danger_states:
        return -1
    elif state == goal_state:
        return 10
    else:
        return living_reward

# Function to print the value function in grid format
def print_value_function(V, size):
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            print(f"{V[(i, j)]:.2f}", end="\t")
        print()
    print()

# Iteratively calculate V_k for k = 1 to 5
for k in range(1, 6):
    V_new = V.copy()
    for state in states:
        action_values = []
        for action in actions:
            intended, unintended = get_transition_states(state, action)
            value = transition_prob['intended'] * (reward(intended) + gamma * V[intended])
            for unintended_state in unintended:
                value += transition_prob['unintended'] * (reward(unintended_state) + gamma * V[unintended_state])
            action_values.append(value)
        V_new[state] = max(action_values)  # Include the living reward for the current state
    V = V_new
    print(f"V{k}:")
    print_value_function(V, size)

# V4 values
V = np.array([
    [-0.35, -0.36,  4.26,  7.55],
    [-0.37,  4.34,  8.56, 15.86],
    [ 3.69,  8.61, 17.69, 27.56],
    [ 8.03, 16.60, 27.73, 30.95]
])

# Parameters
gamma = 0.9
danger_states = [(2, 1), (2, 3)]
goal_state = (3, 3)
reward_danger = -0.1
reward_goal = 10

# Transition probabilities
P = {
    "intended": 0.7,
    "other": 0.1
}

# Rewards function
def get_reward(s):
    if s == goal_state:
        return reward_goal
    elif s in danger_states:
        return reward_danger
    else:
        return -0.1

# Corrected helper function to handle out-of-bounds by returning the current state value
def get_value_corrected(V, current_state, intended_state):
    r, c = intended_state
    cr, cc = current_state
    if 0 <= r < V.shape[0] and 0 <= c < V.shape[1]:
        return V[r, c]
    return V[cr, cc]

# Actions and their effects
actions = {
    "Right": (0, 1),
    "Left": (0, -1),
    "Up": (-1, 0),
    "Down": (1, 0)
}

# Compute Q-values for all states and actions with corrected boundary handling
Q_corrected = np.zeros((V.shape[0], V.shape[1], len(actions)))

for r in range(V.shape[0]):
    for c in range(V.shape[1]):
        for i, (action, (dr, dc)) in enumerate(actions.items()):
            s = (r, c)
            intended_state = (r + dr, c + dc)
            other_states = [
                (r + actions["Right"][0], c + actions["Right"][1]),
                (r + actions["Left"][0], c + actions["Left"][1]),
                (r + actions["Up"][0], c + actions["Up"][1]),
                (r + actions["Down"][0], c + actions["Down"][1])
            ]
            other_states.remove(intended_state)
            
            reward = get_reward(intended_state)
            value = (P["intended"] * get_value_corrected(V, s, intended_state) +
                     P["other"] * get_value_corrected(V, s, other_states[0]) +
                     P["other"] * get_value_corrected(V, s, other_states[1]) +
                     P["other"] * get_value_corrected(V, s, other_states[2]))
            
            Q_corrected[r, c, i] = reward + gamma * value

print(Q_corrected)
