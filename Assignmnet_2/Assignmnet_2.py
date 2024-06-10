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
    if state in danger_states:
        i, j = state
        if action == 'Up':
            return (max(i, 1), j)
        elif action == 'Down':
            return (min(i, size), j)
        elif action == 'Left':
            return (i, max(j, 1))
        elif action == 'Right':
            return (i, min(j, size))
    elif state == goal_state:
        i, j = state
        if action == 'Up':
            return (max(i, 1), j)
        elif action == 'Down':
            return (min(i, size), j)
        elif action == 'Left':
            return (i, max(j, 1))
        elif action == 'Right':
            return (i, min(j, size))
    else:
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