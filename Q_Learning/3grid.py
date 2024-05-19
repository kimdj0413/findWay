import numpy as np

grid_size = 5

Q = np.zeros((grid_size, grid_size, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000

R = np.full((grid_size, grid_size), -10)
R[4,4] = 1000
R[0, 1] = -10000
R[2, 3] = -10000
actions = ["up", "down", "left", "right"]

def next_state(state, action):
    i, j = state
    if action == "up":
        i = max(i - 1, 0)
    elif action == "down":
        i = min(i + 1, grid_size - 1)
    elif action == "left":
        j = max(j - 1, 0)
    elif action == "right":
        j = min(j + 1, grid_size - 1)
    return i, j

for episode in range(episodes):
    state = (0, 0)
    while state != (4, 4):
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]
        
        next_state_ = next_state(state, action)
        reward = R[next_state_]
        
        Q[state[0], state[1], actions.index(action)] += alpha * (
            reward + gamma * np.max(Q[next_state_[0], next_state_[1]]) - Q[state[0], state[1], actions.index(action)])
        
        state = next_state_

state = (0, 0)
optimal_path = [state]
while state != (4, 4):
    action = actions[np.argmax(Q[state[0], state[1]])]
    state = next_state(state, action)
    optimal_path.append(state)
print("최적 경로: ", optimal_path)