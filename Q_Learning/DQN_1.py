import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

grid_size = 5

alpha = 0.1
gamma = 0.9
epsilon = 0.7
episodes = 1000
batch_size = 64
target_update = 10

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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)
    
def compute_loss(batch, model, target_model, gamma):
    state, action, reward, next_state, done = batch
    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = model(state)
    next_q_values = target_model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    return loss
    
env_shape = (grid_size, grid_size)
input_dim = grid_size * grid_size
output_dim = len(actions)

policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
replay_buffer = ReplayBuffer(10000)

for episode in range(episodes):
    state = (0, 0)
    state_one_hot = np.zeros(input_dim)
    state_one_hot[state[0] * grid_size + state[1]] = 1
    total_reward = 0
    while state != (4, 4):
        if np.random.rand() < epsilon:
            action = np.random.choice(len(actions))
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state_one_hot)).argmax().item()

        next_state_ = next_state(state, actions[action])
        next_state_one_hot = np.zeros(input_dim)
        next_state_one_hot[next_state_[0] * grid_size + next_state_[1]] = 1
        reward = R[next_state_]
        done = 1 if next_state_ == (4, 4) else 0

        replay_buffer.push(state_one_hot, action, reward, next_state_one_hot, done)
        state = next_state_
        state_one_hot = next_state_one_hot
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_loss(batch, policy_net, target_net, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")

state = (0, 0)
state_one_hot = np.zeros(input_dim)
state_one_hot[state[0] * grid_size + state[1]] = 1
optimal_path = [state]
while state != (4, 4):
    with torch.no_grad():
        action = policy_net(torch.FloatTensor(state_one_hot)).argmax().item()
    state = next_state(state, actions[action])
    state_one_hot = np.zeros(input_dim)
    state_one_hot[state[0] * grid_size + state[1]] = 1
    optimal_path.append(state)
print("최적 경로: ", optimal_path)
