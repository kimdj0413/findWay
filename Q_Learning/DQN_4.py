import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import matplotlib.pyplot as plt
from PIL import Image

start_time = time.time()
class GridSetting:
    def __init__(self):
        self.grid_size_x = 10
        self.grid_size_y = 10
        self.avoid = [(0,5),(0,6),(0,7),(0,8),(0,9),(1,0),(1,1),(1,2),(1,5),(1,6),(1,7),(1,8),(1,9),(2,0),(2,1),(2,2),(2,5),(2,6),(2,7),(2,8),(2,9),(3,0),(3,1),(3,2),(3,5),(3,6),(3,7),(3,8),(3,9),(4,9),(5,4),(5,5),(5,6),(5,7),(5,9),(6,0),(6,1),(6,2),(6,4),(6,5),(6,6),(6,7),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2),(8,5),(8,6),(8,7),(9,5),(9,6),(9,7)]
        # self.avoid = [(0,1),(1,1),(1,3),(2,3),(3,0),(3,1),(3,2),(3,3),(4,1),(4,2)]
        self.start_state = (0,0)
        self.force_togo = 3
        self.end_state = (self.grid_size_x-1,self.grid_size_y-1)

    def reset(self):
        self.state = self.start_state
        return self.state

    def move(self, state, action, reward):
        x,y = state
        if action == 0:   # 상
            x = max(x - 1,0)
        elif action == 1: # 하
            x = min(x + 1, self.grid_size_x - 1)
        elif action == 2: # 좌
            y = max(y - 1,0)
        elif action == 3: # 우
            y = min(y + 1,self.grid_size_y - 1)
        
        next_state = (x,y)

        if next_state in self.avoid:
            reward += -1
            done = False
        elif next_state == self.end_state:
            reward += 1
            done = True
        else:
            reward += -0.02
            done = False
        
        return next_state, reward, done

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, state):
        return self.fc(state)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, input_size, output_size, hidden_size=64, gamma=0.99):
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(3000)
        self.model = QNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def update_model(self,batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        if actions is None:
            print(states,actions,rewards,next_states,dones)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.SmoothL1Loss()(current_q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def select_action(self, state, epsilon, cnt):
        state = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < epsilon:
            return random.randint(0, 3)
        else :
            q_values = self.model(state)
            action = np.argsort(q_values.detach().numpy().squeeze())[::1]
            if cnt == 0:
                return action[0]
            if cnt == 1:
                return action[1]
            if cnt == 2:
                return action[2]
            if cnt == 3:
                return action[3]
        
    def sort_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return np.argsort(q_values.detach().numpy().squeeze())[::1]
losses=[]
loss_cnt=[]   
def train(agent, env, num_episodes, batch_size):
    with open('training_log.txt', 'w') as file:
        for episode in range(num_episodes):
            state = env.reset()
            visited=set()
            epsilon = max(0.001, 1.0 - episode / num_episodes)#1.0-episode/num_episodes
            done = False
            state_list=[]
            reward = 0.0
            visited.add(state)
            cnt = 0

            while not done:
                while(True):
                    action = agent.select_action(state,epsilon,cnt)
                    next_state, reward, done = env.move(state, action, reward)
                    if next_state not in visited:
                        visited.add(next_state)
                        break
                    else :
                        cnt += 1
                    if cnt > 4:
                        action = env.force_togo
                        break
                cnt = 0
                agent.replay_buffer.push(state, action, reward, next_state, done)
                loss = agent.update_model(batch_size)
                state = next_state
                state_list.append(next_state)
            if loss is not None:
                    loss = loss/100
            else:
                loss = 0
            losses.append(loss)
            loss_cnt.append(episode)
            file.write(f'에피소드: {episode}, Loss: {loss:.2f}\n')
            print(f'에피소드: {episode}, Loss: {loss:.2f}')
           
def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name))
    model.eval()

input_size = 2
output_size = 4
hidden_size = 128
batch_size = 64
num_episodes = 1000

env = GridSetting()
agent = DQNAgent(input_size, output_size, hidden_size)
train(agent, env, num_episodes, batch_size)

save_model(agent.model, 'dqn_test_model.pth')

def find_optimal_path(env, agent, start_state):
    visited = set()
    state = start_state
    optimal_path = [start_state]
    action_path = []
    
    visited.add(state)
    for i in range(0,len(env.avoid)):
        visited.add(env.avoid[i])
    while state != env.end_state:
        action = agent.sort_action(state)
        for i in action:
            next_state, _, done = env.move(state, i,0)
            if next_state in visited:
                continue
            else:
                optimal_path.append(next_state)
                visited.add(next_state)
                action_path.append(action_set[i])
                state = next_state
                # print(optimal_path)
                break
    optimal_path.append(env.end_state)
    return optimal_path, action_path

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

new_agent = DQNAgent(input_size, output_size, hidden_size)
load_model(new_agent.model, 'dqn_test_model.pth')

optimal_path, action_path= find_optimal_path(env, new_agent, env.start_state)
print(f'최적의 경로: {optimal_path,action_path}')

sec = time.time()-start_time
times = str(datetime.timedelta(seconds=sec))
short = times.split(".")[0]
print(f"{short} sec")

# 그림 그리기
plt.ion()
# 손실 함수 그리기
fig2, ax2 = plt.subplots()

loss_x = loss_cnt
loss_y = losses

ax2.plot(loss_x, loss_y)
ax2.set_title('losses Graph')
ax2.set_xlabel('Episodes')
ax2.set_ylabel('Losses')

plt.show()

fig1, ax1 = plt.subplots()

for x in range(env.grid_size_x + 1):
    ax1.axhline(x, lw=2, color='k', zorder=5)
for y in range(env.grid_size_y + 1):
    ax1.axvline(y, lw=2, color='k', zorder=5)

ax1.set_xticks(np.arange(env.grid_size_x + 1))
ax1.set_yticks(np.arange(env.grid_size_y + 1))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.grid(True)

for j in range(len(env.avoid)):
    ax1.add_patch(plt.Rectangle((env.avoid[j][1], env.avoid[j][0]), 1, 1, color='red', alpha=0.5))

for x in range(env.grid_size_x):
    for y in range(env.grid_size_y):
        ax1.text(y + 0.5, x + 0.5, f'({x},{y})', ha='center', va='center', fontsize=5)

ax1.text(env.start_state[1] + 0.1, env.start_state[0] + 1 - 0.1, 'Start', ha='left', va='bottom', fontweight='bold', fontsize=7, color='blue', zorder=10)
ax1.text(env.end_state[1] + 0.1, env.end_state[0] + 1 - 0.1, 'Goal', ha='left', va='bottom', fontweight='bold', fontsize=7, color='red', zorder=10)

for i in range(len(optimal_path) - 1):
    start = optimal_path[i]
    end = optimal_path[i + 1]
    ax1.plot([start[1] + 0.5, end[1] + 0.5], [start[0] + 0.5, end[0] + 0.5], 'o-', color='green', lw=1, markersize=3, zorder=10)

plt.gca().invert_yaxis()
plt.axis('equal')
# plt.savefig("DQN_vector.svg", format="svg")
plt.show()

input("Enter")