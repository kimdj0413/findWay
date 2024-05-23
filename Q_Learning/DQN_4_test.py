import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class GridSetting:
    def __init__(self):
        self.grid_size_x = 5
        self.grid_size_y = 5
        self.avoid = [(1,2),(0,3),(3,4)]
        self.start_state = (0,0)
        self.end_state = (self.grid_size_x-1,self.grid_size_y-1)

    def reset(self):
        self.state = self.start_state
        return self.state

    def move(self, state, action):
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
        reward = -1

        if next_state in self.avoid:
            reward = -10
            done =True
        elif next_state == self.end_state:
            reward =  100
            done = True
        else:
            reward = -1
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
        self.replay_buffer = ReplayBuffer(10000)
        self.model = QNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters())

    def update_model(self,batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # with torch.no_grad():
        next_q_values = self.model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def select_action(self, state, epsilon):
        state = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < epsilon:
            return random.randint(0, 3)
        else :
            q_values = self.model(state)
            return np.argmax(q_values.detach().numpy())
    def sort_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return np.argsort(q_values.detach().numpy().squeeze())[::1]
        
def train(agent, env, num_episodes, batch_size):
    for episode in range(num_episodes):
        state = env.reset()
        epsilon = 1.0-episode/num_episodes
        # reward = 0
        done = False
        state_list=[]

        while not done:
            action = agent.select_action(state,epsilon)
            next_state, reward, done = env.move(state, action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            # agent.update_model(batch_size)
            loss = agent.update_model(batch_size)
            state = next_state
            state_list.append(next_state)
        
        print(f'에피소드: {episode}, state: {state}, 보상: {reward}, Loss: {loss}')
        # print(state_list)

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name))
    model.eval()
    print(model)
    return model

input_size = 2
output_size = 4
hidden_size = 64
batch_size = 32
num_episodes = 3000

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
            next_state, _, done = env.move(state, i)
            if next_state in visited:
                continue
            else:
                optimal_path.append(next_state)
                visited.add(next_state)
                action_path.append(action_set[i])
                state = next_state
                break
    optimal_path.append(env.end_state)
    return optimal_path, action_path

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

env = GridSetting()
new_agent = DQNAgent(input_size, output_size, hidden_size)
load_model(new_agent.model, 'dqn_model.pth')

optimal_path, action_path= find_optimal_path(env, new_agent, env.start_state)
print(f'최적의 경로: {optimal_path,action_path}')