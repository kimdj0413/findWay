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
        self.fc = nn.Linear(input_size,output_size)
    
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
        self.replay_buffer = ReplayBuffer(100)
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
        
def train(agent, env, num_episodes, batch_size):
    for episode in range(num_episodes):
        state = env.reset()
        epsilon = 1-episode/num_episodes
        # reward = 0
        done = False
        state_list=[]

        while not done:
            action = agent.select_action(state,epsilon)
            next_state, reward, done = env.move(state, action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.update_model(batch_size)
            loss = agent.update_model(batch_size)
            state = next_state
            state_list.append(next_state)
        
        print(f'에피소드: {episode}, state: {state}, 보상: {reward}, Loss: {loss}')
        print(state_list)

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name))
    model.eval()

input_size = 2
output_size = 4
hidden_size = 64
batch_size = 32
num_episodes = 3000

env = GridSetting()
agent = DQNAgent(input_size, output_size, hidden_size)
train(agent, env, num_episodes, batch_size)
 # 모델 저장
save_model(agent.model, 'dqn_model.pth')