import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# 환경 설정
class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.obstacles = [(1, 2), (0, 3), (3, 4)]
        self.start_state = (0, 0)
        self.end_state = (4, 4)
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:   # 상
            x = max(x - 1,0)
        elif action == 1: # 하
            x = min(x + 1, self.grid_size - 1)
        elif action == 2: # 좌
            y = max(y - 1,0)
        elif action == 3: # 우
            y = min(y + 1,self.grid_size - 1)

        next_state = (x, y)
        if next_state in self.obstacles:
            reward = -1000
            done =True
        elif next_state == self.end_state:
            reward =  1000
            done = True
        else:
            reward = -5
            done = False

        self.state = next_state
        return next_state, reward, done

# Q-네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, state):
        return self.fc(state)

# 리플레이 버퍼
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

# DQN 에이전트
class DQNAgent:
    def __init__(self, input_size, output_size, hidden_size=64, gamma=0.99):
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(1000)
        self.model = QNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters())

    def update_model(self, batch_size):
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
        next_q_values = self.model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            state = torch.FloatTensor(state)
            q_values = self.model(state)
            return np.argmax(q_values.detach().numpy())

# 학습 루프
def train(agent, env, num_episodes, batch_size):
    for episode in range(num_episodes):
        state = env.reset()
        epsilon = 1-episode/num_episodes # max(0.01, 0.08 - 0.01 * (episode / 200))  Epsilon-greedy 정책
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            agent.update_model(batch_size)
            loss = agent.update_model(batch_size)

        # if episode % 100 == 0:
        print(f'에피소드: {episode}, 보상: {reward}, Loss: {loss}')
# 모델 저장 함수
def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

# 모델 불러오기 함수
def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name))
    model.eval()  # 평가 모드로 설정

# 최적 경로 찾기 함수
def find_optimal_path(env, agent, start_state):
    state = start_state
    optimal_path = [state]
    action_path = []
    while state != env.end_state:
        action = agent.select_action(state, epsilon=0)  # 최적의 행동 선택
        next_state, _, done = env.step(action)
        action_path.append(action)
        optimal_path.append(next_state)
        state = next_state
    return optimal_path, action_path

# 메인
input_size = 2 # 상태는 (x, y) 좌표
output_size = 4 # 상, 하, 좌, 우
hidden_size = 64
batch_size = 32
num_episodes = 1000

env = GridWorld()
agent = DQNAgent(input_size, output_size, hidden_size)
train(agent, env, num_episodes, batch_size)
 # 모델 저장
save_model(agent.model, 'dqn_model.pth')
    
    # 새로운 에이전트 생성 및 모델 불러오기
new_agent = DQNAgent(input_size, output_size, hidden_size)
load_model(new_agent.model, 'dqn_model.pth')
    
    # 최적 경로 찾기
optimal_path, action_path = find_optimal_path(env, new_agent, env.start_state)
print(f'최적의 경로: {optimal_path}')
print(f'최적의 행동: {action_path}')