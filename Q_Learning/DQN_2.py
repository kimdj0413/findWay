import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 환경 파라미터
grid_size = 5
episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
memory_size = 10000

# 보상 함수 설정
R = np.full((grid_size, grid_size), -10)
R[4, 4] = 1000
R[0, 1] = -10000
R[2, 3] = -10000
actions = ["up", "down", "left", "right"]

# Q-Network 정의
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 경험 재생 메모리
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# 다음 상태 계산 함수
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

# DQN 학습 함수
def train_dqn():
    global epsilon 
    for episode in range(episodes):
        state = (0, 0)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        while state != (4, 4):
            # Epsilon-Greedy 정책에 따라 행동 선택
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = actions[torch.argmax(q_values).item()]
            
            next_state_ = next_state(state, action)
            reward = R[next_state_]
            next_state_tensor = torch.tensor(next_state_, dtype=torch.float32).unsqueeze(0)
            
            # 경험 저장
            memory.push((state_tensor, actions.index(action), reward, next_state_tensor))
            
            # 학습 수행
            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
                
                batch_state = torch.cat(batch_state)
                batch_action = torch.tensor(batch_action)
                batch_reward = torch.tensor(batch_reward)
                batch_next_state = torch.cat(batch_next_state)
                
                current_q_values = policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(batch_next_state).max(1)[0].detach()
                expected_q_values = batch_reward + (gamma * next_q_values)
                
                loss = criterion(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state_
            state_tensor = next_state_tensor
        
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

# 초기화
state_size = 2
action_size = len(actions)

policy_net = QNetwork(state_size, action_size)
target_net = QNetwork(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
criterion = nn.MSELoss()
memory = ReplayMemory(memory_size)

# DQN 학습 시작
train_dqn()

# 최적 경로 찾기
state = (0, 0)
optimal_path = [state]
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
while state != (4, 4):
    with torch.no_grad():
        action = actions[torch.argmax(policy_net(state_tensor)).item()]
    state = next_state(state, action)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    optimal_path.append(state)

print("최적 경로: ", optimal_path)
