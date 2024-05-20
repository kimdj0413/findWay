import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

grid_size = 20

alpha = 0.5
gamma = 0.9
epsilon = 0.7
episodes = 1000
batch_size = 64
target_update = 10

R = np.full((grid_size, grid_size), -50)
R[grid_size-1,grid_size-1] = 10000
must_avoid = [(9, 8), (3, 6), (8, 2), (15, 9), (13, 18), (18, 13), (2, 8), (14, 8), (4, 18), (12, 10), (4, 10), (4, 11), (18, 3), (3, 2), (5, 11), (19, 3), (15, 4), (9, 12), (13, 16), (14, 3), (14, 7), (1, 1), (4, 2), (14, 17), (19, 8), (1, 11), (13, 19), (2, 17), (11, 15), (0, 11), (19, 4), (10, 10), (13, 12), (19, 18), (10, 1), (18, 15), (13, 3), (4, 6), (6, 6), (5, 12), (5, 16), (19, 10), (9, 10), (15, 17), (5, 15), (2, 3), (5, 18), (3, 7), (1, 13), (4, 15)]
for j in range(len(must_avoid)):
    R[must_avoid[j]] = -200000
# R[0, 1] = -10000
# R[2, 3] = -10000
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

# 파이토치 nn.Module 상속
# 상속법은 __init__와 forward를 오버라이딩
# input_dim은 state, output_dim은 action
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  #128개의 뉴런을 가진 선형층
            nn.ReLU(),  # 활성화 함수
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 경험 재현 메모리 생성
# 에이전트가 경험한 상태, 행동, 보상, 다음상태, 종료여부 저장 후 랜덤하게 샘플링    
class ReplayBuffer:
    # deque는 collection. 가장 오래된 항목 제거
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    # 튜플 형태로 경험 저장
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    # batch_size는 샘플링 할 경험의 수
    def sample(self, batch_size):
        # random.sample로 buffer내에서 batch_size만큼 랜덤하게 선택해 리스트로 만듬
        # zip : *는 언패킹 연산자. 리스트의 요소들을 개별적인 인자로 전달해 요소별로 튜플을 만듬.
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)
    
def compute_loss(batch, model, target_model, gamma):
    # 데이터 타입을 텐서로 변환 -> PyTorch의 기본 데이터 구조는 텐서.(넘파이와 유사.)
    state, action, reward, next_state, done = batch
    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = model(state)
    next_q_values = target_model(next_state)
    # 현재 Q값 추출
    # action.unsqueeze(1) : 1차원 action을 2차원으로 변환
    # gather(1, index) : 각 index에 해당하는 q values의 해당 열의 값을 추출.
    # squeeze : 다시 1차원으로 축소
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # 다음 상태의 최대 Q 값의 인덱스를 제외하고 값만 추출
    next_q_value = next_q_values.max(1)[0]
    # 목표 Q 값을 계산(미래 보상)
    # 1-done은 에피소드가 종료되지 않은 경우에만 다음 Q 값 고려하게함.
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # 손실 값 계산
    # .detach() : 역전파 중에 expected_q_value가 업데이트되지 않게함.
    # .pow(2).mean() : 제곱 후 제곱 오차의 평균을 구함.
    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    return loss
    
env_shape = (grid_size, grid_size)
input_dim = grid_size * grid_size
output_dim = len(actions)

# 현재 정책을 나타내는 신경망. 즉, agent가 현재 상태에서 취해야 할 행동을 결정.
policy_net = DQN(input_dim, output_dim)
# 목표 정책을 나타내는 신경망. 고정된 목표.
target_net = DQN(input_dim, output_dim)
# policy_net의 가중치를 target_net에 복사함. 초기에는 같은 값을 가짐.
# .state_dict() : policy_net의 모든 가중치와 편향 값을 딕셔너리로 전환.
target_net.load_state_dict(policy_net.state_dict())
# .eval() : target_net을 평가 모드로 전환(드롭아웃이 비활성화 되고 출력이 일관.)
target_net.eval()
## target_net은 최신 policy_net의 상태를 반영.
## target은 일정한 주기마다 업데이트, policy는 매 스텝마다 업데이트.
## 학습의 안정성을 높이기 위해.

# 최적화 알고리즘 : adam은 경사하강법 기반의 최적화 알고리즘.
# policy_net.parameters() : 학습 가능한 파라미터(가중치, 편향)들을 전부 반환.
optimizer = optim.Adam(policy_net.parameters())
replay_buffer = ReplayBuffer(10000)

# 에피소드마다 최대 스텝 수 설정
max_steps_per_episode = 1000

for episode in range(episodes):
    state = (0, 0)
    state_one_hot = np.zeros(input_dim)
    state_one_hot[state[0] * grid_size + state[1]] = 1
    total_reward = 0
    steps = 0  # 스텝 카운터 초기화
    
    while state != (grid_size-1, grid_size-1) and steps < max_steps_per_episode:
        if np.random.rand() < epsilon:
            action = np.random.choice(len(actions))
        else:
            with torch.no_grad():
                action = policy_net(torch.FloatTensor(state_one_hot)).argmax().item()

        next_state_ = next_state(state, actions[action])
        next_state_one_hot = np.zeros(input_dim)
        next_state_one_hot[next_state_[0] * grid_size + next_state_[1]] = 1
        reward = R[next_state_]
        done = 1 if next_state_ == (grid_size-1, grid_size-1) else 0

        replay_buffer.push(state_one_hot, action, reward, next_state_one_hot, done)
        state = next_state_
        state_one_hot = next_state_one_hot
        total_reward += reward
        steps += 1  # 스텝 카운트 증가

        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_loss(batch, policy_net, target_net, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {steps}")

    # 무한 루프 방지를 위한 경고 메시지
    if steps >= max_steps_per_episode:
        print(f"Warning: Episode {episode} reached max steps without termination.")


state = (0, 0)
state_one_hot = np.zeros(input_dim)
state_one_hot[state[0] * grid_size + state[1]] = 1
optimal_path = [state]
while state != (grid_size-1, grid_size-1):
    with torch.no_grad():
        action = policy_net(torch.FloatTensor(state_one_hot)).argmax().item()
    state = next_state(state, actions[action])
    state_one_hot = np.zeros(input_dim)
    state_one_hot[state[0] * grid_size + state[1]] = 1
    optimal_path.append(state)
print("최적 경로: ", optimal_path)
