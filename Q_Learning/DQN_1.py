import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 5x5 그리드 환경 설정
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = [(0, 3), (3, 2)]

# DQN 신경망 모델 정의
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN 학습
def train_dqn():
    input_size = GRID_SIZE * GRID_SIZE
    output_size = 4  # 상/하/좌/우 4개의 행동

    model = DQN(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # 학습 코드 작성 (미로 환경에서 DQN을 학습)

    return model

# 최적 경로 찾기
def find_optimal_path(model):
    current_position = START
    path = [current_position]

    while current_position != GOAL:
        state = np.zeros(GRID_SIZE * GRID_SIZE)
        state[current_position[0] * GRID_SIZE + current_position[1]] = 1
        state_tensor = torch.tensor(state, dtype=torch.float32)

        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

        if action == 0:  # 상
            next_position = (current_position[0] - 1, current_position[1])
        elif action == 1:  # 하
            next_position = (current_position[0] + 1, current_position[1])
        elif action == 2:  # 좌
            next_position = (current_position[0], current_position[1] - 1)
        else:  # 우
            next_position = (current_position[0], current_position[1] + 1)

        if next_position not in OBSTACLES:
            current_position = next_position
            path.append(current_position)
        else:
            break

    return path

# DQN 학습 및 최적 경로 찾기
trained_model = train_dqn()
optimal_path = find_optimal_path(trained_model)

print("Optimal path:", optimal_path)
