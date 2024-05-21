import numpy as np
import torch
import random
from matplotlib import pylab as plt

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

l1= 75
l2 = 256
l3 = 128
l4 = 4

model = torch.nn.Sequential(
        torch.nn.Linear(l1,l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2,l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4)
)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer  = torch.optim.Adam(model.parameters(),lr = learning_rate)

grid_size_x = 5
grid_size_y = 5
gamma = 0.9
epsilon = 1.0
epochs = 2000
losses = []

start=(0,0)
goal=(4,4)
avoid = (1,2),(1,3),(3,2)

# GridMake
def gridMake(grid_size_x, grid_size_y, start, goal, avoid):
        game = np.full((grid_size_x, grid_size_y), " ")
        game[start] = 'S'
        game[goal] = 'G'
        for pos in avoid:
            game[pos] = 'X'
        return game

# stateMake
def stateMake(grid_size_x, grid_size_y):
    state_ = np.zeros((3,grid_size_x, grid_size_y))
    state_[0, start[0],start[1]] = 1
    for i in range(0,len(avoid)):
        state_[1, avoid[i][0], avoid[i][1]] = 1
    state_[2, grid_size_x-1, grid_size_y-1] = 1
    state_=state_.reshape(1,75) + np.random.rand(1,75)/10.0
    state1 = torch.from_numpy(state_).float()
    return state1

def gridMove(current_state, state1, action):
    z = current_state[0]
    x = current_state[1]
    y = current_state[2]
    state_ = state1.clone().reshape(3, grid_size_x, grid_size_y)  # clone()을 사용하여 새로운 텐서를 생성합니다.
    state_[z,x,y] -= 1
    if action == "u":
        x = max(current_state[1] - 1, 0)
    elif action == "d":
        x = min(current_state[1]+1, grid_size_x - 1)
    elif action == "l":
        y = max(current_state[2] - 1, 0)
    elif action == "r":
        y = min(current_state[2] + 1, grid_size_y - 1)
    state_[z,x,y] += 1
    current_state=(z,x,y)
    return state_.reshape(1, 75), current_state  # state_를 다시 reshape한 뒤 반환합니다.

def reward_state(current_state):
    if current_state==(0, goal[0], goal[1]):
        return 100
    elif (current_state[1], current_state[2]) in avoid:
        return -1000
    else:
        return -1

current_state=(0,start[0],start[1])

for i in range(epochs):
    game = gridMake(grid_size_x, grid_size_y,start,goal,avoid)
    state1 = stateMake(grid_size_x, grid_size_y)
    status=1

    while(status==1):
        qval = model(state1)
        qval_ = qval.detach().numpy()
        if (random.random() < epsilon):
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)
        action = action_set[action_]
        state1 , current_state = gridMove(current_state, state1, action)
        reward = reward_state(current_state)
        
        with torch.no_grad():
            newQ = model(state1)
        maxQ = torch.max(newQ)
        if reward == -1:
            Y = reward + (gamma * maxQ)
        else:
            Y = reward

        Y = torch.Tensor([Y]).detach()
        X = qval.squeeze()[action_]
        loss = loss_fn(X, Y)
        print(i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if reward != -1:
            status = 0
        state1 = state1.reshape(1, 75)

    if epsilon > 0.1:
        epsilon -= (1/epochs)
    print(i)

# state = state1.clone().reshape(3, grid_size_x, grid_size_y)
def find_optimal_path(model, start, goal, grid_size_x, grid_size_y, avoid):
    visited = set()  # 방문한 위치를 저장할 집합
    path = []  # 경로를 저장할 리스트
    current_state = (0, start[0], start[1])
    
    while current_state[1:3] != goal:
        state1 = stateMake(grid_size_x, grid_size_y)  # 현재 상태에 대한 입력 벡터 생성
        qval = model(state1)  # 현재 상태에서의 행동 가치 계산
        action_ = torch.argmax(qval).item()  # 최대 가치를 제공하는 행동 선택
        
        # 선택된 행동이 한 번도 방문하지 않은 위치로 이동하는지 확인
        next_state1, next_current_state = gridMove(current_state, state1, action_set[action_])
        if next_current_state[1:3] in visited or (next_current_state[1], next_current_state[2]) in avoid:
            # 이미 방문했거나 피해야 할 위치로 이동하려 할 경우, 다른 행동을 선택
            action_values = qval.detach().numpy().squeeze()
            sorted_actions = np.argsort(action_values)[::-1]  # 가치에 따라 행동을 정렬
            
            for action_idx in sorted_actions:
                next_state1, next_current_state = gridMove(current_state, state1, action_set[action_idx])
                if next_current_state[1:3] not in visited and (next_current_state[1], next_current_state[2]) not in avoid:
                    action_ = action_idx
                    break
        
        # 선택된 행동으로 위치 업데이트
        state1, current_state = gridMove(current_state, state1, action_set[action_])
        visited.add(current_state[1:3])  # 방문한 위치 추가
        # 경로에 행동과 현재 좌표 추가
        path.append((action_set[action_], current_state[1:3]))  # 경로에 행동 추가
        
        if len(visited) == (grid_size_x * grid_size_y - len(avoid)):  # 모든 가능한 위치를 방문했다면 루프 탈출
            break
    
    return path

# 최적의 경로 탐색
optimal_path = find_optimal_path(model, start, goal, grid_size_x, grid_size_y, avoid)
print("Optimal Path:", optimal_path)
