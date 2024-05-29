from collections import deque
import numpy as np
import torch
import random
from matplotlib import pylab as plt
import time
import datetime

start_time = time.time()
action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

grid_size_x = 10
grid_size_y = 10
gamma = 0.9
# epsilon = 1.0
epochs = 10000
batch_size = 32
random_cnt = 3
losses = []
l1= grid_size_x*grid_size_y*3
l2 = 64
l3 = 16
l4 = 4
# l5 = 4
# l6 = 4
# l7 = 4
start=(0,0)
goal=(grid_size_x-1,grid_size_y-1)
avoid = [(0,5),(0,6),(0,7),(0,8),(0,9),(1,0),(1,1),(1,2),(1,5),(1,6),(1,7),(1,8),(1,9),(2,0),(2,1),(2,2),(2,5),(2,6),(2,7),(2,8),(2,9),(3,0),(3,1),(3,2),(3,5),(3,6),(3,7),(3,8),(3,9),(4,9),(5,4),(5,5),(5,6),(5,7),(5,9),(6,0),(6,1),(6,2),(6,4),(6,5),(6,6),(6,7),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2),(8,5),(8,6),(8,7),(9,5),(9,6),(9,7)]

# while len(avoid) < random_cnt:
#     x = random.randint(0, grid_size_x-1)
#     y = random.randint(0, grid_size_y-1)
    
#     if ((x, y) not in avoid) and ((x,y) != (0,0)) and ((x,y) != (grid_size_x-1,grid_size_y-1)):
#         avoid.append((x, y))

# print(avoid)
# input("Enter")

model = torch.nn.Sequential(
        torch.nn.Linear(l1,l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2,l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3,l4)
        # torch.nn.ReLU(),
        # torch.nn.Linear(l4,l5)
        # torch.nn.ReLU(),
        # torch.nn.Linear(l5,l6)
        # torch.nn.ReLU(),
        # torch.nn.Linear(l6,l7)
)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer  = torch.optim.Adam(model.parameters(),lr = learning_rate)

# stateMake
def stateMake(grid_size_x, grid_size_y):
    state_ = np.zeros((3,grid_size_x, grid_size_y))
    state_[0, start[0],start[1]] = 1
    for i in range(0,len(avoid)):
        state_[1, avoid[i][0], avoid[i][1]] = 1
    state_[2, grid_size_x-1, grid_size_y-1] = 1
    state_=state_.reshape(1,l1) + np.random.rand(1,l1)/10.0
    state1 = torch.from_numpy(state_).float()
    return state1

def gridMove(current_state, state1, action):
    z = current_state[0]
    x = current_state[1]
    y = current_state[2]
    state_ = state1.clone().reshape(3, grid_size_x, grid_size_y)
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
    next_state=(z,x,y)
    return state_.reshape(1, l1), next_state

def reward_state(current_state):
    if current_state==(0, goal[0], goal[1]):
        return 200
    elif (current_state[1], current_state[2]) in avoid:
        return -100
    else:
        return -0.1

current_state=(0,start[0],start[1])
num_cnt=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# loss_cnt = deque(maxlen=100)

for i in range(epochs):
    epsilon = max(0.1, 1.0-i/epochs)
    state1 = stateMake(grid_size_x, grid_size_y)
    state1 = torch.tensor(state1, dtype=torch.float32).to(device)
    status=1
    # if len(loss_cnt)>90:
    #     loss_avg = sum(loss_cnt)/len(loss_cnt)
    #     print(f"loss_avg{loss_avg}")
    #     if loss_avg<10:
    #         break

    while(status==1):
        if (random.random() < epsilon):
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)
        action = action_set[action_]
        state1 , next_state = gridMove(current_state, state1, action)
        reward = reward_state(current_state)
        current_state = next_state
        qval = model(state1)
        qval_ = qval.detach().cpu().numpy()
        with torch.no_grad():
            newQ = model(state1)
        maxQ = torch.max(newQ)
        if reward == -10:
            Y = reward + (gamma * maxQ)
        else:
            Y = reward

        Y = torch.tensor([Y], device=device, dtype=torch.float32).detach()
        X = qval.squeeze()[action_]
        loss = loss_fn(X, Y)/1000
        print(f'episodes :{i} , loss :{loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # loss_cnt.append(loss.item())
        num_cnt.append(i)
        
        if reward != -10:
            status = 0
        state1 = state1.reshape(1, l1)

    if epsilon > 0.1:
        epsilon -= (1/epochs)
    # print(i)

def find_optimal_path(model, start, goal, grid_size_x, grid_size_y, avoid):
    visited = set()
    path = [(0,0)]
    action_path = []
    current_state = (0, start[0], start[1])
    
    while current_state[1:3] != goal:
        state1 = stateMake(grid_size_x, grid_size_y)
        state1 = torch.FloatTensor(state1).to(device)
        qval = model(state1)
        action_ = torch.argmax(qval).item()
        
        next_state1, next_current_state = gridMove(current_state, state1, action_set[action_])
        if next_current_state[1:3] in visited or (next_current_state[1], next_current_state[2]) in avoid:
            action_values = qval.detach().cpu().numpy().squeeze()
            sorted_actions = np.argsort(action_values)[::-1]
            
            for action_idx in sorted_actions:
                next_state1, next_current_state = gridMove(current_state, state1, action_set[action_idx])
                if next_current_state[1:3] not in visited and (next_current_state[1], next_current_state[2]) not in avoid:
                    action_ = action_idx
                    break
        
        state1, current_state = gridMove(current_state, state1, action_set[action_])
        visited.add(current_state[1:3])
        path.append(current_state[1:3])
        action_path.append(action_set[action_])
        
        if len(visited) == (grid_size_x * grid_size_y - len(avoid)):
            break
    
    return path, action_path

# 최적의 경로 탐색
sec = time.time()-start_time
times = str(datetime.timedelta(seconds=sec))
short = times.split(".")[0]
print(f"{short} sec")
optimal_path, optiomal_action = find_optimal_path(model, start, goal, grid_size_x, grid_size_y, avoid)
print("Optimal Path:", optimal_path)
print("Optimal action:", optiomal_action)

# 그림 그리기
plt.ion()
# 손실 함수 그리기
fig2, ax2 = plt.subplots()

loss_x = num_cnt
loss_y = losses

ax2.plot(loss_x, loss_y)
ax2.set_title('losses Graph')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Losses')

plt.show()

fig1, ax1 = plt.subplots()

for x in range(grid_size_x + 1):
    ax1.axhline(x, lw=2, color='k', zorder=5)
for y in range(grid_size_y + 1):
    ax1.axvline(y, lw=2, color='k', zorder=5)

ax1.set_xticks(np.arange(grid_size_x + 1))
ax1.set_yticks(np.arange(grid_size_y + 1))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.grid(True)

for j in range(len(avoid)):
    ax1.add_patch(plt.Rectangle((avoid[j][1], avoid[j][0]), 1, 1, color='red', alpha=0.5))

for x in range(grid_size_x):
    for y in range(grid_size_y):
        ax1.text(y + 0.5, x + 0.5, f'({x},{y})', ha='center', va='center', fontsize=10)

ax1.text(start[1] + 0.1, start[0] + 1 - 0.1, 'Start', ha='left', va='bottom', fontweight='bold', fontsize=12, color='blue', zorder=10)
ax1.text(goal[1] + 0.1, goal[0] + 1 - 0.1, 'Goal', ha='left', va='bottom', fontweight='bold', fontsize=12, color='red', zorder=10)

for i in range(len(optimal_path) - 1):
    start = optimal_path[i]
    end = optimal_path[i + 1]
    ax1.plot([start[1] + 0.5, end[1] + 0.5], [start[0] + 0.5, end[0] + 0.5], 'o-', color='green', lw=1, markersize=3, zorder=10)

plt.gca().invert_yaxis()
plt.axis('equal')
plt.show()

input("Enter")