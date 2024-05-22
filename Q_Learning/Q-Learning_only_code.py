import numpy as np
import random
import matplotlib.pyplot as plt
import time
import datetime

start = time.time()
# 그리드 크기
grid_size_x = 100
grid_size_y = 100

# Q-테이블 초기화
Q = np.zeros((grid_size_x, grid_size_y, 4))
alpha = 0.5
gamma = 0.9  
epsilon = 0.7
episodes = 13000

start_point = (0, 0)
goal_point = (grid_size_x-1,grid_size_y-1)
# must_avoid = [(9, 8), (3, 6), (8, 2), (15, 9), (13, 18), (18, 13), (2, 8), (14, 8), (4, 18), (12, 10), (4, 10), (4, 11), (18, 3), (3, 2), (5, 11), (19, 3), (15, 4), (9, 12), (13, 16), (14, 3), (14, 7), (1, 1), (4, 2), (14, 17), (19, 8), (1, 11), (13, 19), (2, 17), (11, 15), (0, 11), (19, 4), (10, 10), (13, 12), (19, 18), (10, 1), (18, 15), (13, 3), (4, 6), (6, 6), (5, 12), (5, 16), (19, 10), (9, 10), (15, 17), (5, 15), (2, 3), (5, 18), (3, 7), (1, 13), (4, 15)]
must_avoid=[]
random_cnt=3000
while len(must_avoid) < random_cnt:
    x = random.randint(0, grid_size_x-1)
    y = random.randint(0, grid_size_y-1)
    
    if ((x, y) not in must_avoid) or ((x,y) != (0,0)) or ((x,y) != (grid_size_x,grid_size_y)):
        must_avoid.append((x, y))
# 보상 테이블 초기화
R = np.full((grid_size_x, grid_size_y), -5)
R[goal_point] = 1000000
for i in must_avoid:
    R[i] = -1000000

# 가능한 행동 정의
actions = ["up", "down", "left", "right"]

# 상태 전환 함수 정의
def next_state(state, action):
    i, j = state
    if action == "up":
        i = max(i - 1, 0)
    elif action == "down":
        i = min(i + 1, grid_size_x - 1)
    elif action == "left":
        j = max(j - 1, 0)
    elif action == "right":
        j = min(j + 1, grid_size_y - 1)
    return i, j

# Q-Learning 알고리즘
for episode in range(episodes):
    state = start_point
    while state != (goal_point):
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]
        next_state_ = next_state(state, action)
        reward = R[next_state_]
        
        # Q-테이블 업데이트(오차를 계산함으로써 정확한 계산을 하기 위함)
        Q[state[0], state[1], actions.index(action)] += alpha * (
            reward + gamma * np.max(Q[next_state_[0], next_state_[1]]) - Q[state[0], state[1], actions.index(action)])

        state = next_state_
    print(f"Episode: {episode}, Reward: {reward}")

sec = time.time()-start
times = str(datetime.timedelta(seconds=sec))
short = times.split(".")[0]
print(f"{short} sec")

# 최적 경로 찾기
state = (0, 0)
visited_states = set()
visited_states.add(state)
optimal_path = [state]
while state != goal_point:
    sorted_actions = np.argsort(Q[state[0], state[1]])[::-1]

    for action_index in sorted_actions:
        action = actions[action_index]
        next_s = next_state(state, action)
        if next_s not in visited_states:
            state = next_s
            optimal_path.append(state)
            visited_states.add(state)
            break
    else:
        break
    print(f"State: {state}, Action: {action}")
print(f"최적 경로 : {optimal_path}")

# 그리드 생성
fig, ax = plt.subplots(figsize=(20, 20))

# 그리드 라인 그리기
for x in range(grid_size_x + 1):
    ax.axhline(x, lw=2, color='k', zorder=5)
for y in range(grid_size_y + 1):
    ax.axvline(y, lw=2, color='k', zorder=5)

# 그리드 셀 사이즈 조정
ax.set_xticks(np.arange(grid_size_x + 1))
ax.set_yticks(np.arange(grid_size_y + 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

# 꼭 지나야 할 점과 가지 말아야 할 점 표시
for j in range(len(must_avoid)):
    ax.add_patch(plt.Rectangle((must_avoid[j][1], must_avoid[j][0]), 1, 1, color='red', alpha=0.5))

# 각 셀에 좌표 표시
for x in range(grid_size_x):
    for y in range(grid_size_y):
        ax.text(y + 0.5, x + 0.5, f'({x},{y})', ha='center', va='center', fontsize=0.5)

# 시작점과 도착점 표시
ax.text(start_point[1] + 0.1, start_point[0] + 1 - 0.1, 'Start', ha='left', va='bottom', fontweight='bold', fontsize=12, color='blue', zorder=10)
ax.text(goal_point[1] + 0.1, goal_point[0] + 1 - 0.1, 'Goal', ha='left', va='bottom', fontweight='bold', fontsize=12, color='red', zorder=10)

# 최적 경로 그리기
for i in range(len(optimal_path) - 1):
    start = optimal_path[i]
    end = optimal_path[i + 1]
    ax.plot([start[1] + 0.5, end[1] + 0.5], [start[0] + 0.5, end[0] + 0.5], 'o-', color='green', lw=1, markersize=3, zorder=10)

plt.gca().invert_yaxis()
plt.axis('equal')
plt.savefig("grid_plot.svg", format="svg")
plt.show()