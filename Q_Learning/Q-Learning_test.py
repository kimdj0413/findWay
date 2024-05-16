import numpy as np
import matplotlib.pyplot as plt

# 그리드 크기
grid_size = 3

# Q-테이블 초기화
Q = np.zeros((grid_size, grid_size, 4))  # 각 상태에서 4가지 행동 (상, 하, 좌, 우)
# 파라미터 설정
# 학습률 : 새로운 정보를 얼마나 빠르게 받아들일지 결정.
# 0 ~ 1 : 학습 x ~ 완전히 새로운 정보
alpha = 0.1
# 할인율 : 미래 보상에 대한 현재 가치 결정
# 0 ~ 1 : 즉각적인 보상만 고려 ~ 미래의 보상을 중요하게 여김 
gamma = 0.9  
# 탐험 확률 : 완전히 새로운 행동을 할 확률.
epsilon = 0.1  
episodes = 1000  # 학습 에피소드 수

start_point = (0, 0)
goal_point = (2, 2)

# 보상 테이블 초기화
R = np.full((grid_size, grid_size), -1)
# np.full = 지정된 크기의 배열을 생성하고 모든 요소를 특정 값으로 채우는 함수. -1.
# -1로 채우는 이유는 목표에 도달하기 위해 불필요한 행동을 줄이기 위해.
R[2, 2] = 100  # 도착점 보상
R[0, 1] = -100  # 가지 말아야 할 점 페널티
R[1, 2] = 10  # 반드시 지나야 할 점 보상

# 가능한 행동 정의
actions = ["up", "down", "left", "right"]

# 상태 전환 함수 정의
# state i, j = 현재 위치 좌표 i, j 값
# max, min으로 범위를 지정해줌으로써 최댓값, 최솟값에 유의
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

# Q-Learning 알고리즘
for episode in range(episodes):
    state = (0, 0)  # 시작점
    while state != (2, 2):
        # 탐욕적 정책을 활용한 탐험과 활용 사이의 균형 조절.
        if np.random.rand() < epsilon:
            # 0 ~ 1 사이로 생성된 난수가 입실론 보다 작으면 탐험을 선택.
            action = np.random.choice(actions)  # 탐험
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]  # 활용
            # Q[state[0],state[1]] = 현재 state=(state[0], state[1]) 값에 대한 모든 행동들의 Q 값
            # np.argmax = 그 중 최고값을 구하는 함수.
            # ex) Q[state[0],state[1]]의 모든 결과는 아래와 같다.
            # Q = [
            # [[-1, 0, -1, -1], [-1, -1, -1, 1], [-1, -1, 0, -1]],
            # [[0, -1, 1, -1], [-1, 1, -1, -1], [1, -1, -1, -1]],
            # [[-1, 1, -1, 0], [-1, -1, 0, -1], [0, -1, -1, 1]]]
            # 만약 (0,1)이면 [-1,-1,-1,1] 이므로 최대 값의 액션은 right
        
        next_state_ = next_state(state, action) # 결과값 i, j
        reward = R[next_state_] # 보상 저장
        
        # Q-테이블 업데이트(오차를 계산함으로써 정확한 계산을 하기 위함)
        Q[state[0], state[1], actions.index(action)] += alpha * (
            reward + gamma * np.max(Q[next_state_[0], next_state_[1]]) - Q[state[0], state[1], actions.index(action)])

        state = next_state_

# 최적 경로 찾기
state = (0, 0)
optimal_path = [state]
while state != (2, 2):
    action = actions[np.argmax(Q[state[0], state[1]])]
    state = next_state(state, action)
    optimal_path.append(state)

print("최적 경로: ", optimal_path)

# 그리드 생성
fig, ax = plt.subplots()
# 그리드 라인 그리기
for x in range(grid_size + 1):
    ax.axhline(x, lw=2, color='k', zorder=5)
    ax.axvline(x, lw=2, color='k', zorder=5)
# 그리드 셀 사이즈 조정
ax.set_xticks(np.arange(grid_size + 1))
ax.set_yticks(np.arange(grid_size + 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

# 각 셀에 좌표 표시
for x in range(grid_size):
    for y in range(grid_size):
        ax.text(x + 0.5, y + 0.5, f'({x},{y})', ha='center', va='center')

# 시작점과 도착점 표시
ax.text(start_point[1] + 0.5, grid_size - start_point[0] - 0.5, 'S', ha='left', va='bottom', fontweight='bold', fontsize=12, color='blue', zorder=10)
ax.text(goal_point[1] + 0.5, grid_size - goal_point[0] - 0.5, 'G', ha='left', va='bottom', fontweight='bold', fontsize=12, color='red', zorder=10)

# 최적 경로 그리기
for i in range(len(optimal_path)-1):
    start = optimal_path[i]
    end = optimal_path[i+1]
    ax.plot([start[1]+0.5, end[1]+0.5], [grid_size-start[0]-0.5, grid_size-end[0]-0.5], 'o-', color='green', lw=1, markersize=3, zorder=10)

plt.gca().invert_yaxis() # y축 방향 반전
plt.axis('equal')  # x, y 축의 비율을 같게 설정
plt.show()