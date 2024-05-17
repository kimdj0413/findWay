import numpy as np

# 그리드 크기
grid_size = 3

# Q-테이블 초기화
Q = np.zeros((grid_size, grid_size, 4))  # 4개의 행동(위, 아래, 왼쪽, 오른쪽)에 대한 각 상태
print(Q)
# 파라미터 설정
alpha = 0.1  # 학습률
gamma = 0.9  # 할인율
epsilon = 0.1  # 탐험 확률
episodes = 1000  # 학습 에피소드 수

# 보상 테이블 초기화
R = np.full((grid_size, grid_size), -1)  # 모든 상태에 대해 기본 보상 -1
R[2, 2] = 100  # 목표 상태에 대한 보상
R[0, 1] = -100  # 피해야 할 상태에 대한 패널티
print(R)
# 가능한 행동 정의
actions = ["up", "down", "left", "right"]

# 상태 전이 함수 정의
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
    state = (0, 0)  # 시작 상태
    while state != (2, 2):
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)  # 탐험
        else:
            action = actions[np.argmax(Q[state[0], state[1]])]  # 이용
        
        next_state_ = next_state(state, action)
        reward = R[next_state_]
        
        # Q-테이블 업데이트
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
print(Q)
print("최적 경로: ", optimal_path)