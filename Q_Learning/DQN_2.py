import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import random
from collections import deque

# tf.config.set_visible_devices([], 'GPU')
print("Visible devices:", tf.config.get_visible_devices())
# 설정
grid_size = 5
alpha = 0.1
gamma = 0.9
epsilon = 0.8
episodes = 1000
batch_size = 128
memory_capacity = 10

# 보상
R = np.full((grid_size, grid_size), -10)
R[4, 4] = 1000
R[0, 1] = -10000
R[2, 3] = -10000

actions = ["up", "down", "left", "right"]

# 상태 전이 함수
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

# 신경망 모델 정의
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(32, input_shape=(grid_size * grid_size,), activation='relu'))
    # model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(4, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=alpha), loss='mse')
    return model

# 경험 재생 버퍼
memory = deque(maxlen=memory_capacity)

# 모델 생성
model = build_model()
target_model = build_model()
target_model.set_weights(model.get_weights())

# 학습 과정
for episode in range(episodes):
    state = (0, 0)
    total_reward = 0
    steps = 0
    episode_done = False  # 에피소드 종료 플래그
    
    while not episode_done and state != (4, 4):
        state_flat = np.eye(grid_size * grid_size)[state[0] * grid_size + state[1]].reshape(1, -1)
        
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            q_values = model.predict(state_flat)
            action = actions[np.argmax(q_values)]
        
        next_state_ = next_state(state, action)
        reward = R[next_state_]
        total_reward += reward
        
        # reward 값이 너무 작아지면 에피소드 종료
        if total_reward <= -1000:
            episode_done = True
            continue
        
        next_state_flat = np.eye(grid_size * grid_size)[next_state_[0] * grid_size + next_state_[1]].reshape(1, -1)
        memory.append((state_flat, action, reward, next_state_flat, state == (4, 4)))
        
        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)
            for state_batch, action_batch, reward_batch, next_state_batch, done_batch in minibatch:
                target = reward_batch
                if not done_batch:
                    target = reward_batch + gamma * np.amax(target_model.predict(next_state_batch, verbose=0))
                
                target_f = model.predict(state_batch, verbose=0)
                target_f[0][actions.index(action_batch)] = target
                model.fit(state_batch, target_f, epochs=1, verbose=0)
        
        state = next_state_
        steps += 1
        
        if steps % 10 == 0:
            target_model.set_weights(model.get_weights())
    
    # 에피소드 종료 시 로그 출력
    if episode_done:
        print(f"Episode: {episode+1} was terminated early at step {steps} with total reward {total_reward}")
    else:
        print(f"Episode: {episode+1}, Total Reward: {total_reward}, Steps: {steps}")


# 최적 경로 도출
state = (0, 0)
optimal_path = [state]
while state != (4, 4):
    state_flat = np.eye(grid_size * grid_size)[state[0] * grid_size + state[1]].reshape(1, -1)
    action = actions[np.argmax(model.predict(state_flat))]
    state = next_state(state, action)
    optimal_path.append(state)
print("최적 경로: ", optimal_path)