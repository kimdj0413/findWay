##  https://limitsinx.tistory.com/153
import torch
import numpy as np

# 임의의 값
gamma = 0.75
alpha = 0.9

# A ~ I 까지 라벨링
location_to_state = {'A': 0,'B': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8}
actions = [0,1,2,3,4,5,6,7,8]

# Reward 표
R = np.array([[0,1,0,1,0,0,0,0,0],
              [1,0,1,0,1,0,0,0,0],
              [0,1,0,0,0,1,0,0,0],
              [1,0,0,0,1,0,1,0,0],
              [0,1,0,1,0,1,0,1,0],
              [0,0,1,0,1,0,0,0,1],
              [0,0,0,1,0,0,0,1,0],
              [0,0,0,0,1,0,1,0,1],
              [0,0,0,0,0,1,0,1,0]])

# 라벨링을 A ~ I 까지
state_to_location = {state: location for location, state in location_to_state.items()}

# 최종 목적지의 보상을 2000으로 바꿔줌
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 2000

    Q = np.array(np.zeros([9,9]))
    
    for i in range(1000):
        current_state = np.random.randint(0,9)
        playable_actions = []
        for j in range(9):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)

        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD

        route = [starting_location]
        next_location = starting_location
        while (starting_location != ending_location):
            starting_state = location_to_state[starting_location]
            next_state = np.argmax(Q[starting_state,])
            next_location = state_to_location[next_state]
            route.append(next_location)
            starting_location = next_location
        return route

print(route('A','I'))