import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pylab as plt

# print(game.display())
# print(game.board.render_np())
# print(game.board.render_np().shape)

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

l1= 64 # 4X4X4
l2 = 150
l3 = 100
l4 = 4 #action

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
 
gamma = 0.9
epsilon = 1.0
epochs = 1000
losses = []

for i in range(epochs):
    # 노이즈 추가 및 1차원 배열로 데이터 변환.
    game = Gridworld(size=4, mode='static')
    state_ = game.board.render_np().reshape(1,l1) + np.random.rand(1,l1)/10.0
    state1 = torch.from_numpy(state_).float()
    status = 1

    while(status == 1):
        qval = model(state1)
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)

        action = action_set[action_]
        game.makeMove(action)
        state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()

        with torch.no_grad(): # 역전파 방지
            newQ = model(state2.reshape(1,64))
        maxQ = torch.max(newQ)
        if reward == -1: # 도착점, 못가는 곳이 아닌 곳을 갈 시
            Y = reward + (gamma * maxQ)
        else:
            Y = reward

        Y = torch.Tensor([Y]).detach()
        print(f"Y : {Y}")
        X = qval.squeeze()[action_] #O
        print(f"X : {X}")
        loss = loss_fn(X, Y) # X : target, Y : Predict
        print(i, loss.item())
       #clear_output(wait=True)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        state1 = state2
        if reward != -1: #Q
            status = 0

    if epsilon > 0.1: #R
        epsilon -= (1/epochs)

 

def test_model(model, mode='static', display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state = torch.from_numpy(state_).float()

    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1

    while(status == 1): #A
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_) #B
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))

        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()

        if display:
            print(test_game.display())

        reward = test_game.reward()

        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1

        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win

test_model(model)