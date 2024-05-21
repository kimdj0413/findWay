import numpy as np
import torch
from Gridworld import Gridworld
import random
from matplotlib import pylab as plt

game = Gridworld(size=4, mode='static')
print(game.display())
print(game.board.render_np())
# print(game.board.render_np().shape)
print(game.board.render_np().reshape(1,64))
# print(np.random.rand(1,64))
state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0 #D
print(state_)
# state1 = torch.from_numpy(state_).float()
# print(state1)
# print(game.reward().shape)