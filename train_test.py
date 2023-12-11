import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from deep_learning.gridworld_with_q_learning import train

from environments import GridWorld
from utils.tools import moving_average

# def test():
#     game = GridWorld()
#     done = False
#     action_map = ['up', 'right', 'down', 'left']
#     while not done:
#         action = np.random.randint(0, 4)
#         state, done = game.take_action(action)
#         print('state %d, action %s, reward %d' % (state, action_map[action], game.reward()))

def runover(model):
    game = GridWorld()
    done = False
    while not done:
        state_ = game.encode_state().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        _, done = game.take_action(action)

        if done:
            if game.reward() > 0:
                return True
            else:
                return False

def test_model(model, num_games):
    win_count = 0
    for i in range(num_games):
        win = runover(model)
        if win:
            win_count += 1

    print('result: %d / %d' % (win_count, num_games))


if __name__ == '__main__':
    # game = GridWorld()
    # print(game.encode_state())
    # game.take_action(1)
    # print(game.encode_state())
    # test()

    l1 = 64
    l2 = 150
    l3 = 100
    l4 = 4

    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
    )

    # epochs = 1000

    # losses = train(model, epochs)
    # test_model(model, 500)

    # plt.xlabel('epochs')
    # plt.ylabel('losses')
    # mavg = moving_average(losses, 50)
    # plt.plot(mavg)
    # plt.show()

