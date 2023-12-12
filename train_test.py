import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from deep_learning.gridworld_with_q_learning import train

from environments import GridWorld
from utils.tools import moving_average, get_model_path

def env_producer():
    return GridWorld(size=5, mode='random')

def runover(env_producer, model):
    game = env_producer()
    done = False
    while not done:
        state_ = game.encode_state()
        state = torch.from_numpy(state_).float()
        qval = model(state)
        qval_ = qval.data.numpy()
        action = np.argmax(qval_)
        _, done, truncated = game.take_action(action)

        if done:
            if game.reward() > 0:
                return True
            else:
                return False
        if truncated:
            return False

def test_model(model, num_games):
    win_count = 0
    for i in range(num_games):
        win = runover(env_producer, model)
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

    epochs = 1000

    # losses = train(env_producer, model, epochs)
    model_path = get_model_path('gridworld-q-learning-simple-network')
    model = torch.load(model_path)
    test_model(model, 500)

    # plt.xlabel('epochs')
    # plt.ylabel('losses')
    # mavg = moving_average(losses, 50)
    # plt.plot(mavg)
    # plt.show()

