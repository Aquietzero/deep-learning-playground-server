import numpy as np
import torch
import random
import copy
from utils.tools import clear_console


def train(env_producer, model, epochs, socketio):
    losses = []

    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    gamma = 0.9 # discount factor
    epsilon = 1.0 # initialized as 1 and then decrease

    for i in range(epochs):
        game = env_producer()
        state_ = game.encode_state()
        state1 = torch.from_numpy(state_).float()
        is_over = False

        epoch_losses = []
        while (not is_over):
            # runs the Q-network to calculate the Q values for all actions
            qval = model(state1)
            qval_ = qval.data.numpy()
            # use epsilon-greedy to select an action
            if random.random() < epsilon:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(qval_)

            # take the action
            _, _, truncated = game.take_action(action)
            # after making the move, finds the maximum Q value from the
            # new state
            state2_ = game.encode_state()
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            with torch.no_grad():
                newQ = model(state2)
            maxQ = torch.max(newQ)

            if reward == -1:
                Y = reward + (gamma * maxQ)
            else:
                Y = reward

            Y = torch.Tensor([Y]).detach()
            X = qval.squeeze()[action]
            loss = loss_fn(X, Y)
            optimizer.zero_grad()
            loss.backward()

            losses.append(loss.item())
            epoch_losses.append(loss.item())

            optimizer.step()
            state1 = state2
            if reward != -1 or truncated:
                is_over = True

        if epsilon > 0.1:
            epsilon -= (1/epochs)

        socketio.emit('progress', {
            'epochs': epochs,
            'current': i + 1,
        })
        socketio.emit('training_info', {
            'loss': np.average(epoch_losses)
        })

        clear_console()
        print('epochs: %d / %d' % (i + 1, epochs))

    return losses
