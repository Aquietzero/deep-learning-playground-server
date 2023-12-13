import torch
import numpy as np
import torch
import random
import copy
from utils.tools import clear_console
from deep_learning.core.trainer import Trainer
from deep_learning.core.experience_replay import ExperienceReplay

class QLearning(Trainer):
    def __init__(self, train_config):
        super().__init__(train_config)

    def train(self, env_producer, model, socketio):
        losses = []

        loss_fn = torch.nn.MSELoss()
        learning_rate = self.train_config.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        epochs = self.train_config.epochs
        gamma = self.train_config.gamma
        epsilon = self.train_config.epsilon

        # target network
        model2 = copy.deepcopy(model)
        model2.load_state_dict(model.state_dict())

        sync_freq = 50
        sync_count = 0
        batch_size = self.train_config.batch_size
        replay = ExperienceReplay(N=1000, batch_size=batch_size)

        for i in range(epochs):
            env = env_producer()
            state_ = env.encode_state()
            state1 = torch.from_numpy(state_).float()
            is_over = False

            epoch_losses = []
            while (not is_over):
                sync_count += 1

                # runs the Q-network to calculate the Q values for all actions
                qval = model(state1)
                qval_ = qval.data.numpy()
                # use epsilon-greedy to select an action
                if random.random() < epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(qval_)

                # take the action
                _, done, truncated = env.take_action(action)
                # after making the move, finds the maximum Q value from the
                # new state
                state2_ = env.encode_state()
                state2 = torch.from_numpy(state2_).float()
                reward = env.reward()

                replay.add_memory(state1, action, reward, state2, done)
                state1 = state2

                if replay.size() > batch_size:
                    state1_batch, action_batch, reward_batch, state2_batch, done_batch = replay.get_batch()

                    Q1 = model(state1_batch)
                    with torch.no_grad():
                        Q2 = model2(state2_batch)

                    Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                    X = Q1.gather(dim=1, index=action_batch.unsqueeze(dim=1)).squeeze()

                    loss = loss_fn(X, Y.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    epoch_losses.append(loss.item())
                    optimizer.step()

                    # sync target network with policy network
                    if sync_count % sync_freq == 0:
                        model2.load_state_dict(model.state_dict())

                if reward != -1 or truncated:
                    is_over = True

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
