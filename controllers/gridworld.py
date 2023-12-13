import os
import numpy as np
import torch
from flask import request

from environments import GridWorld
from deep_learning.gridworld_with_q_learning import train
from deep_learning.core.train_config import TrainConfig
from deep_learning.core.simulator import Simulator
from deep_learning.q_learning import QLearning
from utils.tools import save_model_and_train_result, get_model_path

size = 5
gridworld = GridWorld(size=size)
l1 = size * size * 4
l2 = 250
l3 = 150
l4 = 4


def GridWorldController(app, socketio):
    @app.post('/gridworld/step')
    def gridworld_step():
        data = request.get_json()
        state = data['state']
        action = data['action']
        print('state: %s, action: %s' % (state, action))
        next_state, done = gridworld.step(state, action)
        return {
            'next_state': next_state,
            'done': done,
        }

    @app.get('/gridworld/map')
    def gridworld_map():
        grids = {}
        for i, (id, grid) in enumerate(gridworld.grids.items()):
            grids[id] = grid.__dict__
        return {'map': grids}

    @app.post('/gridworld/train')
    def gridworld_train():
        data = request.get_json()
        model_name = data['modelName']
        mode = data['mode']

        model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4),
        )

        epochs = 1000

        def env_producer():
            return GridWorld(size=5, mode=mode)

        losses = train(env_producer, model, epochs, socketio)

        if model_name:
            result = {'epochs': epochs, 'losses': losses}
            save_model_and_train_result(model, result, model_name)

        return {'message': 'ok'}

    @app.post('/gridworld/train_with_replay')
    def gridworld_train_with_replay():
        data = request.get_json()
        model_name = data['modelName']
        mode = data['mode']

        model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4),
        )

        def env_producer():
            return GridWorld(size=5, mode=mode)

        epochs = 5000
        train_config = TrainConfig(
            epochs=epochs,
            learning_rate=1e-3,
            batch_size=200,
            gamma=0.9,
            epsilon=0.3
        )
        q_learning = QLearning(train_config)
        losses = q_learning.train(env_producer, model, socketio)

        if model_name:
            result = {'epochs': epochs, 'losses': losses}
            save_model_and_train_result(model, result, model_name)

        return {'message': 'ok'}

    @app.post('/gridworld/model_policy')
    def gridworld_model_policy():
        data = request.get_json()
        model_name = data['modelName']

        model_path = get_model_path(model_name)
        model = torch.load(model_path)

        game = GridWorld(size=5, mode='random')
        grids = {}
        for i, (id, grid) in enumerate(game.grids.items()):
            grids[id] = grid.__dict__

            game.current_state = i
            state_ = game.encode_state(noise=False)
            state = torch.from_numpy(state_).float()
            q_values = model(state)
            qs = q_values.squeeze().detach().numpy()
            # calculate softmax of action values
            action_dist = np.exp(qs) / sum(np.exp(qs))
            grids[id]['q_values'] = action_dist.tolist()

        return {'map': grids}
        
    @app.post('/gridworld/model_test')
    def gridworld_model_test():
        data = request.get_json()
        model_name = data['modelName']
        mode = data['mode']

        model_path = get_model_path(model_name)
        model = torch.load(model_path)

        def env_producer():
            return GridWorld(size=5, mode=mode)

        simulator = Simulator(episodes=1000, env_producer=env_producer)
        simulator.simulate(model, socketio)

        return {'message': 'ok'}