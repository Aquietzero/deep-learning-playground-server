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


def GridWorldController(app, socketio):
    @app.post('/gridworld/map')
    def gridworld_map():
        env_params = request.get_json()
        env = GridWorld(**env_params)

        grids = {}
        for i, (id, grid) in enumerate(env.grids.items()):
            grids[id] = grid.__dict__
        return {'map': grids}

    @app.post('/gridworld/train')
    def gridworld_train():
        data = request.get_json()
        model_name = data['model_name']
        env_params = data['env_params']
        train_params = data['train_params']

        size = env_params['size']
        l1 = size * size * 4 # 4 x 4 x 4 = 64
        l2 = l1 * 4          # 64 x 4 = 256
        l3 = l1 * 3          # 64 x 3 = 192
        l4 = 4               # number of actions

        model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4),
        )

        def env_producer():
            return GridWorld(**env_params)

        train_config = TrainConfig(**train_params)
        q_learning = QLearning(train_config)
        losses = q_learning.train(env_producer, model, socketio)

        if model_name:
            result = {
                'epochs': train_params['epochs'],
                'losses': losses,
            }
            save_model_and_train_result(model, result, model_name)

        return {'message': 'ok'}

    @app.post('/gridworld/model_policy')
    def gridworld_model_policy():
        data = request.get_json()
        model_name = data['model_name']
        env_params = data['env_params']

        model_path = get_model_path(model_name)
        model = torch.load(model_path)

        def env_producer():
            return GridWorld(**env_params)

        simulator = Simulator(
            episodes=1,
            env_producer=env_producer)

        result = simulator.runover(model)
        env = simulator.env
        grids = {}
        for i, (id, grid) in enumerate(env.grids.items()):
            grids[id] = grid.__dict__

            env.current_state = i
            state_ = env.encode_state(noise=False)
            state = torch.from_numpy(state_).float()
            q_values = model(state)
            qs = q_values.squeeze().detach().numpy()
            # calculate softmax of action values
            action_dist = np.exp(qs) / sum(np.exp(qs))
            grids[id]['q_values'] = action_dist.tolist()

        return {'map': grids, 'result': result}
        
    @app.post('/gridworld/model_test')
    def gridworld_model_test():
        data = request.get_json()
        model_name = data['model_name']
        env_params = data['env_params']
        test_params = data['test_params']

        model_path = get_model_path(model_name)
        model = torch.load(model_path)

        def env_producer():
            return GridWorld(**env_params)

        simulator = Simulator(
            episodes=test_params['episodes'],
            env_producer=env_producer)
        simulator.simulate(model, socketio)

        return {'message': 'ok'}