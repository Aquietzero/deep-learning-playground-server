import os
import numpy as np
import torch
from flask import request

from environments import GridWorld
from deep_learning.gridworld_with_q_learning import train
from utils.tools import save_model_and_train_result, get_model_path

gridworld = GridWorld(size=4)

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

        losses = train(model, epochs, socketio)

        if model_name:
            result = {'epochs': epochs, 'losses': losses}
            save_model_and_train_result(model, result, model_name)

        return {'message': 'ok'}

    @app.post('/gridworld/test_model')
    def gridworld_test():
        data = request.get_json()
        model_name = data['modelName']

        model_path = get_model_path(model_name)

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
        model.load_state_dict(torch.load(model_path))

        game = GridWorld(size=4)
        grids = {}
        for i, (id, grid) in enumerate(gridworld.grids.items()):
            grids[id] = grid.__dict__

            game.current_state = i
            state_ = game.encode_state().reshape(1, 64)
            state = torch.from_numpy(state_).float()
            q_values = model(state)
            qs = q_values.squeeze().detach().numpy()
            # calculate softmax of action values
            action_dist = np.exp(qs) / sum(np.exp(qs))
            grids[id]['q_values'] = action_dist.tolist()

        return {'map': grids}
        
