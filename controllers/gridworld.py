from flask import request
from environments import GridWorld
from deep_learning.gridworld_with_q_learning import train

import numpy as np
import torch

gridworld = GridWorld()

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
        return {
            'map': grids
        }

    @app.get('/gridworld/train')
    def gridworld_train():
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

        epochs = 100

        losses = train(model, epochs, socketio)
        return {
            'losses': losses
        }
