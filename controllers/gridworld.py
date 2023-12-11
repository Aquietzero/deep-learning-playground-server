from flask import request
from environments import GridWorld

gridworld = GridWorld()

def GridWorldController(app):
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

