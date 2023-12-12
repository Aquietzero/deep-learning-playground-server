import json
import numpy as np
from random import randrange

class Grid:
    def __init__(self, id, type=None):
        self.id = id
        self.type = type

    def is_goal(self):
        return self.type == 'goal'

    def is_done(self):
        return self.type == 'goal' or self.type == 'pit'


class GridWorld:
    def __init__(self, size=4, truncate=15, mode='static'):
        self.size = size
        self.mode = mode
        self.current_state = 0
        self.grids = {}
        self.steps = 0
        self.truncate = truncate

        self.init()

    def encode_state(self, noise=True):
        # the whole tensor consists of 4 layers, each represents
        # player, goal, pit and wall
        state = np.zeros(self.size * self.size * 4)
        offsets = {
            'goal': self.size * self.size,
            'pit': self.size * self.size * 2,
            'wall': self.size * self.size * 3,
        }
        state[self.current_state] = 1
        for grid in self.grids.values():
            if grid.type != None and grid.type != 'start':
                offset = offsets[grid.type]
                state[offset + grid.id] = 1

        # flatten
        state = state.reshape(1, self.size * self.size * 4)
        # add noise to state
        if noise:
            state = state + np.random.rand(1, self.size * self.size * 4) / 10.0

        return state

    def state_hash(self, row, col):
        return row * self.size + col

    def position_from_hash(self, state_hash):
        row = state_hash // self.size
        col = state_hash % self.size
        return row, col

    def inbound(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def init(self):
        for row in range(self.size):
            for col in range(self.size):
                id = self.state_hash(row, col)
                self.grids[id] = Grid(id)

        if self.mode == 'static':
            self.set_grid(0, 0, 'start')
            self.set_grid(0, 3, 'goal')
            self.set_grid(0, 2, 'pit')
            self.set_grid(2, 1, 'pit')
            self.set_grid(2, 3, 'pit')
        if self.mode == 'random':
            self.set_grid(0, 0, 'start')
            goal_row = randrange(1, self.size)
            goal_col = randrange(1, self.size)
            self.set_grid(goal_row, goal_col, 'goal')
            pit_count = 3 
            while pit_count > 0:
                pit_row = randrange(1, self.size)
                pit_col = randrange(1, self.size)
                if (pit_row != goal_row) or (pit_col != goal_col):
                    pit_count -= 1
                    self.set_grid(pit_row, pit_col, 'pit')

        self.steps = 0

    def set_grid(self, row, col, type):
        id = self.state_hash(row, col)
        self.grids[id].type = type

    def actions():
        # 0 up
        # 1 right
        # 2 down
        # 3 left
        return [0, 1, 2, 3]

    def take_action(self, action):
        self.steps += 1
        return self._step(self.current_state, action)

    def step(self, current_state, action):
        return self._step(current_state, action)

    def _step(self, current_state, action):
        done = False

        row, col = self.position_from_hash(current_state)
        next_state = current_state
        if action == 0 and self.inbound(row - 1, col): next_state = self.state_hash(row - 1, col)
        if action == 1 and self.inbound(row, col + 1): next_state = self.state_hash(row, col + 1)
        if action == 2 and self.inbound(row + 1, col): next_state = self.state_hash(row + 1, col)
        if action == 3 and self.inbound(row, col - 1): next_state = self.state_hash(row, col - 1)

        done = self.grids[next_state].is_done()
        truncated = self.steps >= self.truncate

        self.current_state = next_state
        return next_state, done, truncated

    def reward(self):
        if self.grids[self.current_state].type == 'pit':
            return -10
        if self.grids[self.current_state].type == 'goal':
            return 10
        else:
            return -1