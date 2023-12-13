import torch
import numpy as np

class Simulator:
    def __init__(self, episodes, env_producer):
        self.episodes = episodes
        self.env_producer = env_producer

    def runover(self, model):
        env = self.env_producer()
        done = False
        while not done:
            state_ = env.encode_state()
            state = torch.from_numpy(state_).float()
            qval = model(state)
            qval_ = qval.data.numpy()
            action = np.argmax(qval_)
            _, done, truncated = env.take_action(action)

            if done:
                if env.reward() > 0:
                    return True
                else:
                    return False
            if truncated:
                return False

    def simulate(self, model, socketio):
        win_count = 0
        for i in range(self.episodes):
            win = self.runover(model)
            if win:
                win_count += 1

            socketio.emit('test_progress', {
                'num_games': self.episodes,
                'current': i + 1,
            })
            socketio.emit('testing_info', {
                'success_rate': win_count / (i + 1)
            })

        print('result: %d / %d' % (win_count, self.episodes))