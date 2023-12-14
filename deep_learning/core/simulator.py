import torch
import numpy as np

class Simulator:
    def __init__(self, episodes, env_producer):
        self.episodes = episodes
        self.env_producer = env_producer
        self.env = None

    def runover(self, model):
        env = self.env_producer()
        self.env = env
        path = [env.current_state]
        done = False
        while not done:
            state_ = env.encode_state(noise=False)
            state = torch.from_numpy(state_).float()
            qval = model(state)
            qval_ = qval.data.numpy()
            action = np.argmax(qval_)
            _, done, truncated = env.take_action(action)

            path.append(env.current_state)

            if truncated:
                return { 'success': False, 'path': path }
            if done:
                if env.reward() > 0:
                    return { 'success': True, 'path': path }
                else:
                    return { 'success': False, 'path': path }

    def simulate(self, model, socketio):
        success_count = 0
        for i in range(self.episodes):
            result = self.runover(model)
            if result['success']:
                success_count += 1

            socketio.emit('test_progress', {
                'num_games': self.episodes,
                'current': i + 1,
            })
            socketio.emit('testing_info', {
                'success_rate': success_count / (i + 1)
            })

        print('result: %d / %d' % (success_count, self.episodes))