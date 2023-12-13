class Trainer:
    def __init__(self, train_config):
        self.train_config = train_config

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()