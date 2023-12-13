class TrainConfig:
    def __init__(
        self,
        epochs,
        learning_rate=1e-3,
        batch_size=1,
        # discount factor
        gamma=0.9,
        # initialized as 1 and then decrease
        epsilon=1.0
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size