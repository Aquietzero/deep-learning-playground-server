from random import shuffle
import numpy as np
import torch

class ExperienceReplay:
    def __init__(self, N=500, batch_size=100):
        self.N = N
        self.batch_size = batch_size
        self.memory = []
        self.counter = 0

    def size(self):
        return len(self.memory)

    def add_memory(self, state1, action, reward, state2, done):
        self.counter += 1
        if self.counter % self.N == 0:
            self.shuffle_memory()

        # if the memory is not full, adds to the list
        # otherwise replaces a random memory with the new one
        exp = (state1, action, reward, state2, done)
        if len(self.memory) < self.N:
            self.memory.append(exp)
        else:
            rand_index = np.random.randint(0, self.N - 1)
            self.memory[rand_index] = exp

    def shuffle_memory(self):
        shuffle(self.memory)

    def get_batch(self):
        if len(self.memory) < self.batch_size:
            batch_size = len(self.memory)
        else:
            batch_size = self.batch_size
        
        if len(self.memory) < 1:
            print('Error: No data in memory.')
            return None

        indexes = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        batch = [self.memory[i] for i in indexes]
        state1_batch = torch.cat([exp[0] for exp in batch])
        action_batch = torch.Tensor([exp[1] for exp in batch]).long()
        reward_batch = torch.Tensor([exp[2] for exp in batch])
        state2_batch = torch.cat([exp[3] for exp in batch])
        done_batch = torch.Tensor([exp[4] for exp in batch])
        return state1_batch, action_batch, reward_batch, state2_batch, done_batch