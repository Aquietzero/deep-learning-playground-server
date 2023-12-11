import os
import numpy as np

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def moving_average(arr, window_size):
    cumsum = np.cumsum(arr, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size