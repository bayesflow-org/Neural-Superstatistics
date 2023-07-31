import numpy as np
from numba import njit

@njit
def generate_design_matrix(batch_size, T):
    """
    Generates an experimenmt wiht a random sequence of 4 conditions.
    """
    obs_per_condition = int(T / 4)
    context = np.zeros((batch_size, T), dtype=np.int32)
    x = np.repeat([0, 1, 2, 3], obs_per_condition)
    for i in range(batch_size):
        np.random.shuffle(x)
        context[i] = x

    return context