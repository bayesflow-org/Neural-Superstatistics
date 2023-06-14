import numpy as np
from numba import njit
import bayesflow as bf

class RandomWalkDDM:
    """A wrapper for a non-stationary Diffusion Decision Model with a gaussian random walk transition."""

    def __init__(self):
        """Creates an instance of the superstatistical model with given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure`` should be used.
        """

    @njit
    def generate(self, batch_size):
        self.test = 2