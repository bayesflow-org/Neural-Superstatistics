import numpy as np
from numpy.random import default_rng

def diffusion_prior(batch_size, n_cond=4):
    """ TODO
    Parameters
    ----------
    TODO
    Returns
    -------
    TODO
    """
    rng = default_rng()
    a = rng.gamma(4.0, 1/3.0, batch_size)
    ndt = rng.gamma(1.5, 1/5.0, batch_size)
    v = rng.gamma(5, 1/3.0, (batch_size, n_cond))
    return np.c_[v, a, ndt]


def random_walk_prior(batch_size, n_params, alpha=1., beta=25.):
    """ TODO
    Parameters
    ----------
    TODO
    Returns
    -------
    TODO
    """

    return default_rng().beta(alpha, beta, size=(batch_size, n_params))


def length_scale_prior(batch_size, n_params, lower=0.1, upper=10.):
    return np.random.default_rng().uniform(lower, upper, size=(batch_size, n_params)).astype(np.float32)
