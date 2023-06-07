import numpy as np


def sample_poisson_process(theta_t, rng=None):
    """Generates a single simulation from a Poisson process with a varying
    rate parameter.

    Parameters:
    -----------
    theta_t : np.ndarray of shape (theta_t, )
        The latent parameter trajectory
    rng     : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """

    if rng is None:
        rng = np.random.default_rng()

    num_steps = theta_t.shape[0]
    observations = np.zeros(num_steps)
    for t in range(num_steps):
        observations[t] = rng.poisson(lam=theta_t[t])
    return observations
