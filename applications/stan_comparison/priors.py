import numpy as np

from configuration import default_lower_bounds, default_upper_bounds

def sample_scale(alpha=1.0, beta=25.0, rng=None):
    """Generates 3 random draws from a beta prior over the
    scale of the random walk.

    Parameters:
    -----------
    alpha : float, optional, default: 1.0
        The alpha parameter of the beta distribution.
        Default corresponds to the prior specification used with Stan.
    beta  : float, optional, default: 25.0
        The beta parameter of the beta distribution.
        Default corresponds to the prior specification used with Stan.
    rng   : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.
    Returns:
    --------
    scales : np.array
        The randomly drawn scale parameters.
    """

    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()

    return rng.beta(alpha=alpha, beta=beta, size=3)

def sample_ddm_params(shape=(5.0, 4.0, 1.5), scale=(1/3, 1/3, 1/5), rng=None):
    """Generates random draws from a gamma prior over the
    diffusion decision parameters, v, a, tau.

    Parameters:
    -----------
    shape      : tuple, optional, default: (5.0, 4.0, 1.5)
        The shapes of the gamma distribution.
        Default corresponds to the prior specification used with Stan.
    scale      : tuple, optional, default: (1/3, 1/3, 1/5)
        The scales of the gamma distribution.
        Default corresponds to the prior specification used with Stan.
    rng        : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.
    Returns:
    --------
    ddm_params : np.array
        The randomly drawn DDM parameters, v, a, tau.
    """

    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()

    return rng.gamma(shape=shape, scale=scale)

def sample_random_walk(sigma, num_steps=100, lower_bounds=default_lower_bounds, upper_bounds=default_upper_bounds, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigma           : np.array
        The standard deviations of the random walk process.
    num_steps       : int, optional, default: 100
        The number of time steps to take for the random walk.
        Default corresponds to the number of time steps in the Stan implementation.
    lower_bounds    : tuple, optional, default: ``configuration.default_lower_bounds``
        The minimum values the parameters can take.
    upper_bound     : tuple, optional, default: ``configuration.default_upper_bounds``
        The maximum values the parameters can take.
    rng             : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """

    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_ddm_params()

    # Run random walk from initial
    z = rng.normal(size=(num_steps - 1, 3))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + sigma * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t
