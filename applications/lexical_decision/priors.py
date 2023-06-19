import numpy as np
from scipy.stats import halfnorm

LOWER_BOUNDS = np.array([0, 0, 0, 0, 0, 0])
UPPER_BOUNDS = np.array([8, 8, 8, 8, 6, 4])

def sample_scale(loc=[0, 0, 0], scale=[0.1, 0.1, 0.01]):
    """Generates 3 random draws from a half-normal prior over the
    scale of the random walk.

    Parameters:
    -----------
    loc    : list, optional, default: [0, 0, 0]
        The location parameters of the half-normal distribution.
    scale  : flist, optional, default: [0.1, 0.1, 0.01]
        The scale parameters of the half-normal distribution.

    Returns:
    --------
    scales : np.array
        The randomly drawn scale parameters.
    """

    return halfnorm.rvs(loc=loc, scale=scale)

def sample_ddm_params(shapes=[5.0, 5.0, 5.0, 5.0, 4.0, 1.5], scales=[1/3.0, 1/3.0, 1/3.0, 1/3.0, 1/3.0, 1/5.0], rng=None):
    """Generates random draws from a gamma prior over the
    diffusion decision parameters,  4 * v, a, tau.

    Parameters:
    -----------
    shapes     : list, optional, default: [5.0, 5.0, 5.0, 5.0, 4.0, 1.5]
        The shapes of the gamma distribution.
    scales     : list, optional, default: [1/3.0, 1/3.0, 1/3.0, 1/3.0, 1/3.0, 1/5.0]
        The scales of the gamma distribution.
    rng : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    ddm_params : np.array
        The randomly drawn DDM parameters, 4 * v, a, tau.
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.gamma(shape=shapes, scale=scales)

def sample_random_walk(sigmas, num_steps=3200, lower_bounds=LOWER_BOUNDS, upper_bounds=UPPER_BOUNDS, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigmas          : np.array
        The standard deviations of the random walk process.
    num_steps       : int, optional, default: 3200
        The number of time steps to take for the random walk. Default
        corresponds to the maximal number of trials in the Optimal Policy Dataset.
    lower_bounds    : np.array, optional, default: [0, 0, 0, 0, 0 , 0]
        The minimum values the parameters can take.
    upper_bound     : np.array, optional, default: [8, 8, 8, 8, 6, 4]
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
    theta_t = np.zeros((num_steps, 6))
    theta_t[0] = sample_ddm_params()
    # All drift rates share the same sigma
    sigmas = np.c_[np.stack([sigmas[:, 0]] * 4, axis=0).T, sigmas[:, 1:]]
    # Run random walk from initial
    z = rng.normal(size=(num_steps - 1, 6))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + sigmas * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t
