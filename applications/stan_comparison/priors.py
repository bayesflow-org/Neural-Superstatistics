import numpy as np
from scipy.stats import beta, halfnorm


LOWER_BOUNDS = np.array([0.0, 0.0, 0.0])
UPPER_BOUNS = np.array([8.0, 6.0, 1.0])


def sample_scale(a=1, b=25, num_params=3, rng=None):
    """Generates num_params random draws from a half-normal prior over the
    scale parameters of the Gaussian random walk.

    Parameters:
    -----------
    a          : float, optional, default: 1
        The first parameter of the beta distribution
    b          : float, optional, default: 25
        The second parameter of the beta distribution
    num_params : int, default: 3
        Number of local parameters
    rng        : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    scale : float
        The randomly drawn scale parameter
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.beta(a, b, num_params)


def sample_random_walk(sigmas, num_steps=100, lower_bounds=LOWER_BOUNDS, upper_bounds=UPPER_BOUNS, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigmas          : float
        The standard deviations of the random walk process
    num_steps       : int, optional, default: 100
        The number of time steps to take for the random walk. Default
        corresponds to the number of years in the Coal Mining Diseaser Dataset
    lower_bounds    : int, optional, default: 0
        The minimum value the parameter(s) can take.
    upper_bounds    : int, optional, default: 8
        The maximum value the parameter(s) can take.
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
    theta_t[0, 0] = rng.gamma(shape=5.0, scale=1.0/3.0)
    theta_t[0, 1] = rng.gamma(shape=4.0, scale=1.0/3.0)
    theta_t[0, 2] = rng.gamma(shape=1.5, scale=1.0/5.0)

    # Run random walk from initial
    z = rng.random(size=(num_steps - 1, 3))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + sigmas * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t