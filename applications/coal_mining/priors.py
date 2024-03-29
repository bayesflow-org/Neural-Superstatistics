import numpy as np


def sample_scale(a=1, b=25, rng=None):
    """Generates a single random draw from a beta prior over the
    scale of the random walk. The artificial bounds are for comparability with bayesloop.

    Parameters:
    -----------
    a   : float, optional, default: 1
        The first parameter of the beta distribution
    b   : float, optional, default: 25
        The second parameter of the beta distribution
    rng : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    scale : float
        The randomly drawn scale parameter
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.beta(a, b)


def sample_random_walk(sigma, num_steps=110, lower_bound=0, upper_bound=8, expon_scale=1.0, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigmas          : float
        The standard deviations of the random walk process
    num_steps       : int, optional, default: 110
        The number of time steps to take for the random walk. Default
        corresponds to the number of years in the Coal Mining Diseaser Dataset
    lower_bound     : int, optional, default: 0
        The minimum value the parameter(s) can take.
    upper_bound     : int, optional, default: 8
        The maximum value the parameter(s) can take.
    expon_scale     : float, optional, default: 0.5
        The scale of the exponential prior for the first parameter value
    rng             : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """

    # Configure RNG, if provided
    if rng is None:
        rng = np.random.default_rng()

    # Sample initial rate
    theta_t = np.zeros(num_steps)
    theta_t[0] = rng.exponential(scale=expon_scale)

    # Run random walk from initial
    z = rng.normal(size=num_steps - 1)
    for t in range(1, num_steps):
        theta_t[t] = np.clip(theta_t[t - 1] + sigma * z[t - 1], a_min=0, a_max=8)
    return theta_t
