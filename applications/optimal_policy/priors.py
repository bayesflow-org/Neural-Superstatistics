import numpy as np

LOWER_BOUNDS = np.array([0, 0, 0])
UPPER_BOUNDS = np.array([8, 6, 1])

DDM_PARAM_PRIOR_SHAPES = np.array([5.0, 4.0, 1.5])
DDM_PARAM_PRIOR_SCALES = np.array([1/3.0, 1/3.0, 1/5.0])

def sample_scale(a=1, b=25, num_params=3, rng=None):
    """Generates num_params random draws from a beta prior over the
    scale of the random walk.

    Parameters:
    -----------
    a          : float, optional, default: 1
        The first parameter of the beta distribution.
    b          : float, optional, default: 25
        The second parameter of the beta distribution.
    num_params : int, optional, default: 3
        The number of scale parameters.
    rng        : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    scale : np.array
        The randomly drawn scale parameters.
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.beta(a, b, num_params)

def sample_switch_prob(min=0.0, max=0.2, num_params=2, rng=None):
    """Generates num_params random draws from a uniform prior over the
    switch probability of the regime switching transition.

    Parameters:
    -----------
    min        : float, optional, default: 0.0
        The minimum of the uniform distribution.
    max        : float, optional, default: 0.2
        The maximum of the uniform distribution.
    num_params : int, optional, default: 2
        The number of switch probability parameters.
    rng : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    switch_prob : np.array
        The randomly drawn switch probability parameters.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(min, max, num_params)

def sample_ddm_params(shapes=DDM_PARAM_PRIOR_SHAPES, scales=DDM_PARAM_PRIOR_SCALES, rng=None):
    """Generates random draws from a gamma prior over the
    diffusion decision parameters, v, a, tau.

    Parameters:
    -----------
    shapes     : np.array, optional, default: [5.0, 4.0, 1.5]
        The shapes of the gamma distribution.
    scales     : np.array, optional, default: [1/3.0, 1/3.0, 1/5.0]
        The scales of the gamma distribution.
    rng : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    ddm_params : np.array
        The randomly drawn DDM parameters, v, a, tau.
    """

    if rng is None:
        rng = np.random.default_rng()
    return rng.gamma(shape=shapes, scale=scales)

def sample_random_walk(sigmas, num_steps=1320, lower_bounds=LOWER_BOUNDS, upper_bounds=UPPER_BOUNDS, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigmas          : np.array
        The standard deviations of the random walk process.
    num_steps       : int, optional, default: 1320
        The number of time steps to take for the random walk. Default
        corresponds to the maximal number of trials in the Optimal Policy Dataset.
    lower_bounds    : np.array, optional, default: [0, 0, 0]
        The minimum values the parameters can take.
    upper_bound     : np.array, optional, default: [8, 6, 1]
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
            theta_t[t - 1] + sigmas * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t
