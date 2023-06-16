import numpy as np
from scipy.stats import halfnorm

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

def sample_switch_prob(min=0.0, max=0.2, rng=None):
    """Generates 2 random draws from a uniform prior over the
    switch probability of the regime switching transition.

    Parameters:
    -----------
    min          : float, optional, default: 0.0
        The minimum of the uniform distribution.
    max          : float, optional, default: 0.2
        The maximum of the uniform distribution.
    rng : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    switch_probs : np.array
        The randomly drawn switch probability parameters.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(min, max, 2)

def sample_variability(loc=0.0, scale=0.2):
    """Generates 3 random draws from a half-normal prior over the
    scales of the stationary variability.

    Parameters:
    -----------
    loc          : float, optional, default: 0.0
        The location of teh half-normal distribution.
    scale        : float, optional, default: 0.2
        The scale of teh half-normal distribution.

    Returns:
    --------
    variabilities : np.array
        The randomly drawn variability parameters.
    """

    return halfnorm.rvs(loc, scale, 3)

def sample_ddm_params(loc=[0.0, 0.0, 0.0], scale=[2.5, 2.5, 1.0]):
    """Generates random draws from a half-normal prior over the
    diffusion decision parameters, v, a, tau.

    Parameters:
    -----------
    loc        : list, optional, default: [0.0, 0.0, 0.0]
        The shapes of the half-normal distribution.
    scale      : list, optional, default: [2.5, 2.5, 1.0]
        The scales of the half-normal distribution.

    Returns:
    --------
    ddm_params : np.array
        The randomly drawn DDM parameters, v, a, tau.
    """

    return halfnorm.rvs(loc=loc, scale=scale)

def sample_shared_tau(loc=0.0, scale=1.0):
    """Generates a random draw from a half-normal prior over the
    shared tau parameter.

    Parameters:
    -----------
    loc        : flaot, optional, default: 0.0
        The shape of the half-normal distribution.
    scale      : list, optional, default: 1.0
        The scales of the half-normal distribution.

    Returns:
    --------
    tau        : np.array
        The randomly drawn tau parameters.
    """

    return halfnorm.rvs(loc=loc, scale=scale)

def sample_random_walk(sigma, num_steps=1320, lower_bounds=[0, 0, 0], upper_bounds=[8, 6, 4], rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigma           : np.array
        The standard deviations of the random walk process.
    num_steps       : int, optional, default: 1320
        The number of time steps to take for the random walk. Default
        corresponds to the maximal number of trials in the Optimal Policy Dataset.
    lower_bounds    : list, optional, default: [0, 0, 0]
        The minimum values the parameters can take.
    upper_bound     : list, optional, default: [8, 6, 1]
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

def sample_stationary_variability(variability, num_steps=1320, lower_bounds=[0, 0, 0], upper_bounds=[8, 6, 4], rng=None):
    """Generates a single simulation from a stationary variabillty model.

    Parameters:
    -----------
    variability      : np.ndarray of shape (3, )
        The scales of the stationary variabillty process.
    num_steps       : int, optional, default: 1320
        The number of time steps to take for stationary variabillty process. Default
        corresponds to the maximal number of trials in the Optimal Policy Dataset.
    lower_bounds    : list, optional, default: [0, 0, 0]
        The minimum values the parameters can take.
    upper_bound     : list, optional, default: [8, 6, 1]
        The maximum values the parameters can take.
    rng             : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, 3)
        The array of time-varying parameters
    """

    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_ddm_params()
    # Random variability process for v and a
    theta_t[1:, :2] = np.clip(
        rng.normal(loc=theta_t[0, :2], scale=variability[:2], size=(num_steps, 2)),
        lower_bounds, upper_bounds
        )
    # Random variability process for tau
    theta_t[1:, 2] = np.clip(
        rng.uniform(low=theta_t[0, 2] - variability[2]/2, high=theta_t[0, 2] + variability[2]/2),
        lower_bounds, upper_bounds
        )
    return theta_t

def sample_regime_switch(switch_prob, shared_tau, num_steps=1320, lower_bounds=[0, 0], upper_bounds=[8, 6], rng=None):
    """Generates a single simulation from a regime switching model.

    Parameters:
    -----------
    switch_prob     : np.ndarray of shape (2, )
        The probability for a jump parameters
    shared_tau      : float
        The shared tau parameter. Tau is not allowed to jump.
    num_steps       : int, optional, default: 1320
        The number of time steps to take for the random walk. Default
        corresponds to the maximal number of trials in the Optimal Policy Dataset.
    lower_bounds    : list, optional, default: [0, 0]
        The minimum values the parameters can take.
    upper_bound     : list, optional, default: [8, 6]
        The maximum values the parameters can take.
    rng             : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying and shared parameters
    """

    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_ddm_params()
    # shared tau parameter
    theta_t[:, 2] = np.repeat(shared_tau, num_steps)
    # sample regime switches
    switch = np.random.rand(num_steps, 2)
    for p in range(2):
        for t in range(1, num_steps):
            if switch[p, t] > switch_prob[p]:
                theta_t[p, t] = theta_t[p, t-1]
            else:
                theta_t[p, t] = rng.uniform(lower_bounds[p], upper_bounds[p])                
    return theta_t
