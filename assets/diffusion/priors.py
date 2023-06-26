import numpy as np
from scipy.stats import halfnorm

from configuration import *

def sample_scale(loc=default_scale_prior_loc, scale=default_scale_prior_scale):
    """Generates 3 random draws from a half-normal prior over the
    scale of the random walk.

    Parameters:
    -----------
    loc    : tuple, optional, default: ``configuration.default_scale_prior_loc``
        The location parameters of the half-normal distribution.
    scale  : tuple, optional, default: ``configuration.default_scale_prior_scale``
        The scale parameters of the half-normal distribution.

    Returns:
    --------
    scales : np.array
        The randomly drawn scale parameters.
    """

    return halfnorm.rvs(loc=loc, scale=scale)

def sample_ddm_params(loc=default_ddm_params_prior_loc, scale=default_ddm_params_prior_scale):
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

def sample_variability(loc=default_variability_prior_loc, scale=default_variability_prior_scale):
    """Generates 3 random draws from a half-normal prior over the
    scales of the stationary variability.

    Parameters:
    -----------
    loc          : tuple, optional, default: ``configuration.default_variability_prior_loc``
        The location of the half-normal distribution.
    scale        : tuple, optional, default: ``configuration.default_variability_prior_scale``
        The scale of the half-normal distribution.

    Returns:
    --------
    variabilities : np.array
        The randomly drawn variability parameters.
    """

    return halfnorm.rvs(loc=loc, scale=scale)

def sample_random_walk(sigma, num_steps=1320, lower_bounds=default_lower_bounds, upper_bounds=default_upper_bounds, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigma           : np.array
        The standard deviations of the random walk process.
    num_steps       : int, optional, default: 1320
        The number of time steps to take for the random walk. Default
        corresponds to the maximal number of trials in the Optimal Policy Dataset.
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

def sample_regime_switching(num_steps=400, points_of_jump=default_points_of_jump, lower_bounds=default_lower_bounds, upper_bounds=default_upper_bounds, rng=None):
    """Generates a single simulation from a regime switching model.

    Parameters:
    -----------
    num_steps       : int, optional, default: 400
        The number of time steps to take for the regime swiching model. Default
        corresponds to the maximal number of time steps in the simulatino study.
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

    for t in range(1, num_steps):
        if np.any(np.array(points_of_jump) == t):
            theta_t[t] = rng.uniform(lower_bounds, upper_bounds)
        else:
            theta_t[t] = theta_t[t-1]

    return theta_t
