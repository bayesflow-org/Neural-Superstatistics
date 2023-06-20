import numpy as np
from numba import njit

from priors import sample_ddm_params

@njit
def _sample_diffusion_trial(v, a, tau, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single response time from a Diffusion Decision process.

    Parameters:
    -----------
    v        : float
        The drift rate parameter.
    a        : float
        The boundary separation parameter.
    tau      : float
        The non-decision time parameter.
    beta     : float, optional, default: 0.5
        The starting point parameter. The default corresponds to
        no a priori bias.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : float
        A response time samples from the static Diffusion decision process.
        Reaching the lower boundary results in a negative rt.
    """
    n_iter = 0
    x = a * beta
    c = np.sqrt(dt * s)
    while x > 0 and x < a and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
    rt = n_iter * dt
    return rt+tau if x >= 0 else -(rt+tau)

@njit
def sample_static_diffusion_process(theta, num_steps=400, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single simulation from a static
    Diffusion decision process with a static parameters.

    Parameters:
    -----------
    theta     : np.array of shape (3, )
        The 3 latent DDM parameters, v, a, tau.
    num_steps : int, optional, default: 400
        The number of time steps to take for the static diffusion process. Default
        corresponds to the number of time steps in the simulation study.
    beta     : float, optional, default: 0.5
        The starting point parameter. The default corresponds to
        no a priori bias.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : np.array of shape (num_steps, )
        Response time samples from the static Diffusion decision process.
        Reaching the lower boundary results in negative rt's.
    """

    rt = np.zeros(num_steps)
    for t in range(num_steps):
        rt[t] = _sample_diffusion_trial(
            theta[0], theta[1], theta[2], beta,
            dt=dt, s=s, max_iter=max_iter)
    return rt

@njit
def sample_stationary_diffusion_process(theta_t, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single simulation from a Stationary
    Diffusion Decision process with random variability.

    Parameters:
    -----------
    theta_t : np.ndarray of shape (theta_t, 3)
        The trajectory of the 3 latent DDM parameters, v, a, tau.
        corresponds to the maximal number of trials in the Optimal Policy Dataset.
    beta     : float, optional, default: 0.5
        The starting point parameter. The default corresponds to
        no a priori bias.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : np.array of shape (num_steps, )
        Response time samples from the Stationary Diffusion decision process.
        Reaching the lower boundary results in negative rt's.
    """

    num_steps = theta_t.shape[0]
    rt = np.zeros(num_steps)
    for t in range(num_steps):
        rt[t] = _sample_diffusion_trial(
            theta_t[t, 0], theta_t[t, 1], theta_t[t, 2], beta,
            dt=dt, s=s, max_iter=max_iter)
    return rt

@njit
def sample_random_walk_diffusion_process(theta_t, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single simulation from a non-stationary
    Diffusion decision process with a parameters following a random walk.

    Parameters:
    -----------
    theta_t : np.ndarray of shape (theta_t, 3)
        The trajectory of the 3 latent DDM parameters, v, a, tau.
    beta     : float, optional, default: 0.5
        The starting point parameter. The default corresponds to
        no a priori bias.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : np.array of shape (num_steps, )
        Response time samples from the Random Walk Diffusion decision process.
        Reaching the lower boundary results in negative rt's.
    """

    num_steps = theta_t.shape[0]
    rt = np.zeros(num_steps)
    for t in range(num_steps):
        rt[t] = _sample_diffusion_trial(
            theta_t[t, 0], theta_t[t, 1], theta_t[t, 2], beta,
            dt=dt, s=s, max_iter=max_iter)
    return rt

@njit
def sample_regime_switching_diffusion_process(theta_t, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single simulation from a Regime Switching
    Diffusion decision process with a parameters following sudden uniformly distributed jumps.

    Parameters:
    -----------
    theta_t : np.ndarray of shape (theta_t, 3)
        The trajectory of the 3 latent DDM parameters, v, a, tau.
    beta     : float, optional, default: 0.5
        The starting point parameter. The default corresponds to
        no a priori bias.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : np.array of shape (num_steps, )
        Response time samples from the Regime Switching Diffusion decision process.
        Reaching the lower boundary results in negative rt's.
    """

    num_steps = theta_t.shape[0]
    rt = np.zeros(num_steps)
    for t in range(num_steps):
        rt[t] = _sample_diffusion_trial(
            theta_t[t, 0], theta_t[t, 1], theta_t[t, 2], beta,
            dt=dt, s=s, max_iter=max_iter)
    return rt