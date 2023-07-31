import numpy as np
from macro_models import *
from numba import njit


@njit
def diffusion_trial(v, a, ndt, zr=0.5, dt=0.001, s=1.0, max_iter=1e4):
    """
    Simulates a single reaction time from a simple drift-diffusion process.
    """

    n_iter = 0
    x = a * zr
    c = np.sqrt(dt * s)
    
    while x > 0 and x < a and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
        
    rt = n_iter * dt
    return rt+ndt if x >= 0 else -(rt+ndt)


@njit
def dynamic_diffusion(theta_t, context):
    """
    Performs one run of a dynamic diffusion model process with multiple drift rates and context.
    """
    
    T = context.shape[-1]
    rt = np.zeros(T)
    for t in range(T):
        rt[t] = diffusion_trial(theta_t[t, context[t]], theta_t[t, 4], theta_t[t, 5])
    return np.atleast_2d(rt).T


@njit
def simple_dynamic_diffusion(theta_t):
    """
    Performs one run of a dynamic diffusion model process with a single drift rate.
    """
    
    T = theta_t.shape[0]
    rt = np.zeros(T)
    for t in range(T):
        rt[t] = diffusion_trial(theta_t[t, 0], theta_t[t, 1], theta_t[t, 2])
    return np.atleast_2d(rt).T


@njit
def simple_static_diffusion(theta, T):
    """
    Performs one run of a dynamic diffusion model process with a single drift rate.
    """

    rt = np.zeros(T)
    for t in range(T):
        rt[t] = diffusion_trial(theta[0], theta[1], theta[2])
    return np.atleast_2d(rt).T


@njit
def dynamic_batch_diffusion(theta_t, context, diff_fun=dynamic_diffusion):
    B, T = context.shape[0], context.shape[1]
    rt = np.zeros((B, T, 1))
    for b in range(B):
        rt[b] = diff_fun(theta_t[b], context[b])
    return rt


@njit
def simple_batch_diffusion(theta_t, diff_fun=simple_dynamic_diffusion):
    B, T = theta_t.shape[0], theta_t.shape[1]
    rt = np.zeros((B, T, 1))
    for b in range(B):
        rt[b] = diff_fun(theta_t[b])
    return rt


@njit
def static_batch_diffusion(theta, T, diff_fun=simple_static_diffusion):
    B = theta.shape[0]
    rt = np.zeros((B, T, 1))
    for b in range(B):
        rt[b] = diff_fun(theta[b], T)
    return rt

def fast_dm_simulate(params, context):
    n_obs = context.shape[0]
    pred_rt = np.empty(n_obs)
    for t in range(n_obs):
        if params[6] < 0.00001:
            drift = params[context[t]]
        else:
            drift = np.random.normal(params[context[t]], params[6])
        ndt = np.random.uniform(params[5] - params[7]/2, params[5] + params[7]/2)

        pred_rt[t] = diffusion_trial(drift, params[4], ndt)

    return pred_rt