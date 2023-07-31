import numpy as np
from numba import njit
import tensorflow as tf

from helpers import RBF, GP


@njit
def static_params(theta0, T):
    """ TODO

    """

    B = theta0.shape[0]
    D = theta0.shape[1]
    theta_t = np.zeros((B, T, D))
    for t in range(T):
        theta_t[:, t, :] = theta0
    return theta_t


def random_walk(theta0, sigmas, T, lower_bound=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], upper_bound=[8.0, 8.0, 8.0, 8.0, 6.0, 4.0]):
    """ TODO

    """

    B = theta0.shape[0]
    D = theta0.shape[1]
    if D == 3:
        lower_bound=[0.0, 0.0, 0.0]
        upper_bound=[8.0, 6.0, 4.0]
    theta_t = np.zeros((B, T, D))
    theta_t[:, 0, :] = theta0
    z = np.random.randn(B, T-1, D)
    for t in range(1, T):
        theta_t[:, t, :] = np.minimum(
            np.maximum(theta_t[:, t-1, :] + sigmas * z[:, t-1, :], lower_bound),
            upper_bound
        )
    return theta_t


def random_walk_shared_var(theta0, sigmas, T, lower_bound=[0.0, 0.0, 0.0, 0.0, 0.01, 0.01], upper_bound=[8, 8, 8, 8, 6, 4], n_drift=4):
    """ TODO

    """

    sigmas = np.c_[np.stack([sigmas[:, 0]] * n_drift, axis=0).T, sigmas[:, 1:]]
    B = theta0.shape[0]
    D = theta0.shape[1]
    theta_t = np.zeros((B, T, D))
    theta_t[:, 0, :] = theta0
    z = np.random.randn(B, T-1, D)
    for t in range(1, T):
        theta_t[:, t, :] = np.minimum(
            np.maximum(theta_t[:, t-1, :] + sigmas * z[:, t-1, :], lower_bound),
            upper_bound
        )
    return theta_t

def random_walk_mixture(theta0, T, sd=0.1, q=0.01):
    theta_t = []
    theta_t.append(theta0)
    z = np.random.randn(T-1)
    probs = np.random.rand(T-1)
    for t in range(T-1):
        if probs[t] < q:
            theta_t.append(np.random.uniform(low=0.01, high=max(theta_t)))
        else:
            theta_t.append(max(0, theta_t[t] + z[t] * sd))
    return np.array(theta_t)

def gaussian_process(prior_samples, distance_matrix, length_scales=10, amplitudes=0.1, min_val=0.05):
    """Generates draws from a Gaussian process."""

    # Extract time steps (T) and n params (M)
    T = distance_matrix.shape[0]
    M = prior_samples.shape[0]

    # Handle case amplitude shared across parameters
    if type(amplitudes) is float:
        amplitudes = [amplitudes] * M

    # Handle case length scale shared across parameters
    if type(length_scales) is float:
        length_scales = [length_scales] * M

    # Create covariance matrices
    cov_matrices = tf.stack([RBF(distance_matrix, amp, ls) for amp, ls in zip(amplitudes, length_scales)])

    # Create mean
    means = tf.stack([np.array([prior_samples[m]] * T).astype(np.float32) for m in range(M)])

    # Draw GP samples
    samples = GP(mean=means, cov_matrix=cov_matrices, n_samples=1).numpy()
    samples = np.array(samples).astype(np.float32)
    samples[samples < min_val] = min_val 
    # Return samples
    return samples.T

def batched_gaussian_process(theta0, distance_matrix, length_scales, amplitudes=0.1, min_val=0.05):
    """Generates a batch of samples from a GP."""

    B = theta0.shape[0]
    gp_samples = np.array([gaussian_process(
        theta0[b], distance_matrix, length_scales[b], amplitudes, min_val) for b in range(B)])
    return gp_samples