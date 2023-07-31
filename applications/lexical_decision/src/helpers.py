import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions

def squared_dist(x): 
    """Efficiently computes the squared pairwise distance between the element of the vector x."""
    expanded_a = tf.expand_dims(x, 1)
    expanded_b = tf.expand_dims(x, 0)
    distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
    return distances

def RBF(dist, amplitude, length_scale):
    """Computes the radial basis function kernel given a distance matrix and two parameters."""
    return (amplitude**2) * tf.math.exp(-1. / (2 * length_scale**2) * dist)

def GP(mean, cov_matrix, n_samples=1):
    """Draw n_samples from a GP given mean and cov_matrix."""
    samples = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=cov_matrix).sample(n_samples)
    if n_samples == 1:
        return tf.squeeze(samples)
    return samples

def prior_length_scale(lower=0.1, upper=10.):
    return np.random.default_rng().uniform(lower, upper)

def build_distance_matrix(T):
    return squared_dist(np.linspace(0, 1, T)[:, np.newaxis].astype(np.float32))


