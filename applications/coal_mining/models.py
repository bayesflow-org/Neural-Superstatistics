import numpy as np
from functools import partial

import bayesflow as bf

from priors import sample_random_walk, sample_scale
from likelihoods import sample_poisson_process


class PoissonRandomWalk:
    """A wrapper for a non-stationary Poisson process with a random walk transition."""

    def __init__(self, rng=None):
        """Creates an instance of the superstatistical model with given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure`` should be used.

        Parameters:
        -----------
        rng : np.random.Generator or None, default: None
            An optional random number generator to use, if fixing the seed locally.
        """

        # Store local RNG instance
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        # Create prior wrapper
        self.prior = bf.simulation.TwoLevelPrior(
            hyper_prior_fun=partial(sample_scale, rng=self._rng),
            local_prior_fun=partial(sample_random_walk, rng=self._rng),
        )

        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=partial(sample_poisson_process, rng=self._rng),
        )

        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.TwoLevelGenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="poisson_model",
        )

    def generate(self, batch_size, **kwargs):
        """Wraps the call function of ``bf.simulation.TwoLevelGenerativeModel``.

        Parameters:
        -----------
        batch_size : int
            The number of simulations to generate per training batch
        **kwargs   : dict, optional, default: {}
            Optional keyword arguments passed to the call function of ``bf.simulation.TwoLevelGenerativeModel``

        Returns:
        --------
        raw_dict   : dict
            The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
        """

        return self.generator(batch_size, **kwargs)

    def configure(self, raw_dict):
        """Configures the output of self.generator for a BayesFlow pipeline.

        1. Converts float64 to float32 (for TensorFlow)
        2. Appends a trailing dimensions of 1, since model is 1D
        3. Log-transforms parameters and observations

        Parameters:
        -----------
        raw_dict : dict
            A simulation dictionary as returned by ``bayesflow.simulation.TwoLevelGenerativeModel``

        Returns:
        --------
        input_dict : dict
            The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
        """

        # Extract relevant simulation data, convert to float32, and add extra dimension
        # The latter step is needed, since everything is 1D for this toy model
        rates = raw_dict.get("local_prior_draws").astype(np.float32)[..., None]
        scales = raw_dict.get("hyper_prior_draws").astype(np.float32)[..., None]
        observations = raw_dict.get("sim_data").astype(np.float32)[..., None]

        # Pack everything into a dict and log transform data and parameters
        out_dict = dict(
            local_parameters=np.log1p(rates),
            hyper_parameters=np.log1p(scales),
            summary_conditions=np.log1p(observations),
        )
        return out_dict
