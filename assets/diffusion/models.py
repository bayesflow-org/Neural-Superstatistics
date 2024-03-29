from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import bayesflow as bf
from scipy.stats import beta

from diffusion.likelihoods import sample_static_diffusion_process, sample_stationary_diffusion_process, sample_random_walk_diffusion_process, sample_regime_switching_diffusion_process
from diffusion.priors import sample_scale, sample_variability, sample_ddm_params, sample_random_walk, sample_regime_switching

class DiffusionModel(ABC):
    """An interface for running a standardized simulated experiment."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, batch_size, *args, **kwargs):
        pass

    @abstractmethod
    def configure(self, raw_dict, *args, **kwargs):
        pass


class StaticDiffusion(DiffusionModel):
    """A wrapper for a static Diffusion Decision process."""

    def __init__(self):
        """Creates an instance of a static Diffusion Decision Model with given configuration.
        When used in a BayesFlow pipeline, only the attribute ``self.generator`` and
        the method ``self.configure`` should be used.
        """

        # Create prior wrapper
        self.prior = bf.simulation.Prior(
            prior_fun=sample_ddm_params,
        )

        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_static_diffusion_process,
        )

        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.GenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="static_diffusion_model",
        )

    def generate(self, batch_size, *args, **kwargs):
        """Wraps the call function of ``bf.simulation.GenerativeModel``.

        Parameters:
        -----------
        batch_size : int
            The number of simulations to generate per training batch
        **kwargs   : dict, optional, default: {}
            Optional keyword arguments passed to the call function of ``bf.simulation.GenerativeModel``

        Returns:
        --------
        raw_dict   : dict
            The simulation dictionary configured for ``bayesflow.amortizers.Amortizer``
        """

        return self.generator(batch_size, *args, **kwargs)

    def configure(self, raw_dict, transform=False):
        """Configures the output of self.generator for a BayesFlow pipeline.

        1. Converts float64 to float32 (for TensorFlow)
        2. Appends a trailing dimensions of 1 to data
        3. Scale the model parameters

        Parameters:
        -----------
        raw_dict  : dict
            A simulation dictionary as returned by ``bayesflow.simulation.GenerativeModel``
        transform : boolean, optional, default: True
            An indicator to standardize the parameters.

        Returns:
        --------
        input_dict : dict
            The simulation dictionary configured for ``bayesflow.amortizers.Amortizer``
        """

        # Extract relevant simulation data, convert to float32, and add extra dimension
        theta = raw_dict.get("prior_draws").astype(np.float32)
        rt = raw_dict.get("sim_data").astype(np.float32)[..., None]

        if transform:
            # get prior means and stds for scaling
            prior_means, prior_stds = self.prior.estimate_means_and_stds(n_draws=10000)
            prior_means = np.round(prior_means, decimals=1)
            prior_stds = np.round(prior_stds, decimals=1)

            out_dict = dict(
                parameters=(theta - prior_means) / prior_stds,
                summary_conditions=rt,
            )
        else:
            out_dict = dict(
                parameters=theta,
                summary_conditions=rt,
            )

        return out_dict


class StationaryDiffusion(DiffusionModel):
    """A wrapper for a Stationary Diffusion Decision process with
    random variability."""

    def __init__(self):
        """Creates an instance of a Stationary Diffusion Decision Model with given configuration.
        When used in a BayesFlow pipeline, only the attribute ``self.generator`` and
        the method ``self.configure`` should be used.
        """

        # Create prior wrapper
        self.prior = bf.simulation.Prior(
            prior_fun=sample_variability
        )

        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_stationary_diffusion_process,
        )

        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.GenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="stationary_diffusion_model",
        )

    def generate(self, batch_size, *args, **kwargs):
        """Wraps the call function of ``bf.simulation.GenerativeModel``.

        Parameters:
        -----------
        batch_size : int
            The number of simulations to generate per training batch
        **kwargs   : dict, optional, default: {}
            Optional keyword arguments passed to the call function of ``bf.simulation.GenerativeModel``

        Returns:
        --------
        raw_dict   : dict
            The simulation dictionary configured for ``bayesflow.amortizers.Amortizer``
        """

        return self.generator(batch_size, *args, **kwargs)

    def configure(self, raw_dict):
        """Configures the output of self.generator for a BayesFlow pipeline.

        1. Converts float64 to float32 (for TensorFlow)
        2. Appends a trailing dimensions of 1 to data

        Parameters:
        -----------
        raw_dict  : dict
            A simulation dictionary as returned by ``bayesflow.simulation.GenerativeModel``

        Returns:
        --------
        input_dict : dict
            The simulation dictionary configured for ``bayesflow.amortizers.Amortizer``
        """

        # Extract relevant simulation data, convert to float32, and add extra dimension
        rt = raw_dict.get("sim_data").astype(np.float32)[..., None]

        out_dict = dict(
            summary_conditions=rt,
        )

        return out_dict


class RandomWalkDiffusion(DiffusionModel):
    """A wrapper for a Non-Stationary Diffusion Decision process with
    a Gaussian random walk transition model."""

    def __init__(self, num_steps=1320, rng=None):
        """Creates an instance of the Non-Stationary Diffusion Decision model with given configuration.
        When used in a BayesFlow pipeline, only the attribute ``self.generator`` and
        the method ``self.configure`` should be used.

        Parameters:
        -----------
        rng : np.random.Generator or None, default: None
            An optional random number generator to use, if fixing the seed locally.
        """

        self.hyper_prior_mean = beta(a=1, b=25).mean()
        self.hyper_prior_std = beta(a=1, b=25).std()
        self.local_prior_means = np.array([1.75, 1.7, 1])
        self.local_prior_stds = np.array([1.5, 1.25, 1])

        # Store local RNG instance
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        # Create prior wrapper
        self.prior = bf.simulation.TwoLevelPrior(
            hyper_prior_fun=sample_scale,
            local_prior_fun=partial(sample_random_walk, num_steps=num_steps, rng=self._rng),
        )

        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_random_walk_diffusion_process,
        )

        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.TwoLevelGenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="random_walk_diffusion_model",
        )

    def generate(self, batch_size, *args, **kwargs):
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

    def configure(self, raw_dict, transform=True):
        """Configures the output of self.generator for a BayesFlow pipeline.

        1. Converts float64 to float32 (for TensorFlow)
        2. Appends a trailing dimensions of 1 to data
        3. Scale the model parameters if tranform=True

        Parameters:
        -----------
        raw_dict  : dict
            A simulation dictionary as returned by ``bayesflow.simulation.TwoLevelGenerativeModel``
        transform : boolean, optional, default: True
            An indicator to standardize the parameter and log-transform the data samples. 

        Returns:
        --------
        input_dict : dict
            The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
        """

        # Extract relevant simulation data, convert to float32, and add extra dimension
        theta_t = raw_dict.get("local_prior_draws").astype(np.float32)
        scales = raw_dict.get("hyper_prior_draws").astype(np.float32)
        rt = raw_dict.get("sim_data").astype(np.float32)[..., None]

        if transform:
            out_dict = dict(
                local_parameters=(theta_t - self.local_prior_means) / self.local_prior_stds,
                hyper_parameters=(scales - self.hyper_prior_mean) / self.hyper_prior_std,
                summary_conditions=rt,
            )
        else:
            out_dict = dict(
                local_parameters=theta_t,
                hyper_parameters=scales,
                summary_conditions=rt
            )

        return out_dict


class RegimeSwitchingDiffusion(DiffusionModel):
    """A wrapper for a Regime Switching Diffusion Decision process with
    sudden uniformly distributed jumps"""

    def __init__(self, rng=None):
        """Creates an instance of the Regime Switching Diffusion Decision model with given configuration.
        When used in a BayesFlow pipeline, only the attribute ``self.generator`` and
        the method ``self.configure`` should be used.

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
        self.prior = bf.simulation.Prior(
            prior_fun=partial(sample_regime_switching, rng=self._rng),
        )

        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_regime_switching_diffusion_process,
        )

        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.GenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="regime_switching_diffusion_model",
        )

    def generate(self, batch_size, *args, **kwargs):
        """Wraps the call function of ``bf.simulation.GenerativeModel``.

        Parameters:
        -----------
        batch_size : int
            The number of simulations to generate per training batch
        **kwargs   : dict, optional, default: {}
            Optional keyword arguments passed to the call function of ``bf.simulation.GenerativeModel``

        Returns:
        --------
        raw_dict   : dict
            The simulation dictionary configured for ``bayesflow.amortizers.Amortizer``
        """

        return self.generator(batch_size, *args, **kwargs)

    def configure(self, raw_dict):
        """Configures the output of self.generator for a BayesFlow pipeline.

        1. Converts float64 to float32 (for TensorFlow)
        2. Appends a trailing dimensions of 1 to data

        Parameters:
        -----------
        raw_dict  : dict
            A simulation dictionary as returned by ``bayesflow.simulation.GenerativeModel``

        Returns:
        --------
        input_dict : dict
            The simulation dictionary configured for ``bayesflow.amortizers.Amortizer``
        """

        # Extract relevant simulation data, convert to float32, and add extra dimension
        rt = raw_dict.get("sim_data").astype(np.float32)[..., None]

        out_dict = dict(
            summary_conditions=rt,
        )

        return out_dict