import tensorflow as tf
import bayesflow as bf
import bayesloop as bl
import numpy as np
import sympy
import os

from custom_network import OneDimensionalAmortizer
from configuration import default_settings


class NeuralCoalMiningExperiment:
    """Wrapper for estimating the dynamic model of coal mining disasters using the neural method."""

    def __init__(self, model, config=default_settings):
        """Creates an instance of the model with given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure`` should be used.

        Parameters:
        -----------
        model   : an instance of models.RandomWalkPoissonModel
            The model wrapper, should include a callable attribute ``generator`` and a method
            ``configure()``
        config  : dict, optional, default: ``configuration.default_settings``
            A configuration dictionary with the following keys:
            ``lstm1_hidden_units``  - The dimensions of the first summary net
            ``lstm2_hidden_units``  - The dimensions of the second summary net
            ``hidden_units_local``  - The width of the hidden layer of the local network
            ``hidden_units_global`` - The width of the hidden layer of the global network
            ``trainer`` - The settings for the ``bf.trainers.Trainer``, not icnluding
                the ``amortizer``, ``generative_model``, and ``configurator`` keys,
                as these will be provided internaly by the Experiment instance
        """

        self.model = model

        # Two-level summary network -> reduce 3D into 3D and 2D
        # for local and global amortizer, respectively
        self.summary_network = bf.networks.HierarchicalNetwork(
            [
                tf.keras.Sequential(
                    [
                        tf.keras.layers.GRU(
                            config["gru_hidden_units"], return_sequences=True
                        ),
                        tf.keras.layers.LSTM(
                            config["lstm1_hidden_units"], return_sequences=True
                        )
                    ]
                ),
                tf.keras.Sequential(
                    [tf.keras.layers.LSTM(config["lstm2_hidden_units"])]
                ),
            ]
        )

        # Custom amortizer for one-dimensional inference
        self.amortizer = OneDimensionalAmortizer(
            hidden_units_local=config["hidden_units_local"],
            hidden_units_global=config["hidden_units_global"],
            summary_net=self.summary_network,
        )

        # Trainer setup
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=self.model.generate,
            configurator=self.model.configure,
            **config.get("trainer")
        )

    def run(self, epochs=10, iterations_per_epoch=1000, batch_size=32, **kwargs):
        """Proxy for online training."""

        history = self.trainer.train_online(epochs, iterations_per_epoch, batch_size, **kwargs)
        return history


class BayesLoopCoalMiningExperiment:
    """Wrapper for estimating the dynamic model of coal mining disasters using the benchmark BayesLoop method.
    
    BayesLoop is described in detail in the following paper: https://www.nature.com/articles/s41467-018-04241-5
    """

    def __init__(self, grid_length=4000):
        """Creates an instance of the dynamic coal mining model to be estimated with the
        bayesloop software: http://bayesloop.com/

        Parameters:
        -----------
        grid_length  : int, optional, default: 4000
            The length of the approximation grid used by BayesLoop
        """
        
        # Set up hyperparameters
        self.grid_length = grid_length
        self.study = bl.HyperStudy()
        
        # Set up likelihood model
        self.likelihood = bl.observationModels.Poisson(
            "accident_rate",
            bl.oint(0, 15, grid_length),
            prior=sympy.stats.Exponential("expon", 1.0),
        )
        
        # Set up transition model
        self.transition = bl.transitionModels.GaussianRandomWalk(
            "sigma",
            bl.oint(0, 1, grid_length),
            target="accident_rate",
            prior=sympy.stats.Beta("beta", 1, 25),
        )
        self.study.set(self.likelihood)
        self.study.set(self.transition)
        
    def load_results(self, path="../posterior_samples"):
        """Loads posterior means and standard deviations from a previous run.

        Parameters:
        -----------
        path : str or None, optional: default: "../posterior_samples"
            Relative path to a directory to read/write posterior sample means and stds
            
        Returns:
        --------
        post_means : np.array of shape (num_steps, )
            An array of posterior means per time point
        post_stds  : np.array of shape (num_steps, )
            An array of posterior std. deviations per time point
        """

        if os.listdir(path) != []:
            post_means = np.load(os.path.join(path, f"bl_post_means.npy"))
            post_stds = np.load(os.path.join(path, f"bl_post_means.npy"))
            return post_means, post_stds
        else:
            raise IOError('No results found! You may need to run the experiment first by calling the .run() method.')

    def run(self, data, filtering=True, path="../posterior_samples"):
        """Estimates time-varying parameters via grid approximation as implemented in BayesLoop.

        Parameters:
        -----------
        data      : dict with keys "disasters" and "year"
            Year-by-year coal mining accident data
        filtering : boolean, optional, default: True
            A flag to tell BayesLoop to estimate either the
            "filtering" (``filtering=True``) or the "smoothing" (``filtering=False``)
            distribution. As described by BayesLoop, the filtering distribution incorporates 
            only the information of past data points, as in an online (real-time) analysis.
        path : str or None, optional: default: "../posterior_samples"
            Relative path to a directory where the posterior means and standard deviations
            of the analysis will be saved. If None, the results will not be written to disk, 
            so the caller is responsible for saving any results.

        Returns:
        --------
        post_means : np.array of shape (num_steps, )
            An array of posterior means per time point
        post_stds  : np.array of shape (num_steps, )
            An array of posterior std. deviations per time point
        """

        # Initialize and fit
        self.study.load(data["disasters"].astype(np.int32), timestamps=data["year"].astype(np.int32))
        self.study.fit(forwardOnly=filtering)

        # Obtain posterior means and stds
        num_steps = data['year'].shape[0]
        post_densities = np.zeros((num_steps, self.grid_length))
        for t in range(num_steps):
            post_densities[t] = self.study.getParameterDistribution(data["year"][t], "accident_rate")[1]
        post_means = self.study.getParameterMeanValues("accident_rate")
        post_grid = self.study.getParameterDistribution(1852, "accident_rate")[0]
        post_stds = np.zeros(num_steps)
        for i in range(num_steps):
            center_grid = (post_grid - post_means[i])**2
            post_stds[i] = np.sqrt(np.sum(post_densities[i] * center_grid) / np.sum(post_densities[i]))
        
        if path is not None:
            np.save(os.path.join(path, "bl_post_means.npy"), post_means)
            np.save(os.path.join(path, "bl_post_stds.npy"), post_stds)
        return post_means, post_stds