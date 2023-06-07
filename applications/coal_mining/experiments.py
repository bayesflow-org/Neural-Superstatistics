from abc import ABC, abstractmethod

import tensorflow as tf
import bayesflow as bf

from custom_network import OneDimensionalAmortizer
from configuration import default_settings


class CoalMiningExperiment:

    def __init__(self, model, config=default_settings):
        """Creates an instance of the model with given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure`` should be used.

        Parameters:
        -----------
        model   : an instance of models.ToyModel
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
        self.summary_network = bf.networks.HierarchicalNetwork([
            tf.keras.Sequential([tf.keras.layers.LSTM(config['lstm1_hidden_units'], return_sequences=True)]),
            tf.keras.Sequential([tf.keras.layers.LSTM(config['lstm2_hidden_units'])])
        ])

        # Custom amortizer for one-dimensional inference
        self.amortizer = OneDimensionalAmortizer(
            hidden_units_local=config['hidden_units_local'],
            hidden_units_global=config['hidden_units_global'],
            summary_net=self.summary_network,
        )

        # Trainer setup
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=self.model.generate,
            configurator=self.model.configure,
            **config.get("trainer")
        )

    def run(self, epochs=10, iterations_per_epoch=1000, batch_size=32):
        """Proxy for online training."""

        history = self.trainer.train_online(epochs, iterations_per_epoch, batch_size)
        return history