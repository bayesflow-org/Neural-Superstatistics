from abc import ABC, abstractmethod
import bayesflow as bf
import tensorflow as tf

from configuration import default_settings

class Experiment(ABC):
    """An interface for running a standardized simulated experiment."""

    @abstractmethod
    def __init__(self, model, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass


class StaticDiffusionExperiment(Experiment):
    pass


class StationaryDiffusionExperiment(Experiment):
    pass


class RandomWalkDiffusionExperiment(Experiment):
    """Wrapper for estimating the Non-Stationary Diffusion Decision Model with 
    a Gaussian random walk transition model neural superstatistics method."""

    def __init__(self, model, config=default_settings):
        """Creates an instance of the model with given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure`` should be used.

        Parameters:
        -----------
        model   : an instance of models.RandomWalkDiffusion
            The model wrapper, should include a callable attribute ``generator`` and a method
            ``configure()``
        # TODO:
        # config  : dict, optional, default: ``configuration.default_settings``
        #     A configuration dictionary with the following keys:
        #     ``lstm1_hidden_units``        - The dimensions of the first LSTM of the first summary net
        #     ``lstm2_hidden_units``        - The dimensions of the second LSTM of the first summary net
        #     ``transformer_hidden_units``  - The dimensions of the transformer of the second summary net
        #     ``trainer``                   - The settings for the ``bf.trainers.Trainer``, not icnluding
        #         the ``amortizer``, ``generative_model``, and ``configurator`` keys,
        #         as these will be provided internaly by the Experiment instance
        # """

        self.model = model

        # Two-level summary network -> reduce 3D into 3D and 2D
        # for local and global amortizer, respectively
        self.summary_network = bf.networks.HierarchicalNetwork(
            [
                tf.keras.Sequential(
                    [
                        tf.keras.layers.LSTM(
                            config["lstm1_hidden_units"], return_sequences=True
                            ),
                        tf.keras.layers.LSTM(
                            config["lstm2_hidden_units"], return_sequences=True
                        ),
                    ]
                ),
                bf.networks.TimeSeriesTransformer(
                    config["transformer_hidden_units"], template_dim=128, summary_dim=32
                    ),
            ]
        )

        self.local_net = bf.amortizers.AmortizedPosterior(
            bf.networks.InvertibleNetwork(num_params=3,
                                          num_coupling_layers=8,
                                          coupling_design='interleaved'
                                          ))

        self.global_net = bf.amortizers.AmortizedPosterior(
            bf.networks.InvertibleNetwork(num_params=3,
                                          num_coupling_layers=6,
                                          coupling_design='interleaved'
                                          ))

        self.amortizer = bf.amortizers.TwoLevelAmortizedPosterior(
            self.local_net, 
            self.global_net, 
            self.summary_network
            )

        # Trainer setup
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=self.model.generate,
            configurator=self.model.configure,
            **config.get("trainer")
        )

    def run(self, epochs=50, iterations_per_epoch=1000, batch_size=32, **kwargs):
        """Proxy for online training."""

        history = self.trainer.train_online(epochs, iterations_per_epoch, batch_size, **kwargs)
        return history

    def evaluate(self, *args, **kwargs):
        pass


class RegimeSwitchingDiffusionExperiment(Experiment):
    pass

