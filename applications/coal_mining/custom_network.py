import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


class OneDimensionalAmortizer(tf.keras.Model):
    """This network is only usable for a 1D dynamic estimation problem and used
    for the Poisson toy example."""

    def __init__(
        self, hidden_units_local=128, hidden_units_global=128, summary_net=None
    ):
        """Creates an instance of the custom amortized dynamic posterior for one-dimensional problems."""
        super().__init__()

        self.local_amortizer = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_units_local, activation="relu"),
                tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(1)),
                tfpl.IndependentNormal(1),
            ]
        )

        self.global_amortizer = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_units_global, activation="relu"),
                tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(1)),
                tfpl.IndependentNormal(1),
            ]
        )

        if summary_net is None:
            raise NotImplementedError(
                "You need a summary netowork for this toy example!"
            )
        self.summary_net = summary_net

    def call(self, input_dict, **kwargs):
        """Performs a forward pass through the summary and amortizer networks.

        Parameters
        ----------
        input_dict  : dict
            The simulation dictionary, as provided by a PoissonRandomWalk model.
        """

        return self._get_network_outputs(input_dict, **kwargs)

    def sample(self, obs_data, num_samples):
        """A helper function to sample from the dynamic amortized posterior. Deviates from
        all other BayesFlow amortizers in that it takes directly the array input and not a dictionary.

        Parameters
        ----------
        obs_data    : np.ndarray of shape (1, num_steps, 1)
            The observed time series data.
        num_samples : int
            The number of samples to obtain from the (factorized) joint posterior
        """

        # Summarize data
        local_summaries, global_summaries = self.summary_net(obs_data, return_all=True)
        local_summaries = tf.concat([local_summaries] * num_samples, axis=0)

        # Obtain global parameter samples
        global_samples = self.global_amortizer(global_summaries).sample(num_samples)

        # Prepare condition for local: data + global
        _hypers = tf.tile(global_samples, [1, obs_data.shape[1], 1])
        local_summaries = tf.concat([local_summaries, _hypers], axis=-1)

        # Obtain local parameter samples
        local_samples = self.local_amortizer(local_summaries).sample()

        return {
            "global_samples": tf.squeeze(global_samples).numpy(),
            "local_samples": tf.squeeze(local_samples).numpy(),
        }

    def compute_loss(self, input_dict, **kwargs):
        """Compute loss of local and global networks.

        Parameters
        ----------
        input_dict  : dict
            The simulation dictionary, as provided by a PoissonRandomWalk model.
        """

        local_dist, global_dist = self._get_network_outputs(input_dict, **kwargs)
        nll_local = tf.reduce_mean(-local_dist.log_prob(input_dict["local_parameters"]))
        nll_global = tf.reduce_mean(
            -global_dist.log_prob(input_dict["hyper_parameters"])
        )
        return {"Local.Loss": nll_local, "Global.Loss": nll_global}

    def _get_network_outputs(self, input_dict, **kwargs):
        """Helper function to summarize and process simulations."""

        # Obtain summaries
        local_summaries, global_summaries = self.summary_net(
            input_dict["summary_conditions"], return_all=True, **kwargs
        )
        num_steps = input_dict["summary_conditions"].shape[1]

        # Attach hyperparameters as conditions
        _hypers = tf.expand_dims(input_dict.get("hyper_parameters"), axis=1)
        _hypers = tf.tile(_hypers, [1, num_steps, 1])
        local_summaries = tf.concat([local_summaries, _hypers], axis=-1)

        # Obtain posteriors
        local_dist = self.local_amortizer(local_summaries, **kwargs)
        global_dist = self.global_amortizer(global_summaries, **kwargs)
        return local_dist, global_dist
