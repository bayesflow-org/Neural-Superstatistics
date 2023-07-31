import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, GRU
from tensorflow.keras.models import Sequential
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers


class DynamicGaussianNetworkFactorized(tf.keras.Model):
    """TODO: Logic"""

    def __init__(self, meta):
        super(DynamicGaussianNetworkFactorized, self).__init__()

        self.embedding_net = Sequential([
            GRU(meta['embedding_gru_units'], return_sequences=True),
            LSTM(meta['embedding_lstm_units'], return_sequences=True),
            Dense(**meta['dense_pre_args']),
        ])

        self.micro_part = Sequential([
            Dense(**meta['dense_micro_args']),
            Dense(tfpl.MultivariateNormalTriL.params_size(meta['n_micro_params'])),
            tfpl.MultivariateNormalTriL(meta['n_micro_params'])
        ])

        self.macro_part = Sequential([
            LSTM(meta['macro_lstm_units']),
            Dense(tfpl.IndependentNormal.params_size(meta['n_macro_params'])),
            tfpl.IndependentNormal(meta['n_macro_params'])
        ])

    def call(self, data, micro_params):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)
        micro_params : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, n_params)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict dynamic (microscopic) params
        micro_params_hat = self.micro_part(rep)

        # Predict static (macroscopic) params
        macro_params_hat = self.macro_part(tf.concat([rep, micro_params], axis=-1))

        return macro_params_hat, micro_params_hat

    def sample(self, data):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Predict dynamic (microscopic) params
        rep = self.embedding_net(data)
        micro_params_hat = self.micro_part(rep)

        # Predict static (macroscopic) params
        micro_params_rv = micro_params_hat.sample()
        macro_params_hat = self.macro_part(
            tf.concat([rep, micro_params_rv], axis=-1)
        )
        macro_params_rv = macro_params_hat.sample()

        return macro_params_rv, micro_params_rv


    def sample_n(self, data, n_samples=50):
        """ TODO

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict dynamic (microscopic) params
        micro_params_hat = self.micro_part(rep)

        # Prepare placeholders and sample from both distros
        macro_samples = [None] * n_samples
        micro_samples = [None] * n_samples
        for n in range(n_samples):
            micro_params_rv = micro_params_hat.sample()
            macro_params_hat = self.macro_part(
                tf.concat([rep, micro_params_rv], axis=-1)
            )
            macro_params_rv = macro_params_hat.sample()
            macro_samples[n] = macro_params_rv
            micro_samples[n] = micro_params_rv
        return tf.stack(macro_samples), tf.stack(micro_samples)


class DynamicGaussianNetworkJoint(tf.keras.Model):
    """TODO: Logic"""

    def __init__(self, meta):
        super(DynamicGaussianNetworkJoint, self).__init__()

        self.embedding_net = Sequential([
            GRU(meta['embedding_gru_units'], return_sequences=True),
            LSTM(meta['embedding_lstm_units'], return_sequences=True),
            Dense(**meta['embedding_dense_args']),
        ])

        self.posterior = Sequential([
            Dense(**meta['posterior_dense_args']),
            Dense(tfpl.IndependentNormal.params_size(meta['n_macro_params'] + meta['n_micro_params'])),
            tfpl.IndependentNormal(meta['n_macro_params'] + meta['n_micro_params'])
        ])

        self.n_micro = meta['n_micro_params']
        self.n_macro = meta['n_macro_params']


    def call(self, data):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict all params
        params_hat = self.posterior(rep)

        return params_hat

    def sample_n(self, data, n_samples=50):
        """ TODO

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict all params
        params_hat = self.posterior(rep)

        # Sample and split macro and micro
        samples = params_hat.sample(n_samples)
        macro_samples = samples[:, :, :, :self.n_macro]
        micro_samples = samples[:, :, :, self.n_macro:]
        return macro_samples, micro_samples


class DynamicGaussianNetworkOld(tf.keras.Model):

    def __init__(self, meta):
        super(DynamicGaussianNetworkOld, self).__init__()

        self.preprocessor = Sequential([
            GRU(meta['embedding_gru_units'], return_sequences=True), # 64
            LSTM(meta['embedding_lstm_units'], return_sequences=True), # 128
            Dense(**meta['embedding_dense_args']), # 128
        ])

        self.dynamic_predictor = Sequential([
            Dense(**meta['dense_micro_args']), # 64
            tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(meta['n_micro_params'])),
            tfpl.MultivariateNormalTriL(meta['n_micro_params'])
        ])

        self.static_predictor = Sequential([
            LSTM(meta['macro_lstm_units']), # n_params_s
            Dense(tfpl.MultivariateNormalTriL.params_size(meta['n_macro_params'])),
            tfpl.MultivariateNormalTriL(meta['n_macro_params'])
        ])

    def call(self, x):
        """
        Forward pass through the model.
        ----------
        Input:
        np.array of shape (batchsize, n_obs, 5)
        ----------
        Output:
        tf.tensor distribution of shape (batchsize, n_obs, n_params_d)
        tf.tensor distribution of shape (batchsize, n_params_s)
        """

        # obtain representation
        rep = self.preprocessor(x)

        # predict dynamic microscopic params
        preds_dyn = self.dynamic_predictor(rep)

        # predict static macroscopic params
        preds_stat = self.static_predictor(rep)

        return preds_stat, preds_dyn


class DynamicGaussianNetwork(tf.keras.Model):
    """TODO: Logic"""

    def __init__(self, meta):
        super(DynamicGaussianNetwork, self).__init__()

        self.embedding_net = Sequential([
            GRU(meta['embedding_gru_units'], return_sequences=True),
            LSTM(meta['embedding_lstm_units'], return_sequences=True),
            Dense(**meta['dense_pre_args']),
        ])

        self.micro_part = Sequential([
            Dense(**meta['dense_micro_args']),
            Dense(tfpl.MultivariateNormalTriL.params_size(meta['n_micro_params'])),
            tfpl.MultivariateNormalTriL(meta['n_micro_params'])
        ])

        self.macro_part = Sequential([
            LSTM(meta['macro_lstm_units']),
            Dense(tfpl.IndependentNormal.params_size(meta['n_macro_params'])),
            tfpl.IndependentNormal(meta['n_macro_params'])
        ])

    def call(self, data, macro_params):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)
        macro_params : tf.Tensor of np.ndarray of shape (batch_size, n_macro_params)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)

        # Predict dynamic microscopic params
        micro_params_hat = self.micro_part(
            tf.concat([rep, tf.stack([macro_params] * T, axis=1)], axis=-1)
        )

        return macro_params_hat, micro_params_hat


    def sample(self, data):
        """ Performs a forward pass through the network and obtains the posterior(s) over all time points.

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)
        macro_params_rv = macro_params_hat.sample()

        # Predict dynamic microscopic params
        micro_params_hat = self.micro_part(
            tf.concat([rep, tf.stack([macro_params_rv] * T, axis=1)], axis=-1)
        )
        micro_params_rv = micro_params_hat.sample()

        return macro_params_rv, micro_params_rv


    def sample_n(self, data, n_samples=50):
        """ TODO

        Parameters
        ----------
        data         : tf.Tensor of np.ndarray of shape (batch_size, n_time_points, data_dim)

        Returns
        -------
        TODO
        """

        # Extract n time points
        T = data.shape[1]

        # Obtain representation of time-series
        rep = self.embedding_net(data)

        # Predict static macroscopic params
        macro_params_hat = self.macro_part(rep)

        # Prepare placeholders
        macro_samples = [None] * n_samples
        micro_samples = [None] * n_samples
        for n in range(n_samples):

            macro_params_rv = macro_params_hat.sample()
            micro_params_hat = self.micro_part(
                tf.concat([rep, tf.stack([macro_params_rv] * T, axis=1)], axis=-1)
            )
            micro_params_rv = micro_params_hat.sample()
            macro_samples[n] = macro_params_rv
            micro_samples[n] = micro_params_rv
        return tf.stack(macro_samples), tf.stack(micro_samples)