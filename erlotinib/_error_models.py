#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np


class ErrorModel(object):
    """
    A base class for error models for the one-dimensional output of
    :class:`MechanisticModel` instances.
    """

    def __init__(self):
        super(ErrorModel, self).__init__()

        # Set defaults
        self._parameter_names = None
        self._n_parameters = None

    def compute_log_likelihood(self, parameters, model_output, observations):
        """
        Returns the unnormalised log-likelihood score for the model parameters
        of the mechanistic model-error model pair.

        In this method, the model output and the observations are compared
        pair-wise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_ouput`` and ``observations`` are
        already ordered, such that the first entries are correspond to the same
        time, the second entries correspond to the same time, etc.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        """
        raise NotImplementedError

    def get_parameter_names(self):
        """
        Returns the names of the error model parameters.
        """
        self._parameter_names

    def n_parameters(self):
        """
        Returns the number of parameters of the error model.
        """
        return self._n_parameters

    def sample(self, parameters, model_output, n_samples=None, seed=None):
        """
        Returns a samples from the mechanistic model-error model pair in form
        of a NumPy array of shape ``(len(model_output), n_samples)``.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        n_samples
            Number of samples from the error model for each entry in
            ``model_output``. If ``None``, one sample is assumed.
        seed
            Seed for the pseudo-random number generator. If ``None``, the
            pseudo-random number generator is not seeded.
        """
        raise NotImplementedError

    def set_parameter_names(self, names):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`.
        """
        raise NotImplementedError


class ConstantAndMultiplicativeGaussianErrorModel(ErrorModel):
    r"""
    An error model that assumes that the model error is a mixture between a
    Gaussian base-level noise and a Gaussian heteroscedastic noise.

    A ConstantAndMultiplicativeGaussianErrorModel assumes that the observable
    biomarker :math:`X` is related to the :class:`MechanisticModel` biomarker
    output :math:`X^{\text{m}}` by

    .. math::
        X(t, \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
        X^{\text{m}}(t, \psi ) + (\sigma _{\text{base}} + \sigma _{\text{rel}}
        X^{\text{m}}(t, \psi ) \, \epsilon ,

    where :math:`\epsilon` is a i.i.d. standard Gaussian random variable

    .. math::
        \epsilon \sim \mathcal{N}(0, 1).

    As a result, this model assumes that the observed biomarker values
    :math:`X^{\text{obs}}` are realisations of the random variable
    :math:`X`.

    Extends :class:`ErrorModel`.
    """

    def __init__(self, problem):
        super(ConstantAndMultiplicativeGaussianErrorModel, self).__init__()

        # Set defaults
        self._parameter_names = ['Sigma base', 'Sigma rel.']
        self._n_parameters = 2

    def compute_log_likelihood(self, parameters):
        """
        Returns the unnormalised log-likelihood score for the model parameters
        of the mechanistic model-error model pair.

        In this method, the model output and the observations are compared
        pair-wise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_ouput`` and ``observations`` are
        already ordered, such that the first entries are correspond to the same
        time, the second entries correspond to the same time, etc.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        observations
            An array-like object with the observations of a biomarker.
        """
        #TODO: Refactor properly
        # Get parameters from input
        noise_parameters = np.asarray(parameters[-self._np:])
        sigma_base = noise_parameters[:self._no]
        eta = noise_parameters[self._no:2 * self._no]
        sigma_rel = noise_parameters[2 * self._no:]

        # Evaluate noise-free model (n_times, n_outputs)
        function_values = self._problem.evaluate(parameters[:-self._np])

        # Compute error (n_times, n_outputs)
        error = self._values - function_values

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * function_values**eta

        # Compute log-likelihood
        # (inner sums over time points, outer sum over parameters)
        log_likelihood = self._logn - np.sum(
            np.sum(np.log(sigma_tot), axis=0)
            + 0.5 * np.sum(error**2 / sigma_tot**2, axis=0))

        return log_likelihood

    def sample(self, parameters, model_output, n_samples=None, seed=None):
        """
        Returns samples from the mechanistic model-error model pair in form
        of a NumPy array of shape ``(len(model_output), n_samples)``.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`. Each entry is a prediction of the
            mechanistic model for an observed time point in ``observations``.
        n_samples
            Number of samples from the error model for each entry in
            ``model_output``. If ``None``, one sample is assumed.
        seed
            Seed for the pseudo-random number generator. If ``None``, the
            pseudo-random number generator is not seeded.
        """
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of model parameters.')

        # Get number of predicted time points
        model_output = np.asarray(model_output)
        n_times = len(model_output)

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (n_times, int(n_samples))

        # Get parameters
        sigma_base, sigma_rel = parameters

        # Sample from Gaussian distributions
        rng = np.random.default_rng(seed=seed)
        base_samples = rng.normal(mean=0, sigma=sigma_base, size=sample_shape)
        rel_samples = rng.normal(mean=0, sigma=sigma_rel, size=sample_shape)

        # Construct final samples
        samples = base_samples + model_output * rel_samples

        return samples

    def set_parameter_names(self, names):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`.
        """
        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]
