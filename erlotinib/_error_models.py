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
        return self._parameter_names

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
    output by

    .. math::
        X(t, \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
        X^{\text{m}} + \left( \sigma _{\text{base}} + \sigma _{\text{rel}}
        X^{\text{m}}\right) \, \epsilon ,

    where :math:`X^{\text{m}} := X^{\text{m}}(t, \psi )` is the mechanistic
    model output with parameters :math:`\psi`, and :math:`\epsilon` is a
    i.i.d. standard Gaussian random variable

    .. math::
        \epsilon \sim \mathcal{N}(0, 1).

    As a result, this model assumes that the observed biomarker values
    :math:`X^{\text{obs}}` are realisations of the random variable
    :math:`X`.

    The distribution of the observable biomarkers can then be expressed in
    terms of a Gaussian distirbution

    .. math::
        p(X | \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
        \frac{1}{\sqrt{2\pi} \sigma _{\text{tot}}}
        \exp{\left(-\frac{\left(X-X^{\text{m}}\right) ^2}
        {2\sigma^2 _{\text{tot}}} \right)},

    where :math:`\sigma _{\text{tot}} = \sigma _{\text{base}} +
    \sigma _{\text{rel}}X^{\text{m}}`.

    Extends :class:`ErrorModel`.
    """

    def __init__(self):
        super(ConstantAndMultiplicativeGaussianErrorModel, self).__init__()

        # Set defaults
        self._parameter_names = ['Sigma base', 'Sigma rel.']
        self._n_parameters = 2

    def compute_log_likelihood(self, parameters, model_output, observations):
        r"""
        Returns the unnormalised log-likelihood score for the model parameters
        of the mechanistic model-error model pair.

        In this method, the model output :math:`X^{\text{m}}` and the
        observations :math:`X^{\text{obs}}` are compared pair-wise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma _{\text{base}}, \sigma _{\text{rel}} |
            X^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(X^{\text{obs}} _i |
            \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_ouput`` and ``observations`` are already
        ordered, such that the first entries are correspond to the same
        time, the second entries correspond to the same time, etc.

        .. note::
            All constant terms that do not depend on the model parameters are
            dropped when computing the log-likelihood score.

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
        model_output = np.asarray(model_output)
        observations = np.asarray(observations)
        n_observatinos = len(observations)
        if len(model_output) != n_observatinos:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pair-wise.')

        # Get parameters
        sigma_base, sigma_rel = parameters

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * model_output

        # Compute log-likelihood
        log_likelihood = - np.sum(np.log(sigma_tot)) \
            - np.sum((model_output - observations)**2 / sigma_tot**2) / 2

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
