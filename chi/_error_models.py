#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy

from numba import njit
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
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}) =
            \sum _i \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}` are the observed
        biomarkers. :math:`\psi` and :math:`\sigma` are the parameters of the
        mechanistic model and the error model, respectively.

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

    def compute_pointwise_ll(self, parameters, model_output, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value. :math:`\psi` and
        :math:`\sigma` are the parameters of the mechanistic model and the
        error model, respectively.

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

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}) =
            \sum _i \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}` are the observed
        biomarkers. :math:`\psi` and :math:`\sigma` are the parameters of the
        mechanistic model and the error model, respectively.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi} \quad \text{and} \quad
            \frac{\partial L}{\partial \sigma},

        where both :math:`\psi` and :math:`\sigma` can be multi-dimensional.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        raise NotImplementedError

    def get_parameter_names(self):
        """
        Returns the names of the error model parameters.
        """
        return copy.copy(self._parameter_names)

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
            :class:`MechanisticModel`.
        n_samples
            Number of samples from the error model for each entry in
            ``model_output``. If ``None``, one sample is assumed.
        seed
            Seed for the pseudo-random number generator. If ``None``, the
            pseudo-random number generator is not seeded.
        """
        raise NotImplementedError

    def set_parameter_names(self, names=None):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        raise NotImplementedError


class ConstantAndMultiplicativeGaussianErrorModel(ErrorModel):
    r"""
    An error model which assumes that the model error is a mixture between a
    Gaussian base-level noise and a Gaussian heteroscedastic noise.

    A mixture between a Gaussian base-level noise and a Gaussian
    heteroscedastic noise assumes that the observable biomarker :math:`X`
    is related to the :class:`MechanisticModel` output by

    .. math::
        X(t, \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
        x^{\text{m}} + \left( \sigma _{\text{base}} + \sigma _{\text{rel}}
        x^{\text{m}}\right) \, \epsilon ,

    where :math:`x^{\text{m}} := x^{\text{m}}(t, \psi )` is the mechanistic
    model output with parameters :math:`\psi`, and :math:`\epsilon` is a
    i.i.d. standard Gaussian random variable

    .. math::
        \epsilon \sim \mathcal{N}(0, 1).

    As a result, this model assumes that the observed biomarker values
    :math:`x^{\text{obs}}` are realisations of the random variable
    :math:`X`.

    At each time point :math:`t` the distribution of the observable biomarkers
    can be expressed in terms of a Gaussian distribution

    .. math::
        p(x | \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
        \frac{1}{\sqrt{2\pi} \sigma _{\text{tot}}}
        \exp{\left(-\frac{\left(x-x^{\text{m}}\right) ^2}
        {2\sigma^2 _{\text{tot}}} \right)},

    where :math:`\sigma _{\text{tot}} = \sigma _{\text{base}} +
    \sigma _{\text{rel}}x^{\text{m}}`.

    Extends :class:`ErrorModel`.
    """

    def __init__(self):
        super(ConstantAndMultiplicativeGaussianErrorModel, self).__init__()

        # Set defaults
        self._parameter_names = ['Sigma base', 'Sigma rel.']
        self._n_parameters = 2

    @staticmethod
    @njit
    def _compute_log_likelihood(
            parameters, model_output, observations):  # pragma: no cover
        """
        Calculates the log-lieklihood using numba speed up.
        """
        # Get parameters
        sigma_base, sigma_rel = parameters

        if sigma_base <= 0 or sigma_rel <= 0:
            # sigma_base and sigma_rel are strictly positive
            return -np.inf

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * model_output

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.sum(np.log(sigma_tot)) \
            - np.sum((model_output - observations)**2 / sigma_tot**2) / 2

        return log_likelihood

    @staticmethod
    @njit
    def _compute_pointwise_ll(
            parameters, model_output, observations):  # pragma: no cover
        """
        Calculates the pointwise log-lieklihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma_base, sigma_rel = parameters

        if sigma_base <= 0 or sigma_rel <= 0:
            # sigma_base and sigma_rel are strictly positive
            n_obs = len(model_output)
            return np.full(n_obs, -np.inf)

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * model_output

        # Compute log-likelihood
        pointwise_ll = \
            - np.log(2 * np.pi) / 2 \
            - np.log(sigma_tot) \
            - (model_output - observations)**2 / sigma_tot**2 / 2

        return pointwise_ll

    @staticmethod
    @njit
    def _compute_sensitivities(
            parameters, model_output, model_sensitivities,
            observations):  # pragma: no cover
        """
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        """
        # Get parameters
        sigma_base, sigma_rel = parameters

        if sigma_base <= 0 or sigma_rel <= 0:
            # sigma_base and sigma_rel are strictly positive
            n_parameters = model_sensitivities.shape[1] + 2
            return -np.inf, np.full(n_parameters, np.inf)

        # Compute total standard deviation
        sigma_tot = sigma_base + sigma_rel * model_output

        # Compute error and squared error
        error = observations - model_output
        squared_error = error**2

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.sum(np.log(sigma_tot)) \
            - np.sum(squared_error / sigma_tot**2) / 2

        # Compute sensitivities
        dpsi = \
            np.sum(
                error / sigma_tot**2 * model_sensitivities, axis=0) \
            - sigma_rel * np.sum(model_sensitivities / sigma_tot, axis=0) \
            + sigma_rel * np.sum(
                squared_error / sigma_tot**3 * model_sensitivities, axis=0)
        dsigma_base = \
            np.sum(squared_error / sigma_tot**3, axis=0) - \
            np.sum(1 / sigma_tot, axis=0)
        dsigma_rel = \
            np.sum(squared_error / sigma_tot**3 * model_output, axis=0) \
            - np.sum(model_output / sigma_tot, axis=0)
        sensitivities = np.concatenate((dpsi, dsigma_base, dsigma_rel))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`x^{\text{m}}` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma _{\text{base}}, \sigma _{\text{rel}} |
            x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

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
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs)

    def compute_pointwise_ll(self, parameters, model_output, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma _{\text{base}}, \sigma _{\text{rel}} |
            x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

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
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs)

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma _{\text{base}}}, \quad
            \frac{\partial L}{\partial \sigma _{\text{rel}}}.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 2
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs)

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
            :class:`MechanisticModel`.
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
        base_samples = rng.normal(loc=0, scale=sigma_base, size=sample_shape)
        rel_samples = rng.normal(loc=0, scale=sigma_rel, size=sample_shape)

        # Construct final samples
        model_output = np.expand_dims(model_output, axis=1)
        samples = model_output + base_samples + model_output * rel_samples

        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Sigma base', 'Sigma rel.']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class GaussianErrorModel(ErrorModel):
    r"""
    An error model which assumes that the model error follows a Gaussian
    distribution.

    A Gaussian error model assumes that the observable
    biomarker :math:`X` is related to the :class:`MechanisticModel`
    output by

    .. math::
        X(t, \psi , \sigma) = x^{\text{m}} + \sigma \epsilon ,

    where :math:`x^{\text{m}} := x^{\text{m}}(t, \psi )` is the mechanistic
    model output with parameters :math:`\psi`, and :math:`\epsilon` is a
    i.i.d. standard Gaussian random variable

    .. math::
        \epsilon \sim \mathcal{N}(0, 1).

    As a result, this model assumes that the observed biomarker values
    :math:`x^{\text{obs}}` are realisations of the random variable
    :math:`X`.

    At each time point :math:`t` the distribution of the observable biomarkers
    can be expressed in terms of a Gaussian distribution

    .. math::
        p(x | \psi , \sigma ) =
        \frac{1}{\sqrt{2\pi} \sigma }
        \exp{\left(-\frac{\left(x-x^{\text{m}}\right) ^2}
        {2\sigma^2 } \right)}.

    Extends :class:`ErrorModel`.
    """

    def __init__(self):
        super(GaussianErrorModel, self).__init__()

        # Set defaults
        self._parameter_names = ['Sigma']
        self._n_parameters = 1

    @staticmethod
    @njit
    def _compute_log_likelihood(
            parameters, model_output, observations):  # pragma: no cover
        """
        Calculates the log-lieklihood using numba speed up.
        """
        # Get parameters
        sigma = parameters[0]

        if sigma <= 0:
            # sigma is strictly positive
            return -np.inf

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - np.sum((model_output - observations)**2) / sigma**2 / 2

        return log_likelihood

    @staticmethod
    @njit
    def _compute_pointwise_ll(
            parameters, model_output, observations):  # pragma: no cover
        """
        Calculates the pointwise log-lieklihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma = parameters[0]

        if sigma <= 0:
            # sigma is strictly positive
            n_obs = len(model_output)
            return np.full(n_obs, -np.inf)

        # Compute log-likelihood
        pointwise_ll = \
            - (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - (model_output - observations)**2 / sigma**2 / 2

        return pointwise_ll

    @staticmethod
    @njit
    def _compute_sensitivities(
            parameters, model_output, model_sensitivities,
            observations):  # pragma: no cover
        """
        Calculates the log-lieklihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        """
        # Get parameters
        sigma = parameters[0]

        if sigma <= 0:
            # sigma is strictly positive
            n_parameters = model_sensitivities.shape[1] + 1
            return -np.inf, np.full(n_parameters, np.inf)

        # Compute error and squared error
        error = observations - model_output
        summed_squared_error = np.sum(error**2, axis=0)

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - summed_squared_error / sigma**2 / 2
        log_likelihood = log_likelihood[0]

        # Compute sensitivities
        dpsi = \
            np.sum(error * model_sensitivities, axis=0) / sigma**2
        dsigma = \
            summed_squared_error / sigma**3 - n_obs / sigma
        sensitivities = np.concatenate((dpsi, dsigma))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`x^{\text{m}}` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma | x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma ) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`, :math:`x^{\text{m}}`. Each entry is a
            prediction of the mechanistic model for an observed time point in
            ``observations``, :math:`x^{\text{obs}}`.
        observations
            An array-like object with the observations of a biomarker
            :math:`x^{\text{obs}}`.
        """
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs)

    def compute_pointwise_ll(self, parameters, model_output, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

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
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs)

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma }.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 1
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs)

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
            :class:`MechanisticModel`.
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
        sigma = parameters[0]

        # Sample from Gaussian distributions
        rng = np.random.default_rng(seed=seed)
        samples = rng.normal(loc=0, scale=sigma, size=sample_shape)

        # Construct final samples
        model_output = np.expand_dims(model_output, axis=1)
        samples = model_output + samples

        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Sigma']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class LogNormalErrorModel(ErrorModel):
    r"""
    An error model which assumes that the model error follows a Log-normal
    distribution.

    A log-normal error model assumes that the observable
    biomarker :math:`X` is related to the :class:`MechanisticModel`
    output by

    .. math::
        X(t, \psi , \sigma _{\mathrm{log}}) =
        y \, \mathrm{e}^{\mu + \sigma _{\mathrm{log}} \varepsilon },

    where :math:`y := y(t, \psi )` is the mechanistic
    model output with parameters :math:`\psi`, and :math:`\varepsilon` is a
    i.i.d. standard Gaussian random variable

    .. math::
        \varepsilon \sim \mathcal{N}(0, 1).

    Here, :math:`\sigma _{\mathrm{log}}` is the standard deviation of
    :math:`\log X` and
    :math:`\mu := -\sigma _{\mathrm{log}} ^2 / 2` is chosen such that

    .. math::
        \mathbb{E}[X] = y.

    As a result, this model assumes that the observed biomarker values
    :math:`x^{\text{obs}}` are realisations of the random variable
    :math:`X`.

    At each time point :math:`t` the distribution of the observable biomarkers
    can be expressed in terms of a log-normal distribution

    .. math::
        p(x | \psi , \sigma _{\mathrm{log}} ) =
        \frac{1}{\sqrt{2\pi} \sigma _{\mathrm{log}} x}
        \exp{\left(-\frac{
            \left(\log x - \log y + \sigma _{\mathrm{log}}^2/2\right) ^2}
        {2\sigma _{\mathrm{log}}^2 } \right)}.

    Extends :class:`ErrorModel`.
    """

    def __init__(self):
        super(LogNormalErrorModel, self).__init__()

        # Set defaults
        self._parameter_names = ['Sigma log']
        self._n_parameters = 1

    @staticmethod
    @njit
    def _compute_log_likelihood(
            parameters, model_output, observations):  # pragma: no cover
        """
        Calculates the log-likelihood using numba speed up.
        """
        # Get parameters
        sigma = parameters[0]

        if (sigma <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            return -np.inf

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - np.sum(np.log(observations)) \
            - np.sum((
                np.log(model_output) - sigma**2 / 2
                - np.log(observations)
            )**2) / sigma**2 / 2

        return log_likelihood

    @staticmethod
    @njit
    def _compute_pointwise_ll(
            parameters, model_output, observations):  # pragma: no cover
        """
        Calculates the pointwise log-lieklihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma = parameters[0]

        if (sigma <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            n_obs = len(model_output)
            return np.full(n_obs, -np.inf)

        # Compute log-likelihood
        pointwise_ll = \
            - (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - np.log(observations) \
            - (
                np.log(model_output) - sigma**2 / 2
                - np.log(observations)
            )**2 / sigma**2 / 2

        return pointwise_ll

    @staticmethod
    @njit
    def _compute_sensitivities(
            parameters, model_output, model_sensitivities,
            observations):  # pragma: no cover
        """
        Calculates the log-lieklihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        """
        # Get parameters
        sigma = parameters[0]

        if (sigma <= 0) or np.any(model_output <= 0):
            # sigma is strictly positive
            n_parameters = model_sensitivities.shape[1] + 1
            return -np.inf, np.full(n_parameters, np.inf)

        # Compute "error" and squared "error"
        # (Analogous to error for Gaussian model, but not really error here)
        error = np.log(observations) - np.log(model_output) + sigma**2 / 2
        summed_squared_error = np.sum(error**2, axis=0)

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * (np.log(2 * np.pi) / 2 + np.log(sigma)) \
            - np.sum(np.log(observations)) \
            - summed_squared_error / sigma**2 / 2
        log_likelihood = log_likelihood[0]

        # Compute sensitivities
        dpsi = \
            np.sum(error / model_output * model_sensitivities, axis=0) \
            / sigma**2
        dsigma = \
            - np.sum(error) / sigma \
            + summed_squared_error / sigma**3 \
            - n_obs / sigma
        sensitivities = np.concatenate((dpsi, dsigma))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`y` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma _{\mathrm{log}} | x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\mathrm{log}} ) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`, :math:`y`. Each entry is a
            prediction of the mechanistic model for an observed time point in
            ``observations``, :math:`x^{\text{obs}}`.
        observations
            An array-like object with the observations of a biomarker
            :math:`x^{\text{obs}}`.
        """
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs)

    def compute_pointwise_ll(self, parameters, model_output, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma _{\mathrm{log}} | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\mathrm{log}} ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

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
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs)

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma _{\mathrm{log}} }.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 1
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs)

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
            :class:`MechanisticModel`.
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
        sigma_log = parameters[0]
        mean_log = -sigma_log**2 / 2

        # Sample from Gaussian distributions
        rng = np.random.default_rng(seed=seed)
        samples = rng.lognormal(
            mean=mean_log, sigma=sigma_log, size=sample_shape)

        # Construct final samples
        model_output = np.expand_dims(model_output, axis=1)
        samples = model_output * samples

        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Sigma log']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class MultiplicativeGaussianErrorModel(ErrorModel):
    r"""
    An error model which assumes that the model error is a Gaussian
    heteroscedastic noise.

    A Gaussian heteroscedastic noise model assumes that the observable
    biomarker :math:`X` is related to the :class:`MechanisticModel`
    output by

    .. math::
        X(t, \psi , \sigma _{\text{rel}}) =
        x^{\text{m}} + \sigma _{\text{rel}} x^{\text{m}} \, \epsilon ,

    where :math:`x^{\text{m}} := x^{\text{m}}(t, \psi )` is the mechanistic
    model output with parameters :math:`\psi`, and :math:`\epsilon` is a
    i.i.d. standard Gaussian random variable

    .. math::
        \epsilon \sim \mathcal{N}(0, 1).

    As a result, this model assumes that the observed biomarker values
    :math:`x^{\text{obs}}` are realisations of the random variable
    :math:`X`.

    At each time point :math:`t` the distribution of the observable biomarkers
    can be expressed in terms of a Gaussian distribution

    .. math::
        p(x | \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) =
        \frac{1}{\sqrt{2\pi} \sigma _{\text{tot}}}
        \exp{\left(-\frac{\left(x-x^{\text{m}}\right) ^2}
        {2\sigma^2 _{\text{tot}}} \right)},

    where :math:`\sigma _{\text{tot}} = \sigma _{\text{rel}}x^{\text{m}}`.

    Extends :class:`ErrorModel`.
    """

    def __init__(self):
        super(MultiplicativeGaussianErrorModel, self).__init__()

        # Set defaults
        self._parameter_names = ['Sigma rel.']
        self._n_parameters = 1

    @staticmethod
    @njit
    def _compute_log_likelihood(
            parameters, model_output, observations):  # pragma: no cover
        """
        Calculates the log-lieklihood using numba speed up.
        """
        # Get parameters
        sigma_rel = parameters[0]

        if sigma_rel <= 0:
            # sigma_rel are strictly positive
            return -np.inf

        # Compute total standard deviation
        sigma_tot = sigma_rel * model_output

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.sum(np.log(sigma_tot)) \
            - np.sum((model_output - observations)**2 / sigma_tot**2) / 2

        return log_likelihood

    @staticmethod
    @njit
    def _compute_pointwise_ll(
            parameters, model_output, observations):  # pragma: no cover
        """
        Calculates the pointwise log-lieklihood using numba speed up.

        Returns a numpy array of shape (n_times,)
        """
        # Get parameters
        sigma_rel = parameters[0]

        if sigma_rel <= 0:
            # sigma_rel are strictly positive
            n_obs = len(model_output)
            return np.full(n_obs, -np.inf)

        # Compute total standard deviation
        sigma_tot = sigma_rel * model_output

        # Compute log-likelihood
        pointwise_ll = \
            - np.log(2 * np.pi) / 2 \
            - np.log(sigma_tot) \
            - (model_output - observations)**2 / sigma_tot**2 / 2

        return pointwise_ll

    @staticmethod
    @njit
    def _compute_sensitivities(
            parameters, model_output, model_sensitivities,
            observations):  # pragma: no cover
        """
        Calculates the log-lieklihood and its sensitivities using numba
        speed up.

        Expects:
        Shape model output =  (n_obs, 1)
        Shape sensitivities = (n_obs, n_parameters)
        Shape observations =  (n_obs, 1)
        """
        # Get parameters
        sigma_rel = parameters[0]

        if sigma_rel <= 0:
            # sigma_rel are strictly positive
            n_parameters = model_sensitivities.shape[1] + 1
            return -np.inf, np.full(n_parameters, np.inf)

        # Compute total standard deviation
        sigma_tot = sigma_rel * model_output

        # Compute error and squared error
        error = observations - model_output
        squared_error = error**2

        # Compute log-likelihood
        n_obs = len(model_output)
        log_likelihood = \
            - n_obs * np.log(2 * np.pi) / 2 \
            - np.sum(np.log(sigma_tot)) \
            - np.sum(squared_error / sigma_tot**2) / 2

        # Compute sensitivities
        dpsi = \
            np.sum(
                error / sigma_tot**2 * model_sensitivities, axis=0) \
            - sigma_rel * np.sum(model_sensitivities / sigma_tot, axis=0) \
            + sigma_rel * np.sum(
                squared_error / sigma_tot**3 * model_sensitivities, axis=0)
        dsigma_rel = \
            np.sum(squared_error / sigma_tot**3 * model_output, axis=0) \
            - np.sum(model_output / sigma_tot, axis=0)
        sensitivities = np.concatenate((dpsi, dsigma_rel))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, model_output, observations):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output :math:`x^{\text{m}}` and the
        observations :math:`x^{\text{obs}}` are compared pairwise, and the
        log-likelihood score is computed according to

        .. math::
            L(\psi , \sigma _{\text{rel}} |
            x^{\text{obs}}) =
            \sum _{i=1}^N
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\text{rel}}) ,

        where :math:`N` is the number of observations.

        The time-dependence of the values is dealt with implicitly, by
        assuming that ``model_output`` and ``observations`` are already
        ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Parameters
        ----------
        parameters
            An array-like object with the error model parameters.
        model_output
            An array-like object with the one-dimensional output of a
            :class:`MechanisticModel`, :math:`x^{\text{m}}`. Each entry is a
            prediction of the mechanistic model for an observed time point in
            ``observations``, :math:`x^{\text{obs}}`.
        observations
            An array-like object with the observations of a biomarker
            :math:`x^{\text{obs}}`.
        """
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        n_obs = len(observations)
        if len(model) != n_obs:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_log_likelihood(parameters, model, obs)

    def compute_pointwise_ll(self, parameters, model_output, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi , \sigma _{\text{base}}, \sigma _{\text{rel}} |
            x^{\text{obs}}_i) =
            \log p(x^{\text{obs}} _i |
            \psi , \sigma _{\text{base}}, \sigma _{\text{rel}}) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value.

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
        parameters = np.asarray(parameters)
        model = np.asarray(model_output)
        obs = np.asarray(observations)
        n_observations = len(observations)
        if len(model) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pairwise.')

        return self._compute_pointwise_ll(parameters, model, obs)

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        The sensitivities of the log-likelihood are defined as the partial
        derivatives of :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi}, \quad
            \frac{\partial L}{\partial \sigma _{\text{rel}}}.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length 1
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        parameters = np.asarray(parameters)
        n_obs = len(observations)
        model = np.asarray(model_output).reshape((n_obs, 1))
        sens = np.asarray(model_sensitivities)
        obs = np.asarray(observations).reshape((n_obs, 1))
        if len(sens) != n_obs:
            raise ValueError(
                'The first dimension of the sensitivities must match the '
                'number of observations.')

        return self._compute_sensitivities(parameters, model, sens, obs)

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
            :class:`MechanisticModel`.
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
        sigma_rel = parameters[0]

        # Sample from Gaussian distribution
        rng = np.random.default_rng(seed=seed)
        rel_samples = rng.normal(loc=0, scale=sigma_rel, size=sample_shape)

        # Construct final samples
        model_output = np.expand_dims(model_output, axis=1)
        samples = model_output + model_output * rel_samples

        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Sigma rel.']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class ReducedErrorModel(object):
    """
    A class that can be used to permanently fix model parameters of an
    :class:`ErrorModel` instance.

    This may be useful to explore simplified versions of a model.

    Parameters
    ----------
    error_model
        An instance of a :class:`ErrorModel`.
    """

    def __init__(self, error_model):
        super(ReducedErrorModel, self).__init__()

        # Check input
        if not isinstance(error_model, ErrorModel):
            raise ValueError(
                'The error model has to be an instance of a '
                'erlotinib.ErrorModel')

        self._error_model = error_model

        # Set defaults
        self._fixed_params_mask = None
        self._fixed_params_values = None
        self._n_parameters = error_model.n_parameters()
        self._parameter_names = error_model.get_parameter_names()

    def compute_log_likelihood(self, parameters, model_output, observations):
        r"""
        Returns the log-likelihood of the model parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}) =
            \sum _i \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}` are the observed
        biomarkers. :math:`\psi` and :math:`\sigma` are the parameters of the
        mechanistic model and the error model, respectively.

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
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        score = self._error_model.compute_log_likelihood(
            parameters, model_output, observations)
        return score

    def compute_pointwise_ll(self, parameters, model_output, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the pointwise log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}_i) =
            \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}_i` is the
        :math:`i^{\text{th}}` observed biomarker value. :math:`\psi` and
        :math:`\sigma` are the parameters of the mechanistic model and the
        error model, respectively.

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
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        pointwise_ll = self._error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        return pointwise_ll

    def compute_sensitivities(
            self, parameters, model_output, model_sensitivities, observations):
        r"""
        Returns the log-likelihood of the model parameters and its
        sensitivities w.r.t. the parameters.

        In this method, the model output and the observations are compared
        pairwise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
        time, the second entries correspond to the same time, and so on.

        Formally the log-likelihood is given by

        .. math::
            L(\psi, \sigma | x^{\text{obs}}) =
            \sum _i \log p(x^{\text{obs}}_i | \psi , \sigma ) ,

        where :math:`p` is the distribution defined by the mechanistic model-
        error model pair and :math:`x^{\text{obs}}` are the observed
        biomarkers. :math:`\psi` and :math:`\sigma` are the parameters of the
        mechanistic model and the error model, respectively.

        The sensitivities of the log-likelihood is defined as the partial
        derivative of the :math:`L` with respect to the model parameters

        .. math::
            \frac{\partial L}{\partial \psi} \quad \text{and} \quad
            \frac{\partial L}{\partial \sigma},

        where both :math:`\psi` and :math:`\sigma` should be interpreted
        as a collection of multiple parameters.

        :param parameters: An array-like object with the error model
            parameters.
        :type parameters: list, numpy.ndarray of length p
        :param model_output: An array-like object with the one-dimensional
            output of a :class:`MechanisticModel`. Each entry is a prediction
            of the mechanistic model for an observed time point in
            ``observations``.
        :type model_output: list, numpy.ndarray of length t
        :param model_sensitivities: An array-like object with the partial
            derivatives of the model output w.r.t. the model parameters.
        :type model_sensitivities: numpy.ndarray of shape (t, p)
        :param observations: An array-like object with the observations of a
            biomarker.
        :type observations: list, numpy.ndarray of length t
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        score, sensitivities = self._error_model.compute_sensitivities(
            parameters, model_output, model_sensitivities, observations)

        if self._fixed_params_mask is None:
            return score, sensitivities

        # Filter sensitivities for fixed parameters
        n_mechanistic = model_sensitivities.shape[1]
        mask = np.ones(n_mechanistic + self._n_parameters, dtype=bool)
        mask[-self._n_parameters:] = ~self._fixed_params_mask

        return score, sensitivities[mask]

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        Parameters
        ----------
        name_value_dict
            A dictionary with model parameter names as keys, and parameter
            values as values.
        """
        # Check type
        try:
            name_value_dict = dict(name_value_dict)
        except (TypeError, ValueError):
            raise ValueError(
                'The name-value dictionary has to be convertable to a python '
                'dictionary.')

        # If no model parameters have been fixed before, instantiate a mask
        # and values
        if self._fixed_params_mask is None:
            self._fixed_params_mask = np.zeros(
                shape=self._n_parameters, dtype=bool)

        if self._fixed_params_values is None:
            self._fixed_params_values = np.empty(shape=self._n_parameters)

        # Update the mask and values
        for index, name in enumerate(self._parameter_names):
            try:
                value = name_value_dict[name]
            except KeyError:
                # KeyError indicates that parameter name is not being fixed
                continue

            # Fix parameter if value is not None, else unfix it
            self._fixed_params_mask[index] = value is not None
            self._fixed_params_values[index] = value

        # If all parameters are free, set mask and values to None again
        if np.alltrue(~self._fixed_params_mask):
            self._fixed_params_mask = None
            self._fixed_params_values = None

    def get_error_model(self):
        """
        Returns the original error model.
        """
        return self._error_model

    def get_parameter_names(self):
        """
        Returns the names of the error model parameters.
        """
        # Remove fixed model parameters
        names = self._parameter_names
        if self._fixed_params_mask is not None:
            names = np.array(names)
            names = names[~self._fixed_params_mask]
            names = list(names)

        return copy.copy(names)

    def n_fixed_parameters(self):
        """
        Returns the number of fixed model parameters.
        """
        if self._fixed_params_mask is None:
            return 0

        n_fixed = int(np.sum(self._fixed_params_mask))

        return n_fixed

    def n_parameters(self):
        """
        Returns the number of parameters of the error model.
        """
        # Get number of fixed parameters
        n_fixed = 0
        if self._fixed_params_mask is not None:
            n_fixed = int(np.sum(self._fixed_params_mask))

        # Subtract fixed parameters from total number
        n_parameters = self._n_parameters - n_fixed

        return n_parameters

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
            :class:`MechanisticModel`.
        n_samples
            Number of samples from the error model for each entry in
            ``model_output``. If ``None``, one sample is assumed.
        seed
            Seed for the pseudo-random number generator. If ``None``, the
            pseudo-random number generator is not seeded.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Sample from error model
        sample = self._error_model.sample(
            parameters, model_output, n_samples, seed)

        return sample

    def set_parameter_names(self, names=None):
        """
        Sets the names of the error model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._error_model.set_parameter_names(None)
            self._parameter_names = self._error_model.get_parameter_names()
            return None

        if len(names) != self.n_parameters():
            raise ValueError(
                'Length of names does not match n_parameters.')

        # Limit the length of parameter names
        for name in names:
            if len(name) > 50:
                raise ValueError(
                    'Parameter names cannot exceed 50 characters.')

        parameter_names = [str(label) for label in names]

        # Reconstruct full list of error model parameters
        if self._fixed_params_mask is not None:
            names = np.array(
                self._error_model.get_parameter_names(), dtype='U50')
            names[~self._fixed_params_mask] = parameter_names
            parameter_names = names

        # Set parameter names
        self._error_model.set_parameter_names(parameter_names)
        self._parameter_names = self._error_model.get_parameter_names()
