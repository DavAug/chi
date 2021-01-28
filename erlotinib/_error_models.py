#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy

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
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
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
    An error model that assumes that the model error is a mixture between a
    Gaussian base-level noise and a Gaussian heteroscedastic noise.

    A ConstantAndMultiplicativeGaussianErrorModel assumes that the observable
    biomarker :math:`X` is related to the :class:`MechanisticModel` biomarker
    output by

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

    def compute_log_likelihood(self, parameters, model_output, observations):
        r"""
        Returns the unnormalised log-likelihood score for the model parameters
        of the mechanistic model-error model pair.

        In this method, the model output :math:`x^{\text{m}}` and the
        observations :math:`x^{\text{obs}}` are compared pair-wise, and the
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

        .. note::
            All constant terms that do not depend on the model parameters are
            dropped when computing the log-likelihood score.

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
        model_output = np.asarray(model_output)
        observations = np.asarray(observations)
        n_observations = len(observations)
        if len(model_output) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pair-wise.')

        # Get parameters
        sigma_base, sigma_rel = parameters

        if sigma_base <= 0 or sigma_rel <= 0:
            # sigma_base and sigma_rel are strictly positive
            return -np.inf

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


class MultiplicativeGaussianErrorModel(ErrorModel):
    r"""
    An error model that assumes that the model error is a Gaussian
    heteroscedastic noise.

    A MultiplicativeGaussianErrorModel assumes that the observable
    biomarker :math:`X` is related to the :class:`MechanisticModel` biomarker
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

    def compute_log_likelihood(self, parameters, model_output, observations):
        r"""
        Returns the unnormalised log-likelihood score for the model parameters
        of the mechanistic model-error model pair.

        In this method, the model output :math:`x^{\text{m}}` and the
        observations :math:`x^{\text{obs}}` are compared pair-wise, and the
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

        .. note::
            All constant terms that do not depend on the model parameters are
            dropped when computing the log-likelihood score.

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
        model_output = np.asarray(model_output)
        observations = np.asarray(observations)
        n_observations = len(observations)
        if len(model_output) != n_observations:
            raise ValueError(
                'The number of model outputs must match the number of '
                'observations, otherwise they cannot be compared pair-wise.')

        # Get parameters
        sigma_rel = parameters[0]

        if sigma_rel <= 0:
            # sigma_base and sigma_rel are strictly positive
            return -np.inf

        # Compute total standard deviation
        sigma_tot = sigma_rel * model_output

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
        """
        Returns the unnormalised log-likelihood score for the model parameters
        of the mechanistic model-error model pair.

        In this method, the model output and the observations are compared
        pair-wise. The time-dependence of the values is thus dealt with
        implicitly, by assuming that ``model_output`` and ``observations`` are
        already ordered, such that the first entries correspond to the same
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
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        score = self._error_model.compute_log_likelihood(
            parameters, model_output, observations)
        return score

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
