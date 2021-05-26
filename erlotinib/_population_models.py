#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy
import math

from numba import njit
import numpy as np
from scipy.stats import norm, truncnorm


class PopulationModel(object):
    """
    A base class for population models.
    """

    def __init__(self):
        super(PopulationModel, self).__init__()

    def compute_log_likelihood(self, parameters, observations):
        """
        Returns the log-likelihood of the population model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        raise NotImplementedError

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        raise NotImplementedError

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        raise NotImplementedError

    def get_parameter_names(self):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        raise NotImplementedError

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        raise NotImplementedError

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        n_samples
            Number of samples. If ``None``, one sample is returned.
        seed
            A seed for the pseudo-random number generator.
        """
        raise NotImplementedError

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        raise NotImplementedError


class HeterogeneousModel(PopulationModel):
    """
    A population model which imposes no relationship on the model parameters
    across individuals.

    A heterogeneous model assumes that the parameters across individuals are
    independent.

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(HeterogeneousModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 0

        # Set default parameter names
        self._parameter_names = None

    def compute_log_likelihood(self, parameters, observations):
        """
        Returns the log-likelihood of the population model parameters.

        A heterogenous population model imposes no restrictions on the
        individuals, as a result the log-likelihood score is zero irrespective
        of the model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        return 0

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        A heterogenous population model imposes no restrictions on the
        individuals, as a result the log-likelihood score is zero irrespective
        of the model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        return np.zeros(shape=len(observations))

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the parameters and the observations.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        n_observations = len(observations)
        return 0, np.zeros(shape=n_observations)

    def get_parameter_names(self):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        return copy.copy(self._parameter_names)

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        n_ids = int(n_ids)

        return (n_ids, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        A heterogeneous population model has no population parameters.
        However, a name may nevertheless be assigned for convience.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = None
            return None

        if len(names) != 1:
            raise ValueError(
                'Length of names has to be 1.')

        self._parameter_names = [str(label) for label in names]


class LogNormalModel(PopulationModel):
    r"""
    A population model which assumes that model parameters across individuals
    are log-normally distributed.

    A log-normal population model assumes that a model parameter :math:`\psi`
    varies across individuals such that :math:`\psi` is log-normally
    distributed in the population

    .. math::
        p(\psi |\mu _{\text{log}}, \sigma _{\text{log}}) =
        \frac{1}{\psi} \frac{1}{\sqrt{2\pi} \sigma _{\text{log}}}
        \exp\left(-\frac{(\log \psi - \mu _{\text{log}})^2}
        {2 \sigma ^2_{\text{log}}}\right).

    Here, :math:`\mu _{\text{log}}` and :math:`\sigma ^2_{\text{log}}` are the
    mean and variance of :math:`\log \psi` in the population, respectively.

    Any observed individual with parameter :math:`\psi _i` is
    assumed to be a realisation of the random variable :math:`\psi`.

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(LogNormalModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 2

        # Set default parameter names
        self._parameter_names = ['Mean log', 'Std. log']

    @staticmethod
    @njit
    def _compute_log_likelihood(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.
        """
        # Compute log-likelihood score
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum(np.log(observations)) \
            - np.sum((np.log(observations) - mean)**2) / 2 / std**2

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    @njit
    def _compute_pointwise_ll(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Transform observations
        log_psi = np.log(observations)

        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - log_psi \
            - (log_psi - mean) ** 2 / (2 * std**2)

        return log_likelihood

    @staticmethod
    @njit
    def _compute_sensitivities(mean, std, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mean = float
        std = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        n_ids = len(psi)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum(np.log(psi)) \
            - np.sum((np.log(psi) - mean)**2) / 2 / std**2

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = - ((np.log(psi) - mean) / std**2 + 1) / psi

        # Copmute sensitivities w.r.t. parameters
        dmean = np.sum(np.log(psi) - mean) / std**2
        dstd = (np.sum((np.log(psi) - mean)**2) / std**2 - n_ids) / std

        sensitivities = np.concatenate((dpsi, np.array([dmean, dstd])))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a LogNormalModel is the log-pdf evaluated
        at the observations

        .. math::
            L(\mu _{\text{log}}, \sigma _{\text{log}}| \Psi) =
            \sum _{i=1}^N
            \log p(\psi _i |
            \mu _{\text{log}}, \sigma _{\text{log}}) ,

        where
        :math:`\Psi := (\psi _1, \ldots , \psi _N)`
        are the "observed" :math:`\psi` from :math:`N` individuals.

        .. note::
            Note that in the context of PKPD modelling the individual
            parameters are never "observed" directly, but rather inferred
            from biomarker measurements.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu _{\text{log}}`, :math:`\sigma _{\text{log}}`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The standard deviation of log psi is strictly positive
            return -np.inf

        return self._compute_log_likelihood(mean, std, observations)

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a LogNormalModel is the log-pdf
        evaluated at the observations

        .. math::
            L(\mu _{\text{log}}, \sigma _{\text{log}}| \psi _i) =
            \log p(\psi _i |
            \mu _{\text{log}}, \sigma _{\text{log}}) ,

        where
        :math:`\psi _i` are the "observed" parameters :math:`\psi` from
        individual :math:`i`.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu _{\text{log}}`, :math:`\sigma _{\text{log}}`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The standard deviation of log psi is strictly positive
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, std, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if std <= 0:
            # The standard deviation of log psi is strictly positive
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(mean, std, observations)

    def get_mean_and_std(self, parameters):
        r"""
        Returns the mean and the standard deviation of the population
        for given :math:`\mu _{\text{log}}` and :math:`\sigma _{\text{log}}`.

        The mean and variance of the parameter :math:`\psi`,
        :math:`\mu = \mathbb{E}\left[ \psi \right]` and
        :math:`\sigma ^2 = \text{Var}\left[ \psi \right]`, are given by

        .. math::
            \mu = \mathrm{e}^{\mu _{\text{log}} + \sigma ^2_{\text{log}} / 2}
            \quad \text{and} \quad
            \sigma ^2 =
            \mu ^2 \left( \mathrm{e}^{\sigma ^2_{\text{log}}} - 1\right) .

        Parameters
        ----------
        mean_log
            Mean of :math:`\log \psi` in the population.
        std_log
            Standard deviation of :math:`\log \psi` in the population.
        """
        # Check input
        mean_log, std_log = parameters
        if std_log < 0:
            raise ValueError('The standard deviation cannot be negative.')

        # Compute mean and standard deviation
        mean = np.exp(mean_log + std_log**2 / 2)
        std = np.sqrt(
            np.exp(2 * mean_log + std_log**2) * (np.exp(std_log**2) - 1))

        return [mean, std]

    def get_parameter_names(self):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        return copy.copy(self._parameter_names)

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        n_ids = int(n_ids)

        return (n_ids, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        Parameters
        ----------
        parameters
            Parameter values of the top-level parameters that are used for the
            simulation.
        n_samples
            Number of samples. If ``None``, one sample is returned.
        seed
            A seed for the pseudo-random number generator.
        """
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Get parameters
        mean, std = parameters

        if std <= 0:
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        # (Mean and sigma are the mean and standard deviation of
        # the log samples)
        rng = np.random.default_rng(seed=seed)
        samples = rng.lognormal(
            mean=mean, sigma=std, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mean log', 'Std. log']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


class PooledModel(PopulationModel):
    """
    A population model which pools the model parameters across individuals.

    A pooled model assumes that the parameters across individuals do not vary.
    As a result, all individual parameters are set to the same value.

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(PooledModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 1

        # Set default parameter names
        self._parameter_names = ['Pooled']

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the unnormalised log-likelihood score of the population model.

        A pooled population model is a delta-distribution centred at the
        population model parameter. As a result the log-likelihood score
        is 0, if all individual parameters are equal to the population
        parameter, and :math:`-\infty` otherwise.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get the population parameter
        parameter = parameters[0]

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        observations = np.array(observations)
        mask = observations != parameter
        if np.any(mask):
            return -np.inf

        # Otherwise return 0
        return 0

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        A pooled population model is a delta-distribution centred at the
        population model parameter. As a result the log-likelihood score
        is 0, if all individual parameters are equal to the population
        parameter, and :math:`-\infty` otherwise.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get the population parameter
        parameter = parameters[0]

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        log_likelihood = np.zeros(shape=len(observations))
        observations = np.array(observations)
        mask = observations != parameter
        log_likelihood[mask] = -np.inf

        return log_likelihood

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get the population parameter
        parameter = parameters[0]

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        observations = np.array(observations)
        n_obs = len(observations)
        mask = observations != parameter
        if np.any(mask):
            return -np.inf, np.full(shape=n_obs + 1, fill_value=np.inf)

        # Otherwise return 0
        return 0, np.zeros(shape=n_obs + 1)

    def get_parameter_names(self):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        return copy.copy(self._parameter_names)

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        return (0, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the underlying population
        distribution.

        For a PooledModel the input top-level parameters are copied
        ``n_samples`` and are returned.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        Parameters
        ----------
        parameters
            Parameter values of the top-level parameters that are used for the
            simulation.
        n_samples
            Number of samples. If ``None``, one sample is returned.
        seed
            A seed for the pseudo-random number generator.
        """
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        samples = np.asarray(parameters)

        # If only one sample is wanted, return input parameter
        if n_samples is None:
            return samples

        # If more samples are wanted, broadcast input parameter to shape
        # (n_samples,)
        samples = np.broadcast_to(samples, shape=(n_samples,))
        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Pooled']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class ReducedPopulationModel(object):
    """
    A class that can be used to permanently fix model parameters of a
    :class:`PopulationModel` instance.

    This may be useful to explore simplified versions of a model.

    Parameters
    ----------
    population_model
        An instance of a :class:`PopulationModel`.
    """

    def __init__(self, population_model):
        super(ReducedPopulationModel, self).__init__()

        # Check inputs
        if not isinstance(population_model, PopulationModel):
            raise TypeError(
                'The population model has to be an instance of a '
                'erlotinib.PopulationModel.')

        self._population_model = population_model

        # Set defaults
        self._fixed_params_mask = None
        self._fixed_params_values = None
        self._n_parameters = population_model.n_parameters()
        self._parameter_names = population_model.get_parameter_names()

    def compute_log_likelihood(self, parameters, observations):
        """
        Returns the log-likelihood of the population model parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Compute log-likelihood
        score = self._population_model.compute_log_likelihood(
            parameters, observations)

        return score

    def compute_pointwise_ll(self, parameters, observations):
        """
        Returns the pointwise log-likelihood of the population model parameters
        for each observation.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Compute log-likelihood
        scores = self._population_model.compute_pointwise_ll(
            parameters, observations)

        return scores

    def compute_sensitivities(self, parameters, observations):
        """
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Compute log-likelihood and sensitivities
        score, sensitivities = self._population_model.compute_sensitivities(
            parameters, observations)

        if self._fixed_params_mask is None:
            return score, sensitivities

        # Filter sensitivities for fixed parameters
        n_obs = len(observations)
        mask = np.ones(n_obs + self._n_parameters, dtype=bool)
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

        # If population model does not have model parameters, break here
        if self._n_parameters == 0:
            return None

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

    def get_parameter_names(self):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        # Remove fixed model parameters
        names = self._parameter_names
        if self._fixed_params_mask is not None:
            names = np.array(names)
            names = names[~self._fixed_params_mask]
            names = list(names)

        return copy.copy(names)

    def get_population_model(self):
        """
        Returns the original population model.
        """
        return self._population_model

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        # Get individual parameters
        n_indiv, n_pop = self._population_model.n_hierarchical_parameters(
            n_ids)

        # If parameters have been fixed, updated number of population
        # parameters
        if self._fixed_params_mask is not None:
            n_pop = int(np.sum(self._fixed_params_mask))

        return (n_indiv, n_pop)

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
        Returns the number of parameters of the population model.
        """
        # Get number of fixed parameters
        n_fixed = 0
        if self._fixed_params_mask is not None:
            n_fixed = int(np.sum(self._fixed_params_mask))

        # Subtract fixed parameters from total number
        n_parameters = self._n_parameters - n_fixed

        return n_parameters

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the underlying population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        Parameters
        ----------
        parameters
            Parameter values of the top-level parameters that are used for the
            simulation.
        n_samples
            Number of samples. If ``None``, one sample is returned.
        seed
            A seed for the pseudo-random number generator.
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Sample from population model
        sample = self._population_model.sample(parameters, n_samples, seed)

        return sample

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        Parameters
        ----------
        names
            A dictionary that maps the current parameter names to new names.
            If ``None``, parameter names are reset to defaults.
        """
        if names is None:
            # Reset names to defaults
            self._population_model.set_parameter_names()
            self._parameter_names = \
                self._population_model.get_parameter_names()
            return None

        # Check input
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
                self._population_model.get_parameter_names(), dtype='U50')
            names[~self._fixed_params_mask] = parameter_names
            parameter_names = names

        # Set parameter names
        self._population_model.set_parameter_names(parameter_names)
        self._parameter_names = self._population_model.get_parameter_names()


class TruncatedGaussianModel(PopulationModel):
    r"""
    A population model which assumes that model parameters across individuals
    are distributed according to a Gaussian distribution which is truncated at
    zero.

    A truncated Gaussian population model assumes that a model parameter
    :math:`\psi` varies across individuals such that :math:`\psi` is
    Gaussian distributed in the population for :math:`\psi` greater 0

    .. math::
        p(\psi |\mu, \sigma) =
        \frac{1}{1 - \Phi (-\mu / \sigma )} \frac{1}{\sqrt{2\pi} \sigma}
        \exp\left(-\frac{(\psi - \mu )^2}
        {2 \sigma ^2}\right)\quad \text{for} \quad \psi > 0

    and :math:`p(\psi |\mu, \sigma) = 0` for :math:`\psi \leq 0`.
    :math:`\Phi (\psi )` denotes the cumulative distribution function of
    the Gaussian distribution.

    Here, :math:`\mu` and :math:`\sigma ^2` are the
    mean and variance of the untruncated Gaussian distribution.

    Any observed individual with parameter :math:`\psi _i` is
    assumed to be a realisation of the random variable :math:`\psi`.

    Extends :class:`PopulationModel`.
    """

    def __init__(self):
        super(TruncatedGaussianModel, self).__init__()

        # Set number of parameters
        self._n_parameters = 2

        # Set default parameter names
        self._parameter_names = ['Mu', 'Sigma']

    @staticmethod
    @njit
    def _compute_log_likelihood(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.

        We are using the relationship between the Gaussian CDF and the
        error function

        ..math::
            Phi(x) = (1 + erf(x/sqrt(2))) / 2
        """
        # Compute log-likelihood score
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * std**2) / 2 \
            - np.sum((observations - mean) ** 2) / (2 * std**2) \
            - n_ids * np.log(1 - _norm_cdf(-mean/std))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    @njit
    def _compute_pointwise_ll(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - (observations - mean) ** 2 / (2 * std**2) \
            - np.log(1 - math.erf(-mean/std/math.sqrt(2))) + np.log(2)

        return log_likelihood

    @staticmethod
    @njit
    def _compute_sensitivities(mean, std, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mean = float
        std = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        n_ids = len(psi)
        log_likelihood = \
            - n_ids * (np.log(2 * np.pi) / 2 + np.log(std)) \
            - np.sum((psi - mean)**2) / (2 * std**2) \
            - n_ids * np.log(1 - _norm_cdf(-mean/std))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = (mean - psi) / std**2

        # Copmute sensitivities w.r.t. parameters
        dmean = (
            np.sum(psi - mean) / std
            - _norm_pdf(mean/std) / (1 - _norm_cdf(-mean/std)) * n_ids
            ) / std
        dstd = (
            -n_ids + np.sum((psi - mean)**2) / std**2
            + _norm_pdf(mean/std) * mean / std / (1 - _norm_cdf(-mean/std))
            * n_ids
            ) / std

        sensitivities = np.concatenate((dpsi, np.array([dmean, dstd])))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a truncated Gaussian distribution is the log-pdf
        evaluated at the observations

        .. math::
            L(\mu , \sigma | \Psi) =
            \sum _{i=1}^N
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\Psi := (\psi _1, \ldots , \psi _N)`
        are the "observed" :math:`\psi` from :math:`N` individuals.

        .. note::
            Note that in the context of PKPD modelling the individual
            parameters are never "observed" directly, but rather inferred
            from biomarker measurements.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if (mean <= 0) or (std <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return -np.inf

        return self._compute_log_likelihood(mean, std, observations)

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a truncated Gaussian distribution is
        the log-pdf evaluated at the observations

        .. math::
            L(\mu , \sigma | \psi _i) =
            \log p(\psi _i |
            \mu , \sigma ) ,

        where
        :math:`\psi _i` are the "observed" parameters :math:`\psi` from
        individual :math:`i`.

        Parameters
        ----------
        parameters
            An array-like object with the model parameter values, i.e.
            [:math:`\mu`, :math:`\sigma`].
        observations
            An array like object with the parameter values for the individuals,
            i.e. [:math:`\psi _1, \ldots , \psi _N`].
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if (mean <= 0) or (std <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, std, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        Parameters
        ----------
        parameters
            An array-like object with the parameters of the population model.
        observations
            An array-like object with the observations of the individuals. Each
            entry is assumed to belong to one individual.
        """
        observations = np.asarray(observations)
        mean, std = parameters

        if (mean <= 0) or (std <= 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(mean, std, observations)

    def get_mean_and_std(self, parameters):
        r"""
        Returns the mean and the standard deviation of the population
        for given :math:`\mu` and :math:`\sigma`.

        The mean and variance of the parameter :math:`\psi` are given
        by

        .. math::
            \mathbb{E}\left[ \psi \right] =
                \mu + \sigma F(\mu/\sigma)
            \quad \text{and} \quad
            \text{Var}\left[ \psi \right] =
                \sigma ^2 \left[
                    1 - \frac{\mu}{\sigma}F(\mu/\sigma)
                    - F(\mu/\sigma) ^2
                \right],

        where :math:`F(\mu/\sigma) = \phi(\mu/\sigma )/(1-\Phi(-\mu/\sigma))`
        is a function given by the Gaussian probability density function
        :math:`\phi(\psi)` and the Gaussian cumulative distribution function
        :math:`\Phi(\psi)`.

        Parameters
        ----------
        mu
            Mean of untruncated Gaussian distribution.
        sigma
            Standard deviation of untruncated Gaussian distribution.
        """
        # Check input
        mu, sigma = parameters
        if (mu < 0) or (sigma < 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            raise ValueError(
                'The parameters mu and sigma cannot be negative.')

        # Compute mean and standard deviation
        mean = mu + sigma * norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma))
        std = np.sqrt(
            sigma**2 * (
                1 -
                mu / sigma * norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma))
                - (norm.pdf(mu/sigma) / (1 - norm.cdf(-mu/sigma)))**2)
            )

        return [mean, std]

    def get_parameter_names(self):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        return copy.copy(self._parameter_names)

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        Parameters
        ----------
        n_ids
            Number of individuals.
        """
        n_ids = int(n_ids)

        return (n_ids, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        Parameters
        ----------
        parameters
            Parameter values of the top-level parameters that are used for the
            simulation.
        n_samples
            Number of samples. If ``None``, one sample is returned.
        seed
            A seed for the pseudo-random number generator.
        """
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples),)

        # Get parameters
        mu, sigma = parameters

        if (mu < 0) or (sigma < 0):
            # The mean and std. of the Gaussian distribution are
            # strictly positive if truncated at zero
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')

        # Convert seed to int if seed is a rng
        # (Unfortunately truncated normal is not yet available with numpys
        # random number generator API)
        if isinstance(seed, np.random.Generator):
            # Draw new seed such that rng is propagated, but truncated normal
            # samples can also be seeded.
            seed = seed.integers(low=0, high=1E6)
        np.random.seed(seed)

        # Sample from population distribution
        samples = truncnorm.rvs(
            a=0, b=np.inf, loc=mu, scale=sigma, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_parameters`. If ``None``, parameter names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mu', 'Sigma']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


@njit
def _norm_cdf(x):  # pragma: no cover
    """
    Returns the cumulative distribution function value of a standard normal
    Gaussian distribtion.
    """
    return 0.5 * (1 + math.erf(x/math.sqrt(2)))


@njit
def _norm_pdf(x):  # pragma: no cover
    """
    Returns the probability density function value of a standard normal
    Gaussian distribtion.
    """
    return math.exp(-x**2/2) / math.sqrt(2 * math.pi)
