#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy
import math

import numpy as np
from scipy.stats import norm, truncnorm

import chi


class PopulationModel(object):
    """
    A base class for population models.

    Population models can be multi-dimensional, but unless explicitly specfied,
    the dimensions of the model are assumed to be independent.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    """
    def __init__(self, n_dim=1, dim_names=None):
        super(PopulationModel, self).__init__()
        if n_dim < 1:
            raise ValueError(
                'The dimension of the population model has to be greater or '
                'equal to 1.')
        self._n_dim = int(n_dim)
        self._transforms_psi = False

        if dim_names:
            if len(dim_names) != self._n_dim:
                raise ValueError(
                    'The number of dimension names has to match the number of '
                    'dimensions of the population model.')
            dim_names = [str(name) for name in dim_names]
        else:
            dim_names = [
                'Dim. %d' % (id_dim + 1) for id_dim in range(self._n_dim)]
        self._dim_names = dim_names

    def compute_log_likelihood(self, parameters, observations):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihood of individual parameters and population
            parameters.
        :rtype: float
        """
        raise NotImplementedError

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p, n_dim)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihoods for each individual parameter for population
            parameters.
        :rtype: np.ndarray of length (n, n_dim)
        """
        raise NotImplementedError

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihood and its sensitivity to individual parameters
            as well as population parameters.
        :rtype: Tuple[float, np.ndarray of shape (n + p, n_dim)]
        """
        raise NotImplementedError

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        return self._dim_names

    def get_parameter_names(self):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.
        """
        raise NotImplementedError

    def n_dim(self):
        """
        Returns the dimensionality of the population model.
        """
        return self._n_dim

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

    def transforms_individual_parameters(self):
        r"""
        Returns a boolean whether the population model models the individual
        likelihood parameters directly or a transform of those parameters.

        Some population models compute the likelihood of the population
        parameters :math:`\theta` based on estimates of the
        individual likelihood parameters :math:`\Psi = \{\psi _i \} _{i=1}^n`,
        where :math:`n` is the number of individual likelihoods. Here,
        the parameters are not transformed and ``False`` is returned.

        Other population models, in particular the
        :class:`CovariatePopulationModel`, transforms the parameters to a
        latent representation
        :math:`\Psi \rightarrow \{\eta _i \} _{i=1}^n`.
        Here, a transformation of the likelihood parameters is modelled and
        ``True`` is returned.
        """
        return self._transforms_psi

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        raise NotImplementedError

    def set_dim_names(self, names=None):
        r"""
        Sets the names of the population model dimensions.

        Parameters
        ----------
        names
            An array-like object with string-convertable entries of length
            :meth:`n_dim`. If ``None``, dimension names are reset to
            defaults.
        """
        if names is None:
            # Reset names to defaults
            self._dim_names = [
                'Dim. %d' % (id_dim + 1) for id_dim in range(self._n_dim)]
            return None

        if len(names) != self._n_dim:
            raise ValueError(
                'Length of names does not match the number of dimensions.')

        self._dim_names = [str(label) for label in names]

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


class CovariatePopulationModel(PopulationModel):
    r"""
    A CovariatePopulationModel assumes that the individual parameters
    :math:`\psi` are distributed according to a population model that is
    conditional on the model parameters :math:`\vartheta` and the covariates
    :math:`\chi`

    .. math::
        \psi \sim \mathbb{P}(\cdot | \vartheta, \chi).

    Here, covariates identify subpopulations in the population and can vary
    from one individual to the next, while the model parameters
    :math:`\vartheta` are the same for all individuals.

    To simplify this dependence, CovariatePopulationModels make the assumption
    that the distribution :math:`\mathbb{P}(\psi | \vartheta, \chi)`
    deterministically varies with the covariates, such that the distribution
    can be rewritten in terms of a covariate-independent distribution of
    inter-individual fluctuations :math:`\eta`

    .. math::
        \eta \sim \mathbb{P}(\cdot | \theta)

    and a set of deterministic relationships for the individual parameters
    :math:`\psi`  and the new population parameters :math:`\theta`

    .. math::
        \theta = f(\vartheta)  \quad \mathrm{and} \quad
        \psi = g(\vartheta , \eta, \chi ).

    The ``population_model`` input defines the distribution of :math:`\eta`
    and the ``covariate_model`` defines the functions :math:`f` and :math:`g`.

    Extends :class:`PopulationModel`.

    :param population_model: Defines the distribution of :math:`\eta`.
    :type population_model: PopulationModel
    :param covariate_model: Defines the covariate model.
    :type covariate_model: CovariateModel
    """

    def __init__(self, population_model, covariate_model):
        super(CovariatePopulationModel, self).__init__()

        # Check inputs
        if not isinstance(population_model, PopulationModel):
            raise TypeError(
                'The population model has to be an instance of a '
                'chi.PopulationModel.')
        if not isinstance(covariate_model, chi.CovariateModel):
            raise TypeError(
                'The covariate model has to be an instance of a '
                'chi.CovariateModel.')

        # Check compatibility of population model with covariate model
        covariate_model.check_compatibility(population_model)

        # Remember models
        self._population_model = population_model
        self._covariate_model = covariate_model

        # Set transform psis to true
        self._transforms_psi = True

    def compute_individual_parameters(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi`.

        By default ``covariates`` are set to ``None``, such that model
        does not rely on covariates. Each derived :class:`CovariateModel`
        needs to make sure that model reduces to sensible values for
        this edge case.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters :math:`\psi`.
        :rtype: np.ndarray of length (n,)
        """
        return self._covariate_model.compute_individual_parameters(
            parameters, eta, covariates)

    def compute_individual_sensitivities(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi` and their sensitivities
        with respect to the model parameters :math:`\vartheta` and the relevant
        fluctuations :math:`\eta`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters :math:`\psi` and sensitivities
            (:math:`\partial _{\eta} \psi` ,
            :math:`\partial _{\vartheta _1} \psi`, :math:`\ldots`,
            :math:`\partial _{\vartheta _p} \psi`).
        :rtype: Tuple[np.ndarray, np.ndarray] of shapes (n,) and (1 + p, n)
        """
        return self._covariate_model.compute_individual_sensitivities(
            parameters, eta, covariates)

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the model parameters.

        :param parameters: Values of the model parameters :math:`\vartheta`.
        :type parameters: List, np.ndarray of length (p,)
        :param observations: "Observations" of the individuals :math:`\eta`.
            Typically refers to the inter-individual fluctuations of the
            mechanistic model parameter.
        :type observations: List, np.ndarray of length (n,)
        :returns: Log-likelihood of individual parameters and population
            parameters.
        :rtype: float
        """
        # Compute population parameters
        parameters = self._covariate_model.compute_population_parameters(
            parameters)

        # Compute log-likelihood
        score = self._population_model.compute_log_likelihood(
            parameters, observations)

        return score

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        :param parameters: Values of the model parameters :math:`\vartheta`.
        :type parameters: List, np.ndarray of length (p,)
        :param observations: "Observations" of the individuals :math:`\eta`.
            Typically refers to the inter-individual fluctuations of the
            mechanistic model parameter.
        :type observations: List, np.ndarray of length (n,)
        :returns: Log-likelihoods of individual parameters for population
            parameters.
        :rtype: np.ndarray of length (n,)
        """
        # Compute population parameters
        parameters = self._covariate_model.compute_population_parameters(
            parameters)

        # Compute log-likelihood
        score = self._population_model.compute_pointwise_ll(
            parameters, observations)

        return score

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the observations and the parameters.

        The sensitivities are computed with respect to the individual
        :math:`\eta _i` and the population parameters :math:`\vartheta`

        .. math::
            \left(
                \partial _{\eta _i}\log p(\eta _i | \theta),
                \sum _{i,j}\partial _{\theta _j}\log p(\eta _i | \theta_j)
                    \frac{\partial f_j}{\partial \vartheta _k}\right) .

        :param parameters: Parameters of the population model.
        :type parameters: List, np.ndarray of length (p,)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: List, np.ndarray of length (n,)
        :returns: Log-likelihood and its sensitivity to individual parameters
            as well as population parameters.
        :rtype: Tuple[float, np.ndarray], where array is of shape (n + p,)
        """
        # Compute population parameters and sensitivities dtheta/dvartheta
        params, dvartheta = \
            self._covariate_model.compute_population_sensitivities(
                parameters)

        # Compute log-likelihood and sensitivities dscore/deta, dscore/dtheta
        score, sensitivities = self._population_model.compute_sensitivities(
            params, observations)

        # Propagate sensitivity of score to vartheta
        # i.e. dscore/dvartheta = sum_i dscore/dtheta_i * dtheta_i/dvartheta
        # Note dvartheta has shape (p, p') and dtheta has shape (p')
        n = len(observations)
        deta = sensitivities[:n]
        dtheta = sensitivities[n:]
        dvartheta = dvartheta @ dtheta

        # Stack results
        sensitivities = np.hstack((deta, dvartheta))

        return (score, sensitivities)

    def get_covariate_model(self):
        """
        Returns the covariate model.
        """
        return self._covariate_model

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        return self._covariate_model.get_covariate_names()

    def get_parameter_names(self):
        """
        Returns the names of the model parameters. If name is
        not set, defaults are returned.
        """
        return self._covariate_model.get_parameter_names()

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
        # Get number of individual parameters
        n_ids, _ = self._population_model.n_hierarchical_parameters(n_ids)

        return (n_ids, self._covariate_model.n_parameters())

    def n_covariates(self):
        """
        Returns the number of covariates.
        """
        return self._covariate_model.n_covariates()

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._covariate_model.n_parameters()

    def sample(
            self, parameters, n_samples=None, seed=None, covariates=None,
            return_psi=False):
        r"""
        Returns random samples from the population distribution.

        By default samples from

        .. math::
            \psi \sim \mathbb{P}(\cdot | \vartheta, \chi)

        are returned. If ``return_psi=False`` samples from

        .. math::
            \eta \sim \mathbb{P}(\cdot | \theta)

        are returned.

        :param parameters: Values of the model parameters.
        :type parameters: List, np.ndarray of shape (p,)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, np.random.Generator, optional
        :param covariates: Values for the covariates. If ``None``, default
            is assumed defined by the :class:`CovariateModel`.
        :type covariates: List, np.ndarray of shape (c,)
        :param return_psi: Boolean flag that indicates whether the parameters
            of the individual likelihoods are returned or the transformed
            inter-individual fluctuations.
        :type return_psi: bool, optional
        :returns: Samples from population model conditional on covariates.
        :rtype: np.ndarray of shape (n_samples,)
        """
        # Check that covariates has the correct dimensions
        if covariates is not None:
            covariates = np.array(covariates)
            n_covariates = self._covariate_model.n_covariates()
            if len(covariates) != n_covariates:
                raise ValueError(
                    'Covariates must be of length n_covariates.')

            # Add dimension to fit shape (n, c) for later convenience
            covariates = np.reshape(covariates, (1, n_covariates))

        # Compute population parameters
        eta_dist_params = self._covariate_model.compute_population_parameters(
            parameters)

        # Sample eta from population model
        eta = self._population_model.sample(eta_dist_params, n_samples, seed)

        if not return_psi:
            return eta

        # Compute psi
        psi = self._covariate_model.compute_individual_parameters(
            parameters, eta, covariates)

        return psi

    def set_covariate_names(self, names=None, update_param_names=False):
        """
        Sets the names of the covariates.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List
        :param update_param_names: Boolean flag indicating whether parameter
            names should be updated according to new covariate names. By
            default parameter names are not updated.
        :type update_param_names: bool, optional
        """
        self._covariate_model.set_covariate_names(names, update_param_names)

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
        self._covariate_model.set_parameter_names(names)


class GaussianModel(PopulationModel):
    r"""
    A population model which assumes that model parameters across individuals
    are distributed according to a Gaussian distribution.

    A Gaussian population model assumes that a model parameter
    :math:`\psi` varies across individuals such that :math:`\psi` is
    Gaussian distributed in the population

    .. math::
        p(\psi |\mu, \sigma) =
        \frac{1}{\sqrt{2\pi} \sigma}
        \exp\left(-\frac{(\psi - \mu )^2}
        {2 \sigma ^2}\right).

    Here, :math:`\mu` and :math:`\sigma ^2` are the
    mean and variance of the Gaussian distribution.

    Any observed individual with parameter :math:`\psi _i` is
    assumed to be a realisation of the random variable :math:`\psi`.

    Extends :class:`PopulationModel`.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    """

    def __init__(self, n_dim=1, dim_names=None):
        super(GaussianModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = [
            'Mean ' + dim_name for dim_name in self._dim_names] + [
            'Std. ' + dim_name for dim_name in self._dim_names]

    @staticmethod
    def _compute_log_likelihood(mus, vars, observations):
        r"""
        Calculates the log-likelihood.

        mus shape: (n_dim,)
        vars shape: (n_dim,)
        observations: (n_ids, n_dim)
        """
        # Compute log-likelihood score
        log_likelihood = - np.sum(
            np.log(2 * np.pi * vars) / 2 + (observations - mus) ** 2
            / (2 * vars))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(mean, var, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * var) / 2 \
            - (observations - mean) ** 2 / (2 * var)

        # If score evaluates to NaN, return -infinity
        mask = np.isnan(log_likelihood)
        if np.any(mask):
            log_likelihood[mask] = -np.inf
            return log_likelihood

        return log_likelihood

    def _compute_sensitivities(self, mus, vars, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mus shape: (n_dim,)
        vars shape: (n_dim,)
        observations: (n_ids, n_dim)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_ids * n_dim + 2 * n_dim,)
        """
        # Compute log-likelihood score
        n_ids = len(psi)
        log_likelihood = self._compute_log_likelihood(mus, vars, psi)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf, np.full(shape=n_ids + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = (mus - psi) / vars

        # Compute sensitivities w.r.t. parameters
        dmus = np.sum(psi - mus, axis=0) / vars
        dstd = (-n_ids + np.sum((psi - mus)**2, axis=0) / vars) / np.sqrt(vars)

        # Collect sensitivities
        # ([psis dim 1, ..., psis dim d, mu dim 1, ..., mu dim d, sigma dim 1,
        # ..., sigma dim d])
        sensitivities = np.concatenate((
            dpsi.T.flatten(), np.array([dmus, dstd]).flatten()))

        return log_likelihood, sensitivities

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        The log-likelihood of a Gaussian distribution is the log-pdf
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
            from measurements.

        :param parameters: Parameters of the population model, i.e.
            [:math:`\mu`, :math:`\sigma`]. If the population model is
            multi-dimensional :math:`\mu` and :math:`\sigma` are expected to be
            vector-valued. The parameters can then either be concatenated or
            used to define a parameter matrix.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual, i.e. [:math:`\psi _1, \ldots , \psi _N`].
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihood of individual parameters and population
            parameters.
        :rtype: float
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Parse parameters
        mus = parameters[0]
        sigmas = parameters[1]
        vars = sigmas**2

        eps = 1E-12
        if np.any(sigmas <= 0) or np.any(vars <= eps):
            # The std. of the Gaussian distribution is strictly positive
            return -np.inf

        return self._compute_log_likelihood(mus, vars, observations)

    def compute_pointwise_ll(
            self, parameters, observations):  # pragma: no cover
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        The pointwise log-likelihood of a Gaussian distribution is
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
        # TODO: Needs proper research to establish which pointwise
        # log-likelihood makes sense for hierarchical models.
        # Also needs to be adapted to match multi-dimensional API.
        raise NotImplementedError
        observations = np.asarray(observations)
        mean, std = parameters
        var = std**2

        eps = 1E-6
        if (std <= 0) or (var <= eps):
            # The std. of the Gaussian distribution is strictly positive
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, var, observations)

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivity w.r.t. the observations and the parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihood and its sensitivity to individual parameters
            as well as population parameters.
        :rtype: Tuple[float, np.ndarray of shape (n + p, n_dim)]
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Parse parameters
        mus = parameters[0]
        sigmas = parameters[1]
        vars = sigmas**2

        eps = 1E-6
        if np.any(sigmas <= 0) or np.any(vars <= eps):
            # The std. of the Gaussian distribution is strictly positive
            n_obs = len(observations)
            return -np.inf, np.full(
                shape=(n_obs + 2, self._n_dim), fill_value=np.inf)

        return self._compute_sensitivities(mus, vars, observations)

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

        return (n_ids * self._n_dim, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(self, parameters, n_samples=None, seed=None):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)

        # Get parameters
        mus = parameters[0]
        sigmas = parameters[1]

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution are
            # strictly positive
            raise ValueError(
                'A Gaussian distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        rng = np.random.default_rng(seed=seed)
        samples = rng.normal(
            loc=mus, scale=sigmas, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a GaussianModel are the population mean
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
            self._parameter_names = [
                'Mean ' + dim_name for dim_name in self._dim_names] + [
                'Std. ' + dim_name for dim_name in self._dim_names]
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        self._parameter_names = [str(label) for label in names]


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
    def _compute_log_likelihood(mean, var, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.
        """
        # Compute log-likelihood score
        n_ids = len(observations)
        log_likelihood = \
            - n_ids * np.log(2 * np.pi * var) / 2 \
            - np.sum(np.log(observations)) \
            - np.sum((np.log(observations) - mean)**2) / 2 / var

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_pointwise_ll(mean, var, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Transform observations
        log_psi = np.log(observations)

        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * var) / 2 \
            - log_psi \
            - (log_psi - mean) ** 2 / (2 * var)

        # If score evaluates to NaN, return -infinity
        mask = np.isnan(log_likelihood)
        if np.any(mask):
            log_likelihood[mask] = -np.inf
            return log_likelihood

        return log_likelihood

    def _compute_sensitivities(self, mean, var, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities using numba
        speed up.

        Expects:
        mean = float
        var = float
        Shape observations =  (n_obs,)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        n_ids = len(psi)
        log_likelihood = self._compute_log_likelihood(mean, var, psi)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf, np.full(shape=n_ids + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = - ((np.log(psi) - mean) / var + 1) / psi

        # Copmute sensitivities w.r.t. parameters
        dmean = np.sum(np.log(psi) - mean) / var
        dstd = (np.sum((np.log(psi) - mean)**2) / var - n_ids) / np.sqrt(var)

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
        var = std**2

        eps = 1E-12
        if (std <= 0) or (var <= eps) or np.any(observations == 0):
            # The standard deviation of log psi is strictly positive
            return -np.inf

        return self._compute_log_likelihood(mean, var, observations)

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
        var = std**2

        eps = 1E-12
        if (std <= 0) or (var <= eps) or np.any(observations == 0):
            # The standard deviation of log psi is strictly positive
            return np.full(shape=len(observations), fill_value=-np.inf)

        return self._compute_pointwise_ll(mean, var, observations)

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
        var = std**2

        eps = 1E-12
        if (std <= 0) or (var <= eps) or np.any(observations == 0):
            # The standard deviation of log psi is strictly positive
            n_obs = len(observations)
            return -np.inf, np.full(shape=(n_obs + 2,), fill_value=np.inf)

        return self._compute_sensitivities(mean, var, observations)

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
                'chi.PopulationModel.')

        self._population_model = population_model

        # Set defaults
        self._fixed_params_mask = None
        self._fixed_params_values = None
        self._n_parameters = population_model.n_parameters()
        self._parameter_names = population_model.get_parameter_names()

    def compute_individual_parameters(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi`.

        If wrapped model does not transform the individual likelihood
        parameters ``eta`` is returned.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters :math:`\psi`.
        :rtype: np.ndarray of length (n,)
        """
        if not self.transforms_individual_parameters():
            return eta

        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        return self._population_model.compute_individual_parameters(
            parameters, eta, covariates)

    def compute_individual_sensitivities(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi` and their sensitivities
        with respect to the model parameters :math:`\vartheta` and the relevant
        fluctuation :math:`\eta`.

        If wrapped model does not transform the individual likelihood
        parameters ``eta`` is returned and the sensitivities are trivially
        1 for the relevant fluctuations and 0 for the model parameters.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters and sensitivities of shape (1 + p, n).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if not self.transforms_individual_parameters():
            n = len(eta)
            p = len(parameters)
            sens = np.vstack((
                np.ones(shape=(1, n)),
                np.zeros(shape=(p, n))
            ))
            return eta, sens

        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        return self._population_model.compute_individual_sensitivities(
            parameters, eta, covariates)

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

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        try:
            return self._population_model.get_covariate_names()
        except AttributeError:
            return []

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

    def n_covariates(self):
        """
        Returns the number of covariates.
        """
        try:
            return self._population_model.n_covariates()
        except AttributeError:
            return 0

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

    def sample(
            self, parameters, n_samples=None, seed=None, covariates=None,
            return_psi=True):
        r"""
        Returns random samples from the underlying population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        :param parameters: Values of the model parameters.
        :type parameters: List, np.ndarray of shape (p,)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, np.random.Generator, optional
        :param covariates: Values for the covariates. If ``None``, default
            is assumed defined by the :class:`CovariateModel`.
        :type covariates: List, np.ndarray of shape (c,)
        :param return_psi: Boolean flag that indicates whether the parameters
            of the individual likelihoods are returned or the transformed
            inter-individual fluctuations.
        :type return_psi: bool, optional
        :returns: Samples from population model conditional on covariates.
        :rtype: np.ndarray of shape (n_samples,)
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Sample from population model
        if self.transforms_individual_parameters():
            sample = self._population_model.sample(
                parameters, n_samples, seed, covariates, return_psi)
        else:
            sample = self._population_model.sample(parameters, n_samples, seed)

        return sample

    def set_covariate_names(self, names=None, update_param_names=False):
        """
        Sets the names of the covariates.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List
        :param update_param_names: Boolean flag indicating whether parameter
            names should be updated according to new covariate names. By
            default parameter names are not updated.
        :type update_param_names: bool, optional
        """
        try:
            self._population_model.set_covariate_names(
                names, update_param_names)
        except AttributeError:
            return None

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

    def transforms_individual_parameters(self):
        r"""
        Returns a boolean whether the population model models the individual
        likelihood parameters directly or a transform of those parameters.

        Some population models compute the likelihood of the population
        parameters :math:`\theta` based on estimates of the
        individual likelihood parameters :math:`\Psi = \{\psi _i \} _{i=1}^n`,
        where :math:`n` is the number of individual likelihoods. Here,
        the parameters are not transformed and ``False`` is returned.

        Other population models, in particular the
        :class:`CovariatePopulationModel`, transforms the parameters to a
        latent representation
        :math:`\Psi \rightarrow \Eta = \{\psi _i \} _{i=1}^n`.
        Here, a transformation of the likelihood parameters is modelled and
        ``True`` is returned.
        """
        return self._population_model.transforms_individual_parameters()


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
    def _compute_pointwise_ll(mean, std, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        log_likelihood = \
            - np.log(2 * np.pi * std**2) / 2 \
            - (observations - mean) ** 2 / (2 * std**2) \
            - np.log(1 - math.erf(-mean/std/math.sqrt(2))) + np.log(2)

        # If score evaluates to NaN, return -infinity
        mask = np.isnan(log_likelihood)
        if np.any(mask):
            log_likelihood[mask] = -np.inf
            return log_likelihood

        return log_likelihood

    @staticmethod
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
                'A truncated Gaussian distribution only accepts strictly '
                'positive means and standard deviations.')

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


def _norm_cdf(x):  # pragma: no cover
    """
    Returns the cumulative distribution function value of a standard normal
    Gaussian distribtion.
    """
    return 0.5 * (1 + math.erf(x/math.sqrt(2)))


def _norm_pdf(x):  # pragma: no cover
    """
    Returns the probability density function value of a standard normal
    Gaussian distribtion.
    """
    return math.exp(-x**2/2) / math.sqrt(2 * math.pi)
