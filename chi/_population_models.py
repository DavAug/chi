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
from scipy.special import erf

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
        self._needs_covariates = False

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
        :rtype: Tuple[float, np.ndarray of shape (n * n_dim + p,)]
        """
        raise NotImplementedError

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        return []

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        return copy.copy(self._dim_names)

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        raise NotImplementedError

    def n_dim(self):
        """
        Returns the dimensionality of the population model.
        """
        return self._n_dim

    def needs_covariates(self):
        """
        A boolean flag indicating whether the population model is conditionally
        defined on covariates.
        """
        return self._needs_covariates

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

    def n_covariates(self):
        """
        Returns the number of covariates.
        """
        return 0

    def n_ids(self):
        """
        Returns the number of modelled individuals.

        If the behaviour of the population model does not change with the
        number of modelled individuals 0 is returned.
        """
        return 0

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

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
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

    def set_n_ids(self, n_ids):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        return None

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        raise NotImplementedError


class ComposedPopulationModel(PopulationModel):
    r"""
    A multi-dimensional composed of mutliple population models.

    A :class:`ComposedPopulationModel` assumes that its constituent population
    models are independent. The probability density function of the composed
    population model is therefore given by the product of the probability
    density functions.

    For constituent population models
    :math:`p(\psi _1 | \theta _1), \ldots , p(\psi _n | \theta _n)`, the
    probability density function of the composed population model is given by

    .. math::
        p(\psi _1, \ldots , \psi _n | \theta _1, \ldots , \theta _n) =
            \prod _{k=1}^n p(\psi _k | \theta _k) .

    Extends :class:`chi.PopulationModel`.

    :param population_models: A list of population models.
    :type population_models: List[chi.PopulationModel]
    """
    def __init__(self, population_models):
        super(PopulationModel, self).__init__()
        # Check inputs
        for pop_model in population_models:
            if not isinstance(pop_model, chi.PopulationModel):
                raise TypeError(
                    'The population models have to be instances of '
                    'chi.PopulationModel.')

        # Check that number of modelled individuals is compatible
        n_ids = 0
        for pop_model in population_models:
            if (n_ids > 0) and (pop_model.n_ids() > 0) and (
                    n_ids != pop_model.n_ids()):
                raise ValueError(
                    'All population models must model the same number of '
                    'individuals.')
            n_ids = n_ids if n_ids > 0 else pop_model.n_ids()
        self._population_models = population_models
        self._n_ids = n_ids

        # Get properties of population models
        n_dim = 0
        n_parameters = 0
        n_covariates = 0
        transforms_psi = []
        needs_covariates = False
        for idp, pop_model in enumerate(self._population_models):
            needs_cov = pop_model.needs_covariates()
            needs_covariates = needs_covariates | needs_cov
            if pop_model.transforms_individual_parameters():
                transforms_psi.append(
                    [idp, needs_cov, n_dim, n_parameters])
            if needs_cov:
                n_covariates += pop_model.n_covariates()
            n_dim += pop_model.n_dim()
            n_parameters += pop_model.n_parameters()

        self._n_dim = n_dim
        self._n_parameters = n_parameters
        self._n_covariates = n_covariates
        self._transforms_psi = True if len(transforms_psi) > 0 else False
        self._transform_psi_models = transforms_psi
        self._needs_covariates = needs_covariates

        # Make sure that models have unique parameter names
        # (if not enumerate dimensions to make them unique in most cases)
        names = self.get_parameter_names()
        if len(np.unique(names)) != len(names):
            dim_names = [
                'Dim. %d' % (dim_id + 1) for dim_id in range(self._n_dim)]
            self.set_dim_names(dim_names)

    def compute_individual_parameters(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi`.

        If the model does not transform the bottom parameters, i.e.
        :math:`\eta = \psi`, the input :math:`\eta` are returned.

        If the population model does not use covariates, the covariate input
        is ignored.

        If the population model uses covariates, the covariates of the
        constituent population models are expected to be concatinated in the
        order of the consitutent models. The order of the covariates can be
        checked with :meth:`get_covariate_names`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (n_parameters,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of shape (n_ids, n_dim)
        :param covariates: Individual covariates :math:`\chi`.
        :type covariates: np.ndarray of shape (n_ids, n_cov)
        :returns: Individual parameters :math:`\psi`.
        :rtype: np.ndarray of shape (n_ids, n_dim)
        """
        if not self._transforms_psi:
            return eta

        current_cov = 0
        psis = np.empty(shape=eta.shape)
        psis[:, :] = eta[:, :]
        for info in self._transform_psi_models:
            idp, needs_cov, current_dim, current_parameters = info
            pop_model = self._population_models[idp]

            # Get covariates
            cov = None
            if needs_cov:
                end_cov = current_cov+pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            # Transform parameters
            # NOTE: This only works because CovariatePopulationModel can only
            # be 1-dimensional. Should they be extended to multiple dimensions
            # we need to start slicing the dimensions.
            end_parameters = current_parameters + pop_model.n_parameters()
            psis[:, current_dim] = \
                pop_model.compute_individual_parameters(
                    parameters[current_parameters:end_parameters],
                    eta[:, current_dim], cov)

        return psis

    def compute_individual_sensitivities(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi` and their sensitivities
        with respect to the model parameters :math:`\vartheta` and the relevant
        fluctuations :math:`\eta`.

        If the model does not transform the bottom parameters, i.e.
        :math:`\eta = \psi`, the input :math:`\eta` are returned.

        If the population model does not use covariates, the covariate input
        is ignored.

        If the population model uses covariates, the covariates of the
        constituent population models are expected to be concatinated in the
        order of the consitutent models. The order of the covariates can be
        checked with :meth:`get_covariate_names`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (n_parameters,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of shape (n_ids, n_dim)
        :param covariates: Individual covariates :math:`\chi`.
        :type covariates: np.ndarray of shape (n_ids, n_cov)
        :returns: Individual parameters :math:`\psi` and sensitivities
            (:math:`\partial _{\eta} \psi` ,
            :math:`\partial _{\vartheta _1} \psi`, :math:`\ldots`,
            :math:`\partial _{\vartheta _p} \psi`).
        :rtype: Tuple[np.ndarray, np.ndarray] of shapes (n_ids, n_dim) and
            (1 + n_parameters, n_ids, n_dim)
        """
        sensitivities = np.zeros(
            shape=(1 + self._n_parameters, len(eta), self._n_dim))
        sensitivities[0] = 1
        if not self._transforms_psi:
            return eta, sensitivities

        current_cov = 0
        psis = np.empty(shape=eta.shape)
        psis[:, :] = eta[:, :]
        for info in self._transform_psi_models:
            idp, needs_cov, current_dim, current_parameters = info
            pop_model = self._population_models[idp]

            # Get covariates
            cov = None
            if needs_cov:
                end_cov = current_cov+pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            # Transform parameters
            # NOTE: This only works because CovariatePopulationModel can only
            # be 1-dimensional. Should they be extended to multiple dimensions
            # we need to start slicing the dimensions.
            end_parameters = current_parameters + pop_model.n_parameters()
            psi, sens = \
                pop_model.compute_individual_sensitivities(
                    parameters[current_parameters:end_parameters],
                    eta[:, current_dim], cov)

            psis[:, current_dim] = psi
            sensitivities[0, :, current_dim] = sens[0]
            sensitivities[
                1+current_parameters:1+end_parameters, :, current_dim
            ] = sens[1:]

        return psis, sensitivities

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
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)

        score = 0
        current_dim = 0
        current_param = 0
        for pop_model in self._population_models:
            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()
            score += pop_model.compute_log_likelihood(
                parameters=parameters[current_param:end_param],
                observations=observations[:, current_dim:end_dim]
            )
            current_dim = end_dim
            current_param = end_param

        return score

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

        The sensitivities are returned as a 1-dimensional array

        ..math::
            (\psi _1, \ldots , \psi _n, \theta _1, \dlots , \theta _k).

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihood and its sensitivity to individual parameters
            as well as population parameters.
        :rtype: Tuple[float, np.ndarray of shape (n * n_dim + p,)]
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)

        score = 0
        n_ids = len(observations)
        n_bottom = n_ids*self._n_dim
        bottom_sens = np.empty(shape=(n_ids, self._n_dim))
        sensitivities = np.empty(shape=n_bottom+self._n_parameters)
        current_dim = 0
        current_param = 0
        for pop_model in self._population_models:
            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()
            s, sens = pop_model.compute_sensitivities(
                parameters=parameters[current_param:end_param],
                observations=observations[:, current_dim:end_dim])

            # Add score and sensitivities
            score += s
            bottom_sens[:, current_dim:end_dim] = sens[
                :n_ids*pop_model.n_dim()].reshape(n_ids, pop_model.n_dim())
            sensitivities[
                    n_bottom+current_param:
                    n_bottom+end_param] = sens[n_ids*pop_model.n_dim():]

            current_dim = end_dim
            current_param = end_param

        # Add bottom sensitivitities
        # (Final order psi_1, ..., psi_n, theta_1, ..., theta_k)
        sensitivities[:n_bottom] = bottom_sens.flatten()

        return score, sensitivities

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        names = []
        for pop_model in self._population_models:
            if pop_model.needs_covariates():
                names += pop_model.get_covariate_names()
        return names

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        names = []
        for pop_model in self._population_models:
            names += pop_model.get_dim_names()

        return names

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        names = []
        for pop_model in self._population_models:
            names += pop_model.get_parameter_names(exclude_dim_names)

        return names

    def get_population_models(self):
        """
        Returns the constituent population models.
        """
        return self._population_models

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
        n_bottom, n_top = 0, 0
        for pop_model in self._population_models:
            n_b, n_t = pop_model.n_hierarchical_parameters(n_ids)
            n_bottom += n_b
            n_top += n_t

        return n_bottom, n_top

    def n_covariates(self):
        """
        Returns the number of covariates.
        """
        return self._n_covariates

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

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
        :class:`CovariatePopulationModel`, transform the parameters to a
        latent representation
        :math:`\Psi \rightarrow \{\eta _i \} _{i=1}^n`.
        Here, a transformation of the likelihood parameters is modelled and
        ``True`` is returned.
        """
        return self._transforms_psi

    def sample(self, parameters, n_samples=None, seed=None, covariates=None):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If the model does not depend on covariates the ``covariate`` input is
        ignored.

        If the population model uses covariates, the covariates of the
        constituent population models are expected to be concatinated in the
        order of the consitutent models. The order of the covariates can be
        checked with :meth:`get_covariate_names`.

        :param parameters: Values of the model parameters.
        :type parameters: List, np.ndarray of shape (n_parameters,)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, np.random.Generator, optional
        :param covariates: Values for the covariates. If ``None``, default
            is assumed defined by the :class:`CovariateModel`.
        :type covariates: List, np.ndarray of shape (n_cov,)
        :returns: Samples from population model conditional on covariates.
        :rtype: np.ndarray of shape (n_samples, n_dim)
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)
        samples = np.empty(shape=sample_shape)

        # Transform seed to random number generator, so all models use the same
        # seed
        rng = np.random.default_rng(seed=seed)

        # Sample from constituent population models
        current_dim = 0
        current_param = 0
        current_cov = 0
        for pop_model in self._population_models:
            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()

            # Get covariates
            cov = None
            if pop_model.needs_covariates():
                cov = covariates[:, current_cov:pop_model.n_covariates()]
                current_cov += pop_model.n_covariates()

            # Sample bottom-level parameters
            samples[:, current_dim:end_dim] = pop_model.sample(
                    parameters=parameters[current_param:end_param],
                    n_samples=n_samples,
                    seed=rng,
                    covariates=cov)
            current_dim = end_dim
            current_param = end_param

        return samples

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
            # Reset dimension names
            for pop_model in self._population_models:
                pop_model.set_dim_names()
            return None

        if len(names) != self._n_dim:
            raise ValueError(
                'Length of names does not match the number of dimensions.')

        # Set dimension names
        names = [str(label) for label in names]
        current_dim = 0
        for pop_model in self._population_models:
            end_dim = current_dim + pop_model.n_dim()
            pop_model.set_dim_names(names[current_dim:end_dim])
            current_dim = end_dim

    def set_n_ids(self, n_ids):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        # Check cheap option first: Behaviour is not changed by input
        n_ids = int(n_ids)
        if (self._n_ids == 0) or (n_ids == self._n_ids):
            return None

        n_parameters = 0
        for pop_model in self._population_models:
            pop_model.set_n_ids(n_ids)
            n_parameters += pop_model.n_parameters()

        # Update n_ids and n_parameters
        self._n_ids = n_ids
        self._n_parameters = n_parameters

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset parameter names
            for pop_model in self._population_models:
                pop_model.set_parameter_names()
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

        # Set parameter names
        names = [str(label) for label in names]
        current_param = 0
        for pop_model in self._population_models:
            end_param = current_param + pop_model.n_parameters()
            pop_model.set_parameter_names(names[current_param:end_param])
            current_param = end_param


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
    :param dim_names: Name of dimensions.
    :type dim_names: List[str], optional
    """
    def __init__(self, population_model, covariate_model, dim_names=None):
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
        if population_model.n_dim() != 1:
            raise ValueError(
                'Only 1-dimensional population models are currently supported '
                'as inputs. If you want to construct multi-dimensional '
                'covariate models, use chi.ComposedPopulationModel together '
                'with 1-dimensional covariate models.')

        # Check compatibility of population model with covariate model
        covariate_model.check_compatibility(population_model)

        # Remember models
        self._population_model = population_model
        self._covariate_model = covariate_model

        # Set transform psis to true
        self._transforms_psi = True
        self._needs_covariates = \
            True if self._covariate_model.n_covariates() > 0 else False
        self._n_dim = self._population_model.n_dim()
        if (not dim_names) or (len(dim_names) != 1):
            dim_names = self._population_model.get_dim_names()
        self._dim_names = dim_names

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
        :type covariates: np.ndarray of shape (n, c)
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
        raise NotImplementedError
        # # Compute population parameters
        # parameters = self._covariate_model.compute_population_parameters(
        #     parameters)

        # # Compute log-likelihood
        # score = self._population_model.compute_pointwise_ll(
        #     parameters, observations)

        # return score

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

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        names = self._covariate_model.get_parameter_names()
        if exclude_dim_names:
            return names

        # Append dim name to model parameters
        names = [name + ' ' + self._dim_names[0] for name in names]
        return names

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
        :rtype: np.ndarray of shape (n_samples, n_dims)
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

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
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
        self._parameter_names = ['Mean'] * self._n_dim + ['Std.'] * self._n_dim

    @staticmethod
    def _compute_log_likelihood(mus, vars, observations):  # pragma: no cover
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
            return -np.inf, np.full(
                shape=(n_ids + 2) * self._n_dim, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = (mus - psi) / vars

        # Compute sensitivities w.r.t. parameters
        dmus = np.sum(psi - mus, axis=0) / vars
        dstd = (-n_ids + np.sum((psi - mus)**2, axis=0) / vars) / np.sqrt(vars)

        # Collect sensitivities
        # ([psi_1 all dim, ..., psi_n all dim, mu dim 1, ..., mu dim d,
        # sigma dim 1, ..., sigma dim d])
        sensitivities = np.concatenate((
            dpsi.flatten(), np.hstack([dmus, dstd]).flatten()))

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
            vector-valued. The parameters can then either be defined as a
            one-dimensional array or a matrix.
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
            n_ids = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_ids, self._n_dim)

        # Parse parameters
        mus = parameters[0]
        sigmas = parameters[1]
        vars = sigmas**2

        if np.any(sigmas <= 0):
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
        :rtype: Tuple[float, np.ndarray of shape (n * n_dim + p,)]
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

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

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

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
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

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = [
                'Mean'] * self._n_dim + ['Std.'] * self._n_dim
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

    .. note::
        Heterogeneous population models are special: the number of parameters
        depends on the number of modelled individuals.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    :param n_ids: Number of modelled individuals.
    :type n_ids: int, optional
    """

    def __init__(self, n_dim=1, dim_names=None, n_ids=1):
        super(HeterogeneousModel, self).__init__(n_dim, dim_names)
        self._n_ids = 0  # This is a temporary dummy value
        self.set_n_ids(n_ids)

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population model parameters.

        A heterogenous population model is formally equivalent to a
        multi-dimensional delta-distribution, where each individual parameter
        is determined by a separate delta-distribution.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual, i.e. [:math:`\psi _1, \ldots , \psi _N`].
        :type observations: np.ndarray of shape (n_ids, n_dim)
        :returns: Log-likelihood of individual parameters and population
            parameters.
        :rtype: float
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            parameters = parameters.reshape(self._n_ids, self._n_dim)

        # Return -inf if any of the observations do not equal the heterogenous
        # parameters
        mask = observations != parameters
        if np.any(mask):
            return -np.inf

        # Otherwise return 0
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
        raise NotImplementedError

    def compute_sensitivities(self, parameters, observations):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the parameters and the observations.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape (n_ids, n_dim)
        :returns: Log-likelihood and its sensitivity to individual parameters
            as well as population parameters.
        :rtype: Tuple[float, np.ndarray of shape (n_ids * n_dim + p,)]
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            parameters = parameters.reshape(self._n_ids, self._n_dim)

        # Return -inf if any of the observations does not equal the
        # heterogenous parameters
        mask = observations != parameters
        if np.any(mask):
            return -np.inf, np.full(
                shape=2 * self._n_ids * self._n_dim, fill_value=np.inf)

        # Otherwise return 0
        return 0, np.zeros(shape=2 * self._n_ids * self._n_dim)

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

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

        return (0, n_ids * self._n_dim)

    def n_ids(self):
        """
        Returns the number of modelled individuals.

        If the behaviour of the population model does not change with the
        number of modelled individuals 0 is returned.
        """
        return self._n_ids

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        For ``n_samples > 1`` the samples are randomly drawn from the ``n_ids``
        individuals.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            parameters = parameters.reshape(self._n_ids, self._n_dim)

        # Randomly sample from n_ids
        ids = np.arange(self._n_ids)
        rng = np.random.default_rng(seed=seed)
        n_samples = n_samples if n_samples else 1
        parameters = parameters[
            rng.choice(ids, size=n_samples, replace=True)]

        return parameters

    def set_n_ids(self, n_ids):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        n_ids = int(n_ids)

        if n_ids < 1:
            raise ValueError(
                'The number of modelled individuals needs to be greater or '
                'equal to 1.')

        if n_ids == self._n_ids:
            return None

        self._n_ids = n_ids
        self._n_parameters = self._n_ids * self._n_dim
        self._parameter_names = []
        for _id in range(self._n_ids):
            self._parameter_names += ['ID %d' % (_id + 1)] * self._n_dim

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = []
            for _id in range(self._n_ids):
                self._parameter_names += ['ID %d' % (_id + 1)] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match the number of parameters.')

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

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    """

    def __init__(self, n_dim=1, dim_names=None):
        super(LogNormalModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = [
            'Log mean'] * self._n_dim + ['Log std.'] * self._n_dim

    @staticmethod
    def _compute_log_likelihood(mus, vars, observations):
        r"""
        Calculates the log-likelihood using.

        mus shape: (n_dim,)
        vars shape: (n_dim,)
        observations: (n_ids, n_dim)
        """
        # Compute log-likelihood score
        log_likelihood = - np.sum(
            np.log(2 * np.pi * vars) / 2 + np.log(observations)
            + (np.log(observations) - mus)**2 / 2 / vars)

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
            return -np.inf, np.full(
                shape=(n_ids + 2) * self._n_dim, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = - ((np.log(psi) - mus) / vars + 1) / psi

        # Copmute sensitivities w.r.t. parameters
        dmus = np.sum(np.log(psi) - mus, axis=0) / vars
        dstd = \
            (np.sum((np.log(psi) - mus)**2, axis=0) / vars - n_ids) \
            / np.sqrt(vars)

        # Collect sensitivities
        # ([psi_1 all dim, ..., psi_n all dim, mu dim 1, ..., mu dim d,
        # sigma dim 1, ..., sigma dim d])
        sensitivities = np.concatenate((
            dpsi.flatten(), np.hstack([dmus, dstd]).flatten()))

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

        :param parameters: Parameters of the population model, i.e.
            [:math:`\mu _{\text{log}}`, :math:`\sigma _{\text{log}}`].
            If the population model is
            multi-dimensional :math:`\mu _{\text{log}}` and
            :math:`\sigma _{\text{log}}` are expected to be
            vector-valued. The parameters can then either be defined as a
            one-dimensional array or a matrix.
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
            # The std. is strictly positive
            return -np.inf

        return self._compute_log_likelihood(mus, vars, observations)

    def compute_pointwise_ll(
            self, parameters, observations):  # pragma: no cover
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
        # TODO: Needs proper research to establish which pointwise
        # log-likelihood makes sense for hierarchical models.
        # Also needs to be adapted to match multi-dimensional API.
        raise NotImplementedError
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

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihood and its sensitivity to individual parameters
            as well as population parameters.
        :rtype: Tuple[float, np.ndarray of shape (n * n_dim + p,)]
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

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        """
        # Check input
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        mus = parameters[0]
        sigmas = parameters[1]
        if np.any(sigmas < 0):
            raise ValueError('The standard deviation cannot be negative.')

        # Compute mean and standard deviation
        mean = np.exp(mus + sigmas**2 / 2)
        std = np.sqrt(
            np.exp(2 * mus + sigmas**2) * (np.exp(sigmas**2) - 1))

        return np.vstack([mean, std])

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

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

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

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

        if np.any(sigmas <= 0):
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        # (Mean and sigma are the mean and standard deviation of
        # the log samples)
        rng = np.random.default_rng(seed=seed)
        samples = rng.lognormal(
            mean=mus, sigma=sigmas, size=sample_shape)

        return samples

    def set_parameter_names(self, names=None):
        r"""
        Sets the names of the population model parameters.

        The population parameter of a LogNormalModel are the population mean
        and standard deviation of the parameter :math:`\psi`.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = [
                'Log mean'] * self._n_dim + ['Log std.'] * self._n_dim
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

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    """

    def __init__(self, n_dim=1, dim_names=None):
        super(PooledModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = self._n_dim

        # Set default parameter names
        self._parameter_names = ['Pooled'] * self._n_dim

    def compute_log_likelihood(self, parameters, observations):
        r"""
        Returns the unnormalised log-likelihood score of the population model.

        A pooled population model is a delta-distribution centred at the
        population model parameter. As a result the log-likelihood score
        is 0, if all individual parameters are equal to the population
        parameter, and :math:`-\infty` otherwise.

        :param parameters: Parameters of the population model.
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
            parameters = parameters.reshape(1, self._n_dim)

        # Return -inf if any of the observations do not equal the pooled
        # parameter
        mask = observations != parameters
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

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual, i.e. [:math:`\psi _1, \ldots , \psi _N`].
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihoods for each individual parameter for population
            parameters.
        :rtype: np.ndarray of length (n, n_dim)
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            parameters = parameters.reshape(1, self._n_dim)

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        log_likelihood = np.zeros_like(observations, dtype=float)
        mask = observations != parameters
        log_likelihood[mask] = -np.inf

        return log_likelihood

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
        :rtype: Tuple[float, np.ndarray of shape (n * n_dim + p,)]
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            parameters = parameters.reshape(1, self._n_dim)

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        n_ids = len(observations)
        mask = observations != parameters
        if np.any(mask):
            return -np.inf, np.full(
                shape=(n_ids + 1) * self._n_dim, fill_value=np.inf)

        # Otherwise return 0
        return 0, np.zeros(shape=(n_ids + 1) * self._n_dim)

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

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

    def sample(self, parameters, n_samples=None, *args, **kwargs):
        r"""
        Returns random samples from the underlying population
        distribution.

        For a PooledModel the input top-level parameters are copied
        ``n_samples`` and are returned.

        The returned value is a NumPy array with shape ``(n_samples,)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        """
        parameters = np.asarray(parameters)
        if len(parameters.flatten()) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)

        # If only one sample is requested, return input parameters
        if n_samples is None:
            return parameters

        # If more samples are wanted, broadcast input parameter to shape
        # (n_samples, n_dim)
        sample_shape = (int(n_samples), self._n_dim)
        samples = np.broadcast_to(parameters, shape=sample_shape)
        return samples

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list with string-convertable entries of
            length :meth:`n_parameters`. If ``None``, parameter names are
            reset to defaults.
        :type names: List[str]
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Pooled'] * self._n_dim
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]


class ReducedPopulationModel(PopulationModel):
    """
    A class that can be used to permanently fix model parameters of a
    :class:`PopulationModel` instance.

    This may be useful to explore simplified versions of a model.

    Extends :class:`chi.PopulationModel`.

    :param population_model: A population model.
    :type population_model: chi.PopulationModel
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
        r"""
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,)
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual, i.e. [:math:`\psi _1, \ldots , \psi _N`].
        :type observations: np.ndarray of shape (n, n_dim)
        :returns: Log-likelihood of individual parameters and population
            parameters.
        :rtype: float
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
        # TODO: Needs proper research to establish which pointwise
        # log-likelihood makes sense for hierarchical models.
        # Also needs to be adapted to match multi-dimensional API.
        raise NotImplementedError
        # # Get fixed parameter values
        # if self._fixed_params_mask is not None:
        #     self._fixed_params_values[~self._fixed_params_mask] = parameters
        #     parameters = self._fixed_params_values

        # # Compute log-likelihood
        # scores = self._population_model.compute_pointwise_ll(
        #     parameters, observations)

        # return scores

    def compute_sensitivities(self, parameters, observations):
        """
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
        :rtype: Tuple[float, np.ndarray of shape (n * n_dim + p,)]
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
        mask = np.ones(n_obs * self._n_dim + self._n_parameters, dtype=bool)
        mask[-self._n_parameters:] = ~self._fixed_params_mask

        return score, sensitivities[mask]

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        :param name_value_dict: A dictionary with model parameter names as
            keys, and parameter values as values.
        :type name_value_dict: Dict[str:float]
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
        for index, name in enumerate(
                self._population_model.get_parameter_names()):
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
        return self._population_model.get_covariate_names()

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.
        """
        names = self._population_model.get_parameter_names(exclude_dim_names)

        # Remove fixed model parameters
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
        return self._population_model.n_covariates()

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
        sample = self._population_model.sample(
            parameters, n_samples, seed, covariates, return_psi)

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

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    """
    def __init__(self, n_dim=1, dim_names=None):
        super(TruncatedGaussianModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = ['Mu'] * self._n_dim + ['Sigma'] * self._n_dim

    @staticmethod
    def _compute_log_likelihood(mus, sigmas, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood using numba speed up.

        We are using the relationship between the Gaussian CDF and the
        error function

        ..math::
            Phi(x) = (1 + erf(x/sqrt(2))) / 2

        mus shape: (n_dim,)
        sigmas shape: (n_dim,)
        observations: (n_ids, n_dim)
        """
        # Return infinity if any psis are negative
        if np.any(observations < 0):
            return -np.inf

        # Compute log-likelihood score
        log_likelihood = - np.sum(
            np.log(2 * np.pi * sigmas**2) / 2
            + (observations - mus) ** 2 / (2 * sigmas**2)
            + np.log(1 - _norm_cdf(-mus/sigmas)))

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

    def _compute_sensitivities(self, mus, sigmas, psi):  # pragma: no cover
        r"""
        Calculates the log-likelihood and its sensitivities.

        Expects:
        mus shape: (n_dim,)
        sigmas shape: (n_dim,)
        observations: (n_ids, n_dim)

        Returns:
        log_likelihood: float
        sensitivities: np.ndarray of shape (n_obs + 2,)
        """
        # Compute log-likelihood score
        log_likelihood = self._compute_log_likelihood(mus, sigmas, psi)

        if np.isinf(log_likelihood):
            n_obs = len(psi)
            return -np.inf, np.full(shape=n_obs + 2, fill_value=np.inf)

        # Compute sensitivities w.r.t. observations (psi)
        dpsi = (mus - psi) / sigmas**2

        # Copmute sensitivities w.r.t. parameters
        dmus = np.sum((
            (psi - mus) / sigmas - _norm_pdf(mus/sigmas)
            / (1 - _norm_cdf(-mus/sigmas))
            ) / sigmas, axis=0)
        dsigmas = np.sum(
            -1 + (psi - mus)**2 / sigmas**2 + _norm_pdf(mus/sigmas) * mus
            / sigmas / (1 - _norm_cdf(-mus/sigmas)), axis=0) / sigmas

        # Collect sensitivities
        # ([psi_1 all dim, ..., psi_n all dim, mu dim 1, ..., mu dim d,
        # sigma dim 1, ..., sigma dim d])
        sensitivities = np.concatenate((
            dpsi.flatten(), np.hstack([dmus, dsigmas]).flatten()))

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

        :param parameters: Parameters of the population model, i.e.
            [:math:`\mu`, :math:`\sigma`]. If the population model is
            multi-dimensional :math:`\mu` and :math:`\sigma` are expected to be
            vector-valued. The parameters can then either be defined as a
            one-dimensional array or a matrix.
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
            # Gaussians are only defined for positive sigmas.
            return -np.inf

        return self._compute_log_likelihood(mus, sigmas, observations)

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
        # TODO: Needs proper research to establish which pointwise
        # log-likelihood makes sense for hierarchical models.
        # Also needs to be adapted to match multi-dimensional API.
        raise NotImplementedError
        # observations = np.asarray(observations)
        # mean, std = parameters

        # if (mean <= 0) or (std <= 0):
        #     # The mean and std. of the Gaussian distribution are
        #     # strictly positive if truncated at zero
        #     return np.full(shape=len(observations), fill_value=-np.inf)

        # return self._compute_pointwise_ll(mean, std, observations)

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
        :rtype: Tuple[float, np.ndarray of shape (n * n_dim + p,)]
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
            # Gaussians are only defined for positive sigmas.
            n_obs = len(observations)
            return -np.inf, np.full(
                shape=(n_obs + 2, self._n_dim), fill_value=np.inf)

        return self._compute_sensitivities(mus, sigmas, observations)

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

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape (p,) or (p_per_dim, n_dim)
        """
        # Check input
        parameters = np.asarray(parameters)
        if parameters.ndim != 2:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        mus = parameters[0]
        sigmas = parameters[1]
        if np.any(sigmas < 0):
            raise ValueError('The standard deviation cannot be negative.')

        # Compute mean and standard deviation
        mean = \
            mus + sigmas * norm.pdf(mus/sigmas) / (1 - norm.cdf(-mus/sigmas))
        std = np.sqrt(
            sigmas**2 * (
                1 -
                mus / sigmas * norm.pdf(mus/sigmas)
                / (1 - norm.cdf(-mus/sigmas))
                - (norm.pdf(mus/sigmas) / (1 - norm.cdf(-mus/sigmas)))**2)
            )

        return [mean, std]

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the name of the the population model parameters. If name were
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        if exclude_dim_names:
            return copy.copy(self._parameter_names)

        # Append dimension names
        names = []
        for name_id, name in enumerate(self._parameter_names):
            current_dim = name_id % self._n_dim
            names += [name + ' ' + self._dim_names[current_dim]]

        return names

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

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        r"""
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples,)``.

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
                'A truncated Gaussian distribution only accepts strictly '
                'positive standard deviations.')

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
            a=0, b=np.inf, loc=mus, scale=sigmas, size=sample_shape)

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
            self._parameter_names = [
                'Mu'] * self._n_dim + ['Sigma'] * self._n_dim
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
    return 0.5 * (1 + erf(x/np.sqrt(2)))


def _norm_pdf(x):  # pragma: no cover
    """
    Returns the probability density function value of a standard normal
    Gaussian distribtion.
    """
    return np.exp(-x**2/2) / np.sqrt(2 * np.pi)
