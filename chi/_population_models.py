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

    Population models can be multi-dimensional, but unless explicitly specfied
    in the model description, the dimensions of the model are modelled
    independently.

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
        self._n_hierarchical_dim = self._n_dim
        self._special_dims = []
        self._n_pooled_dims = 0
        self._n_hetero_dims = 0
        self._n_covariates = 0
        self._n_ids = 1

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

    def _shape(self, score, dpsi, dtheta, reduce, flattened):
        """
        Returns the score, dpsi and dtheta.

        If reduce is True, dpsi and dtheta are flattened according to the
        hierarchical ordering.

        If reduce is False, dpsi and dtheta are returned separately.

        If flattened is True, dtheta is flattened. If flattened is False,
        dtheta is returned in shape (n_ids, n_param_per_dim, n_dim)
        """
        if reduce or flattened:
            # Sum contributions across individuals and flatten
            dtheta = np.sum(dtheta, axis=0).flatten()
        if reduce:
            return score, np.hstack((dpsi.flatten(), dtheta))

        return score, dpsi, dtheta

    def compute_individual_parameters(
            self, parameters, eta, return_eta=False, *args, **kwargs):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids * n_dim)`` or
            ``(n_ids, n_dim)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        :param return_eta: A boolean flag indicating whether eta is returned
            regardless of whether the parametrisation is centered or not.
        :type return_eta: boolean, optional
        """
        raise NotImplementedError

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        raise NotImplementedError

    def compute_pointwise_ll(self, parameters, observations,  *args, **kwargs):
        """
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

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, reduce=False,
            *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
        :type reduce: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
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

    def get_special_dims(self):
        r"""
        Returns information on pooled and heterogeneously modelled dimensions.

        Returns a tuple with 3 entries:
            1. A list of lists, each entry containing 1. the start and end
                dimension of the special dimensions; 2. the associated
                start and end index of the model parameters; 3. and a boolean
                indicating whether the dimension is pooled (``True``) or
                heterogeneous (``False``).
            2. The number of pooled dimensions.
            3. The number of heterogeneous dimensions.
        """
        return self._special_dims, self._n_pooled_dims, self._n_hetero_dims

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the population model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        raise NotImplementedError

    def n_covariates(self):
        """
        Returns the number of covariates.
        """
        return self._n_covariates

    def n_dim(self):
        """
        Returns the dimensionality of the population model.
        """
        return self._n_dim

    def n_hierarchical_dim(self):
        """
        Returns the number of parameter dimensions whose samples are not
        deterministically defined by the population parameters.

        I.e. the number of dimensions minus the number of pooled and
        heterogeneously modelled dimensions.
        """
        return self._n_hierarchical_dim

    def n_hierarchical_parameters(self, n_ids):
        """
        Returns a tuple of the number of individual parameters and the number
        of population parameters that this model expects in context of a
        :class:`HierarchicalLogLikelihood`, when ``n_ids`` individuals are
        modelled.

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        raise NotImplementedError

    def set_covariate_names(self, names=None):
        """
        Sets the names of the covariates.

        If the model has no covariates, input is ignored.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List[str]
        """
        # Default is that models do not have covariates.
        return None

    def set_dim_names(self, names=None):
        """
        Sets the names of the population model dimensions.

        :param names: A list of dimension names. If ``None``, dimension names
            are reset to defaults.
        :type names: List[str], optional
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
        n_ids = int(n_ids)
        if n_ids < 1:
            raise ValueError(
                'n_ids is invalid. The number of individuals has to be '
                'positive.')
        self._n_ids = n_ids

    def set_parameter_names(self, names=None):
        """
        Sets the names of the population model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List[str]
        """
        raise NotImplementedError


class ComposedPopulationModel(PopulationModel):
    r"""
    A multi-dimensional population model composed of mutliple population
    models.

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
        n_ids = 1
        for pop_model in population_models:
            if (n_ids > 1) and (pop_model.n_ids() > 1) and (
                    n_ids != pop_model.n_ids()):
                raise ValueError(
                    'All population models must model the same number of '
                    'individuals.')
            n_ids = n_ids if n_ids > 1 else pop_model.n_ids()
        self._population_models = population_models
        self._n_ids = n_ids
        self._n_bottom, self._n_top = self.n_hierarchical_parameters(
            self._n_ids)

        # Get properties of population models
        self._set_population_model_properties()

        # Make sure that models have unique parameter names
        # (if not enumerate dimensions to make them unique in most cases)
        names = self.get_parameter_names()
        if len(np.unique(names)) != len(names):
            dim_names = [
                'Dim. %d' % (dim_id + 1) for dim_id in range(self._n_dim)]
            self.set_dim_names(dim_names)

    def _compute_reduced_sensitivities(
            self, parameters, observations, dlogp_dpsi, covariates, *args,
            **kwargs):
        """
        Returns the likelihood of the population model and its sensitivities
        with respect to the individual-level parameters and the
        population-level parameters.

        Counterpart to _compute_sensitivities, but instead of returning the
        sensitivities with respect to the individual-level parameters and the
        population-level parameters separately, only the sensitivies of the
        parameters exposed in a hierarchical inference settings are returned.
        This affects predominantly pooled and heterogenous dimensions.
        """
        score = 0
        dscore = np.empty(shape=self._n_bottom + self._n_top)
        dpsi = np.empty(shape=(self._n_ids, self._n_hierarchical_dim))

        cov = None
        dlp_dpsi = None
        current_cov = 0
        current_dim = 0
        current_top = 0
        current_hdim = 0
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov
            # Get dlogp/dpsi
            n_dim = pop_model.n_dim()
            end_dim = current_dim + n_dim
            if dlogp_dpsi is not None:
                dlp_dpsi = dlogp_dpsi[:, current_dim:end_dim]

            n_b, n_t = pop_model.n_hierarchical_parameters(self._n_ids)
            end_top = current_top + n_t
            s, ds = pop_model.compute_sensitivities(
                parameters=parameters[current_top:end_top],
                observations=observations[:, current_dim:end_dim],
                covariates=cov,
                dlogp_dpsi=dlp_dpsi,
                reduce=True)

            # Add score and top sensitivities
            score += s
            dscore[self._n_bottom+current_top:self._n_bottom+end_top] = \
                ds[n_b:]

            # Collect bottom-level sensitivities
            if n_b > 0:
                end_hdim = current_hdim + n_dim
                dpsi[:, current_hdim:end_hdim] = ds[:n_b].reshape(
                    self._n_ids, n_dim)
                current_hdim = end_hdim

            current_dim = end_dim
            current_top = end_top

        # Add bottom-level sensitivities
        dscore[:self._n_bottom] = np.array(dpsi).flatten()

        return score, dscore

    def _compute_sensitivities(
            self, parameters, observations, dlogp_dpsi, covariates, *args,
            **kwargs):
        """
        Returns the likelihood of the population model and its sensitivities
        with respect to the individual-level parameters and the
        population-level parameters.
        """
        score = 0
        n_ids = len(observations)
        dpsi = np.zeros(shape=(n_ids, self._n_dim))
        dtheta = np.empty(shape=self._n_parameters)

        cov = None
        dlp_dpsi = None
        current_cov = 0
        current_dim = 0
        current_param = 0
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov
            # Get dlogp/dpsi
            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()
            if dlogp_dpsi is not None:
                dlp_dpsi = dlogp_dpsi[:, current_dim:end_dim]

            s, dp, dt = pop_model.compute_sensitivities(
                parameters=parameters[current_param:end_param],
                observations=observations[:, current_dim:end_dim],
                covariates=cov,
                dlogp_dpsi=dlp_dpsi)

            # Add score and sensitivities
            score += s
            dpsi[:, current_dim:end_dim] = dp
            dtheta[current_param:end_param] = dt

            current_dim = end_dim
            current_param = end_param

        return score, dpsi, dtheta

    def _set_population_model_properties(self):
        """
        Sets the properties of the population model based on the submodels.
        """
        n_dim = 0
        n_parameters = 0
        n_covariates = 0
        n_hierarchical_dim = 0
        special_dim = []
        n_pooled = 0
        n_hetero = 0
        for pop_model in self._population_models:
            s, p, h = pop_model.get_special_dims()
            n_pooled += p
            n_hetero += h
            for entry in s:
                # Need to shift dimensions and parameter index
                special_dim += [[
                    entry[0] + n_dim,
                    entry[1] + n_dim,
                    entry[2] + n_parameters,
                    entry[3] + n_parameters,
                    entry[4]
                ]]
            n_covariates += pop_model.n_covariates()
            n_dim += pop_model.n_dim()
            n_hierarchical_dim += pop_model.n_hierarchical_dim()
            n_parameters += pop_model.n_parameters()

        self._n_dim = n_dim
        self._n_parameters = n_parameters
        self._n_covariates = n_covariates
        self._n_hierarchical_dim = n_hierarchical_dim
        self._special_dims = special_dim
        self._n_pooled_dims = n_pooled
        self._n_hetero_dims = n_hetero

    def _shape_eta(self, eta):
        """
        Reshapes eta to numpy.ndarry of shape (n_ids, n_dim).
        """
        n_dim = len(eta) // self._n_ids
        eta = eta.reshape(self._n_ids, n_dim)
        if n_dim == self._n_dim:
            # There are no special dimensions and we can just reshape and
            # return
            return eta

        # Some dimensions have do be filled with dummies, because pooled
        # and heterogeneous dimensions are special and replace dummy values
        # later on.
        if (n_dim + self._n_pooled_dims + self._n_hetero_dims) != self._n_dim:
            raise ValueError(
                'eta is invalid. Eta cannot be reshaped into (n_ids, n_dim), '
                'even after taking pooled and heterogenuous dimensions into '
                'account.')
        start = 0
        shift = 0
        eta_prime = np.empty(shape=(self._n_ids, self._n_dim))
        for s in self._special_dims:
            end = s[0]
            eta_prime[:, start:end] = eta[:, start-shift:end-shift]
            start = s[1]
            shift += start - end

        # Fill trailing dimensions
        eta_prime[:, start:] = eta[:, start-shift:]

        return eta_prime

    def compute_individual_parameters(
            self, parameters, eta, covariates=None, return_eta=False):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        If the population model does not use covariates, the covariate input
        is ignored.

        If the population model uses covariates, the covariates of the
        constituent population models are expected to be concatinated in the
        order of the consitutent models. The order of the covariates can be
        checked with :meth:`get_covariate_names`.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids * n_dim)`` or
            ``(n_ids, n_dim)``
        :param covariates: Covariates of the individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :param return_eta: A boolean flag indicating whether eta is returned
            regardless of whether the parametrisation is centered or not.
        :type return_eta: boolean, optional
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        eta = np.asarray(eta)
        parameters = np.asarray(parameters)

        if eta.ndim == 1:
            eta = self._shape_eta(eta)

        # Compute individual parameters
        cov = None
        current_p = 0
        current_dim = 0
        current_cov = 0
        psis = np.empty(shape=eta.shape)
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            end_p = current_p + pop_model.n_parameters()
            end_dim = current_dim + pop_model.n_dim()
            psis[:, current_dim:end_dim] = \
                pop_model.compute_individual_parameters(
                    parameters=parameters[current_p:end_p],
                    eta=eta[:, current_dim:end_dim],
                    covariates=cov,
                    return_eta=return_eta)
            current_p = end_p
            current_dim = end_dim

        return psis

    def compute_log_likelihood(
            self, parameters, observations, covariates=None):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of the individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: float
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)

        score = 0
        cov = None
        current_dim = 0
        current_param = 0
        current_cov = 0
        for pop_model in self._population_models:
            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()
            score += pop_model.compute_log_likelihood(
                parameters=parameters[current_param:end_param],
                observations=observations[:, current_dim:end_dim],
                covariates=cov
            )
            current_dim = end_dim
            current_param = end_param

        return score

    def compute_pointwise_ll(self, parameters, observations):
        r"""
        Returns the pointwise log-likelihood of the model parameters for
        each observation.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p, n_dim)``
        :param observations: "Observations" of the individuals. Typically
            refers to the values of a mechanistic model parameter for each
            individual.
        :type observations: np.ndarray of shape ``(n, n_dim)``
        :returns: Log-likelihoods for each individual parameter for population
            parameters.
        :rtype: np.ndarray of length ``(n, n_dim)``
        """
        raise NotImplementedError

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, reduce=False,
            covariates=None, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
            ``reduce`` is prioritised over ``flattened``.
        :type reduce: bool, optional
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        parameters = np.asarray(parameters)

        if reduce:
            return self._compute_reduced_sensitivities(
                parameters, observations, dlogp_dpsi, covariates, *args,
                **kwargs
            )

        return self._compute_sensitivities(
            parameters, observations, dlogp_dpsi, covariates, *args, **kwargs)

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        names = []
        for pop_model in self._population_models:
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

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(
            self, parameters, n_samples=None, seed=None, covariates=None,
            *args, **kwargs):
        """
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
        :param covariates: Covariate values, specifying the sampled
            subpopulation.
        :type covariates: List, np.ndarray of shape ``(n_cov,)`` or
            ``(n_samples, n_cov)``, optional
        :returns: Samples from population model conditional on covariates.
        :rtype: np.ndarray of shape (n_samples, n_dim)
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')
        if (self._n_covariates > 0):
            covariates = np.asarray(covariates)
            if covariates.ndim == 1:
                covariates = covariates[np.newaxis, :]
            if covariates.shape[1] != self._n_covariates:
                raise ValueError(
                    'Provided covariates do not match the number of '
                    'covariates.')

        # Define shape of samples
        if n_samples is None:
            n_samples = 1
        sample_shape = (int(n_samples), self._n_dim)
        samples = np.empty(shape=sample_shape)

        # Transform seed to random number generator, so all models use the same
        # seed
        rng = np.random.default_rng(seed=seed)

        # Sample from constituent population models
        cov = None
        current_dim = 0
        current_param = 0
        current_cov = 0
        for pop_model in self._population_models:
            end_dim = current_dim + pop_model.n_dim()
            end_param = current_param + pop_model.n_parameters()

            # Get covariates
            if self._n_covariates > 0:
                end_cov = current_cov + pop_model.n_covariates()
                cov = covariates[:, current_cov:end_cov]
                current_cov = end_cov

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
        if n_ids == self._n_ids:
            return None

        for pop_model in self._population_models:
            pop_model.set_n_ids(n_ids)

        # Update n_ids and model properties
        self._n_ids = n_ids
        self._set_population_model_properties()
        self._n_bottom, self._n_top = self.n_hierarchical_parameters(
            self._n_ids)

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
    A population model that models the parameters across individuals
    conditional on covariates of the inter-individual variability.

    A covariate population model partitions a population into subpopulations
    which are characterised by covariates :math:`\chi`. The inter-individual
    variability within a subpopulation is modelled by a non-covariate
    population model

    .. math::
        p(\psi | \theta, \chi) = p(\psi | \vartheta (\theta, \chi)),

    where :math:`\vartheta` are the population parameters of the subpopulation
    which depend on global population parameters :math:`\theta` and the
    covariates :math:`\chi`.

    The ``population_model`` input defines the non-covariate population model
    for the subpopulations :math:`p(\psi | \vartheta )` and the
    ``covariate_model`` defines the relationship between the subpopulations and
    the covariates :math:`\vartheta (\theta, \chi)`.

    Extends :class:`PopulationModel`.

    :param population_model: Defines the distribution of the subpopulations.
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
        if isinstance(population_model, ComposedPopulationModel):
            raise TypeError(
                'The population model cannot be an instance of a '
                'chi.ComposedPopulationModel. Please compose multiple '
                'covariate models instead.')
        if isinstance(population_model, ReducedPopulationModel):
            raise TypeError(
                'The population model cannot be an instance of a '
                'chi.ReducedPopulationModel. Please define a covariate '
                'population model before fixing parameters.')

        # Remember models
        self._population_model = copy.deepcopy(population_model)
        self._covariate_model = copy.deepcopy(covariate_model)

        # Get properties
        self._n_dim = self._population_model.n_dim()
        self._n_pop = self._population_model.n_parameters()
        self._n_hierarchical_dim = self._population_model.n_hierarchical_dim()
        self._n_covariates = self._covariate_model.n_covariates()
        self._special_dims, self._n_pooled_dims, self._n_hetero_dims = \
            self._population_model.get_special_dims()

        # Set names and all parameters to be modelled by the covariate model
        n_cov = self._covariate_model.n_covariates()
        self._population_model.set_dim_names(dim_names)
        indices = []
        for dim_id in range(self._n_dim):
            for param_id in range(self._n_pop // self._n_dim):
                indices.append([param_id, dim_id])
        self._covariate_model.set_population_parameters(indices)
        names = []
        for name in self._population_model.get_parameter_names():
            names += [name] * n_cov
        self._covariate_model.set_parameter_names(names)

    def compute_individual_parameters(
            self, parameters, eta, covariates, return_eta=False):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids * n_dim)`` or
            ``(n_ids, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :param return_eta: A boolean flag indicating whether eta is returned
            regardless of whether the parametrisation is centered or not.
        :type return_eta: boolean, optional
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        # Split into covariate model parameters and population parameters
        parameters = np.asarray(parameters)
        pop_params = parameters[:self._n_pop]
        cov_params = parameters[self._n_pop:]

        # Reshape population parameters to (n_params_per_dim, n_dim)
        # NOTE: Assumes that all dimensions have the same number of parameters
        # TODO: Need to introduce a population model owned method that
        # transforms n_p to (n_p, n_d) for varying parameters across
        # dimensions.
        n_params_per_dim = self._n_pop // self._n_dim
        pop_params = pop_params.reshape(n_params_per_dim, self._n_dim)

        # Compute vartheta(theta, chi)
        parameters = self._covariate_model.compute_population_parameters(
            cov_params, pop_params, covariates)

        # Compute psi(eta, vartheta)
        psi = self._population_model.compute_individual_parameters(
            parameters, eta, return_eta=return_eta)

        return psi

    def compute_log_likelihood(self, parameters, observations, covariates):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: float
        """
        # Split into covariate model parameters and population parameters
        parameters = np.asarray(parameters)
        pop_params = parameters[:self._n_pop]
        cov_params = parameters[self._n_pop:]

        # Reshape population parameters to (n_params_per_dim, n_dim)
        n_params_per_dim = self._n_pop // self._n_dim
        pop_params = pop_params.reshape(n_params_per_dim, self._n_dim)

        # Compute vartheta(theta, chi)
        parameters = self._covariate_model.compute_population_parameters(
            cov_params, pop_params, covariates)

        # Compute log-likelihood
        score = self._population_model.compute_log_likelihood(
            parameters, observations)

        return score

    # def compute_pointwise_ll(self, parameters, observations):
    #     r"""
    #     Returns the pointwise log-likelihood of the model parameters for
    #     each observation.

    #     :param parameters: Values of the model parameters :math:`\vartheta`.
    #     :type parameters: List, np.ndarray of length (p,)
    #     :param observations: "Observations" of the individuals :math:`\eta`.
    #         Typically refers to the inter-individual fluctuations of the
    #         mechanistic model parameter.
    #     :type observations: List, np.ndarray of length (n,)
    #     :returns: Log-likelihoods of individual parameters for population
    #         parameters.
    #     :rtype: np.ndarray of length (n,)
    #     """
    #     raise NotImplementedError
    #     # # Compute population parameters
    #     # parameters = self._covariate_model.compute_population_parameters(
    #     #     parameters)

    #     # # Compute log-likelihood
    #     # score = self._population_model.compute_pointwise_ll(
    #     #     parameters, observations)

    #     # return score

    def compute_sensitivities(
            self, parameters, observations, covariates, dlogp_dpsi=None,
            reduce=False):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
            ``reduce`` is prioritised over ``flattened``.
        :type reduce: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        # Split into covariate model parameters and population parameters
        parameters = np.asarray(parameters)
        pop_params = parameters[:self._n_pop]
        cov_params = parameters[self._n_pop:]

        # Reshape population parameters to (n_params_per_dim, n_dim)
        n_params_per_dim = self._n_pop // self._n_dim
        pop_params = pop_params.reshape(n_params_per_dim, self._n_dim)

        # Compute vartheta(theta, chi) and dvartheta/dtheta
        parameters = self._covariate_model.compute_population_parameters(
            cov_params, pop_params, covariates)

        # Compute log-likelihood and sensitivities dscore/deta,
        # dscore/dvartheta
        score, dpsi, dvartheta = self._population_model.compute_sensitivities(
            parameters, observations, dlogp_dpsi=dlogp_dpsi,
            flattened=False)

        # Propagate sensitivities of score to population model parameters
        dpop, dcov = self._covariate_model.compute_sensitivities(
            cov_params, pop_params, covariates, dvartheta)
        dtheta = np.hstack([dpop, dcov])

        if reduce:
            return score, np.hstack((dpsi.flatten(), dtheta))

        return score, dpsi, dtheta

    def get_covariate_names(self):
        """
        Returns the names of the covariates. If name is
        not set, defaults are returned.
        """
        return self._covariate_model.get_covariate_names()

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        return self._population_model.get_dim_names()

    def get_parameter_names(self, exclude_dim_names=False):
        """
        Returns the names of the model parameters. If name is
        not set, defaults are returned.

        :param exclude_dim_names: A boolean flag that indicates whether the
            dimension name is appended to the parameter name.
        :type exclude_dim_names: bool, optional
        """
        names = self._population_model.get_parameter_names(exclude_dim_names)
        names += self._covariate_model.get_parameter_names()
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

        return (n_ids, self.n_parameters())

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        n_parameters = self._population_model.n_parameters()
        n_parameters += self._covariate_model.n_parameters()
        return n_parameters

    def sample(self, parameters, covariates, n_samples=None, seed=None):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        :param covariates: Covariate values, specifying the sampled
            subpopulation.
        :type covariates: List, np.ndarray of shape ``(n_cov,)`` or
            ``(n_samples, n_cov)``
        """
        covariates = np.array(covariates)
        if covariates.ndim == 1:
            covariates = covariates[np.newaxis, :]
        if covariates.shape[1] != self._n_covariates:
            raise ValueError(
                'Provided covariates do not match the number of covariates.')
        if n_samples is None:
            n_samples = 1
        n_samples = int(n_samples)
        covariates = np.broadcast_to(
            covariates, (n_samples, self._n_covariates))

        # Split parameters into covariate model parameters and population model
        # parameters
        parameters = np.asarray(parameters)
        pop_params = parameters[:self._n_pop]
        cov_params = parameters[self._n_pop:]

        # Reshape population parameters to (n_params_per_dim, n_dim)
        n_params_per_dim = self._n_pop // self._n_dim
        pop_params = pop_params.reshape(n_params_per_dim, self._n_dim)

        # Compute population parameters
        pop_params = self._covariate_model.compute_population_parameters(
            cov_params, pop_params, covariates)

        # Sample parameters from population model
        seed = np.random.default_rng(seed)
        psi = np.empty(shape=(n_samples, self._n_dim))
        for ids, params in enumerate(pop_params):
            psi[ids] = self._population_model.sample(
                params, n_samples=1, seed=seed)[0]

        return psi

    def set_covariate_names(self, names=None):
        """
        Sets the names of the covariates.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List
        """
        self._covariate_model.set_covariate_names(names)

    def set_dim_names(self, names=None):
        """
        Sets the names of the population model dimensions.

        Setting the dimension names overwrites the parameter names of the
        covariate model.

        :param names: A list of dimension names. If ``None``, dimension names
            are reset to defaults.
        :type names: List[str], optional
        """
        self._population_model.set_dim_names(names)

        # Get names of parameters affected by the covariate model
        names = self._population_model.get_parameter_names()
        names = np.array(names).reshape(
            self._n_pop // self._n_dim, self._n_dim)
        pidx, didx = self._covariate_model.get_set_population_parameters()
        names = names[pidx, didx]

        n = []
        for name in names:
            n += [name] * self._n_covariates
        self._covariate_model.set_parameter_names(n)

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
            self._population_model.set_parameter_names()
            self._covariate_model.set_parameter_names()
            return None

        self._population_model.set_parameter_names(names[:self._n_pop])
        self._covariate_model.set_parameter_names(names[self._n_pop:])

    def set_population_parameters(self, indices):
        """
        Sets the parameters of the population model that are transformed by the
        covariate model.

        Note that this influences the number of model parameters.

        :param indices: A list of parameter indices
            [param index, dim index].
        :type indices: List[Tuple[int, int]]
        """
        # Check that indices are in bounds
        indices = np.array(indices)
        upper = np.max(indices, axis=0)
        n_pop = self._n_pop // self._n_dim
        out_of_bounds = \
            (upper[0] >= n_pop) or (upper[1] >= self._n_dim) or \
            (np.min(indices) < 0)
        if out_of_bounds:
            raise IndexError('The provided indices are out of bounds.')
        self._covariate_model.set_population_parameters(indices)

        # Update parameter names
        names = np.array(self._population_model.get_parameter_names())
        names = names.reshape(n_pop, self._n_dim)[indices[:, 0], indices[:, 1]]
        n = []
        for name in names:
            n += [name] * self._n_covariates
        self._covariate_model.set_parameter_names(n)


class GaussianModel(PopulationModel):
    r"""
    A population model which models parameters across individuals
    with a Gaussian distribution.

    A Gaussian population model assumes that a model parameter
    :math:`\psi` varies across individuals such that :math:`\psi` is
    normally distributed in the population

    .. math::
        p(\psi |\mu, \sigma) =
        \frac{1}{\sqrt{2\pi} \sigma}
        \exp\left(-\frac{(\psi - \mu )^2}
        {2 \sigma ^2}\right).

    Here, :math:`\mu` and :math:`\sigma ^2` are the
    mean and variance of the Gaussian distribution.

    Any observed individual with parameter :math:`\psi _i` is
    assumed to be a realisation of the random variable :math:`\psi`.

    If ``centered = False`` the parametrisation is non-centered, i.e.

    .. math::
        \psi = \mu + \sigma \eta ,

    where :math:`\eta` models the inter-individual variability and is
    standard normally distributed.

    Extends :class:`PopulationModel`.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    :param centered: Boolean flag indicating whether parametrisation is
        centered or non-centered.
    :type centered: bool, optional
    """
    def __init__(self, n_dim=1, dim_names=None, centered=True):
        super(GaussianModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = ['Mean'] * self._n_dim + ['Std.'] * self._n_dim

        self._centered = bool(centered)

    def _compute_dpsi(self, sigma, observations):
        """
        Computes the partial derivatives of psi = mu + sigma eta w.r.t.
        eta, mu and sigma.

        sigma: (n_ids, n_dim)
        observations: (n_ids, n_dim)

        rtype: np.ndarray of shape (n_ids, n_dim),
            np.ndarray of shape (n_ids, 2, n_dim)
        """
        n_ids, n_dim = observations.shape
        dpsi_deta = sigma
        dpsi_dtheta = np.empty(shape=(n_ids, 2, n_dim))
        dpsi_dtheta[:, 0] = np.ones(shape=(n_ids, n_dim))
        dpsi_dtheta[:, 1] = observations
        return dpsi_deta, dpsi_dtheta

    @staticmethod
    def _compute_log_likelihood(mus, vars, observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood.

        mus shape: (n_ids, n_dim)
        vars shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)
        """
        # Compute log-likelihood score
        with np.errstate(divide='ignore'):
            log_likelihood = - np.sum(
                np.log(2 * np.pi * vars) / 2 + (observations - mus) ** 2
                / (2 * vars))

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    def _compute_non_centered_sensitivities(
            self, sigmas, observations, dlogp_dpsi):
        """
        Returns the log-likelihood and the sensitivities with respect to
        eta and theta.
        """
        # Copmute score
        zeros = np.zeros(shape=(1, self._n_dim))
        ones = np.ones(shape=(1, self._n_dim))
        score = self._compute_log_likelihood(zeros, ones, observations)

        # Compute sensitivities
        if dlogp_dpsi is None:
            dlogp_dpsi = np.zeros((1, self._n_dim))
        deta = self._compute_sensitivities(zeros, ones, observations)
        dpsi_deta, dpsi_dtheta = self._compute_dpsi(sigmas, observations)
        dlogp_deta = dlogp_dpsi * dpsi_deta + deta
        dlogp_dtheta = dlogp_dpsi[:, np.newaxis, :] * dpsi_dtheta

        return score, dlogp_deta, dlogp_dtheta

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
        mus shape: (n_ids, n_dim)
        vars shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)

        Returns:
        deta for non-centered of shape (n_ids, n_dim)

        and
        deta and dtheta for centered
        dtheta: np.ndarray of shape (n_ids, n_parameters, n_dim)
        """
        # Compute sensitivities w.r.t. observations (psi)
        with np.errstate(divide='ignore'):
            dpsi = (mus - psi) / vars
        if not self._centered:
            # Despite the naming, this is really deta
            return dpsi

        # Compute sensitivities w.r.t. parameters
        n_ids = len(psi)
        with np.errstate(divide='ignore'):
            dmus = (psi - mus) / vars
            dstd = (-1 + (psi - mus)**2 / vars) / np.sqrt(vars)

        # Collect sensitivities
        n_ids, n_dim = psi.shape
        dtheta = np.empty(shape=(n_ids, 2, n_dim))
        dtheta[:, 0] = dmus
        dtheta[:, 1] = dstd

        return dpsi, dtheta

    def compute_individual_parameters(
            self, parameters, eta, return_eta=False, *args, **kwargs):
        r"""
        Returns the individual parameters.

        If ``centered = True``, the model does not transform the parameters
        and ``eta`` is returned.

        If ``centered = False``, the individual parameters are defined as

        .. math::
            \psi = \mu + \sigma \eta,

        where :math:`\mu` and :math:`\sigma` are the model parameters and
        :math:`\eta` are the inter-individual fluctuations.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids * n_dim)`` or
            ``(n_ids, n_dim)``
        :param return_eta: A boolean flag indicating whether eta is returned
            regardless of whether the parametrisation is centered or not.
        :type return_eta: boolean, optional
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        eta = np.asarray(eta)
        if eta.ndim == 1:
            eta = eta.reshape(self._n_ids, self._n_dim)
        if self._centered or return_eta:
            return eta

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            n_parameters = parameters[np.newaxis, ...]

        mu = parameters[:, 0]
        sigma = parameters[:, 1]

        if np.any(sigma < 0):
            return np.full(shape=eta.shape, fill_value=np.nan)

        psi = mu + sigma * eta

        return psi

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        If ``centered = False``, the log-likelihood of the standard normal
        is returned. The contribution of the population parameters to the
        log-likelihood can be computed with the log-likelihood of the
        individual parameters, see :class:`chi.ErrorModel`.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        if not self._centered:
            mus = np.zeros(shape=(1, self._n_dim))
            vars = np.ones(shape=(1, self._n_dim))
            score = self._compute_log_likelihood(mus, vars, observations)
            return score

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]
        vars = sigmas**2

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution is strictly positive
            return -np.inf

        return self._compute_log_likelihood(mus, vars, observations)

    # def compute_pointwise_ll(
    #         self, parameters, observations):  # pragma: no cover
    #     r"""
    #     Returns the pointwise log-likelihood of the model parameters for
    #     each observation.

    #     The pointwise log-likelihood of a Gaussian distribution is
    #     the log-pdf evaluated at the observations

    #     .. math::
    #         L(\mu , \sigma | \psi _i) =
    #         \log p(\psi _i |
    #         \mu , \sigma ) ,

    #     where
    #     :math:`\psi _i` are the "observed" parameters :math:`\psi` from
    #     individual :math:`i`.

    #     Parameters
    #     ----------
    #     parameters
    #         An array-like object with the model parameter values, i.e.
    #         [:math:`\mu`, :math:`\sigma`].
    #     observations
    #         An array like object with the parameter values for the
    #         individuals.
    #     """
    #     # TODO: Needs proper research to establish which pointwise
    #     # log-likelihood makes sense for hierarchical models.
    #     # Also needs to be adapted to match multi-dimensional API.
    #     raise NotImplementedError
    #     observations = np.asarray(observations)
    #     mean, std = parameters
    #     var = std**2

    #     eps = 1E-6
    #     if (std <= 0) or (var <= eps):
    #         # The std. of the Gaussian distribution is strictly positive
    #         return np.full(shape=len(observations), fill_value=-np.inf)

    #     return self._compute_pointwise_ll(mean, var, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, reduce=False,
            flattened=True, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivities to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
            ``reduce`` is prioritised over ``flattened``.
        :type reduce: bool, optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]
        vars = sigmas**2

        if np.any(sigmas < 0):
            # The std. of the Gaussian distribution is strictly positive
            score = -np.inf
            dtheta = np.empty((len(observations), 2, self._n_dim))
            dpsi = np.empty(observations.shape)
            return self._shape(score, dpsi, dtheta, reduce, flattened)

        if not self._centered:
            score, dpsi, dtheta = self._compute_non_centered_sensitivities(
                sigmas, observations, dlogp_dpsi)
            return self._shape(score, dpsi, dtheta, reduce, flattened)

        # Compute for centered parametrisation
        score = self._compute_log_likelihood(mus, vars, observations)
        dpsi, dtheta = self._compute_sensitivities(mus, vars, observations)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi

        return self._shape(score, dpsi, dtheta, reduce, flattened)

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

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        n_ids = int(n_ids)

        return (n_ids * self._n_dim, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If ``centered = False`` random samples from the standard normal
        distribution are returned.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
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
        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)

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
        self._n_hierarchical_dim = 0
        self.set_n_ids(n_ids, no_shortcut=True)

        # Set special dimensions
        self._n_pooled_dims = 0
        self._n_hetero_dims = self._n_dim

    def _shape(self, score, dpsi, dtheta, reduce, flattened):
        """
        Returns the score, dpsi and dtheta.

        If reduce is True, dpsi and dtheta are flattened according to the
        hierarchical ordering.

        If reduce is False, dpsi and dtheta are returned separately.

        If flattened is True, dtheta is flattened. If flattened is False,
        dtheta is returned in shape (n_ids, n_param_per_dim, n_dim)
        """
        if reduce or flattened:
            # Since all dtheta are zero, this is a more efficient
            # implementation than summing
            dtheta = np.zeros(self._n_parameters)
        if reduce:
            # non-zero entries can come from dpsi
            dtheta = dpsi.flatten()
            return score, dtheta

        return score, dpsi, dtheta

    def compute_individual_parameters(
            self, parameters, eta, return_eta=False, *args, **kwargs):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        :param return_eta: A boolean flag indicating whether eta is returned
            regardless of whether the parametrisation is centered or not.
        :type return_eta: boolean, optional
        """
        if parameters.ndim == 1:
            parameters = parameters.reshape(self._n_ids, self._n_dim)
        elif parameters.ndim == 3:
            parameters = np.diagonal(parameters, axis1=0, axis2=1).T

        return parameters

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        r"""
        Returns the log-likelihood of the population model parameters.

        A heterogenous population model is equivalent to a
        multi-dimensional delta-distribution, where each bottom-level parameter
        is determined by a separate delta-distribution.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        elif parameters.ndim == 3:
            # Heterogenous model is special, because n_param_per_dim = n_ids.
            # But after covariate transformation, the covariate information is
            # in the n_ids dimension.
            parameters = parameters[:, 0, :]

        # Return -inf if any of the observations do not equal the heterogenous
        # parameters
        mask = np.not_equal(observations, parameters)
        if np.any(mask):
            return -np.inf

        # Otherwise return 0
        return 0

    # def compute_pointwise_ll(self, parameters, observations):
    #     r"""
    #     Returns the pointwise log-likelihood of the model parameters for
    #     each observation.

    #     A heterogenous population model imposes no restrictions on the
    #     individuals, as a result the log-likelihood score is zero
    #     irrespective
    #     of the model parameters.

    #     Parameters
    #     ----------
    #     parameters
    #         An array-like object with the parameters of the population model.
    #     observations
    #         An array-like object with the observations of the individuals.
    #         Each entry is assumed to belong to one individual.
    #     """
    #     raise NotImplementedError

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, reduce=False,
            flattened=True, *args, **kwargs):
        r"""
        Returns the log-likelihood of the population parameters and its
        sensitivities w.r.t. the parameters and the observations.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
            ``reduce`` is prioritised over ``flattened``.
        :type reduce: bool, optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        elif parameters.ndim == 3:
            # Heterogenous model is special, because n_param_per_dim = n_ids.
            # But after covariate transformation, the covariate information is
            # in the n_ids dimension.
            parameters = parameters[:, 0, :]

        # Return -inf if any of the observations does not equal the
        # heterogenous parameters
        n_ids = len(observations)
        mask = observations != parameters
        if np.any(mask):
            score = -np.inf
            dpsi = np.empty(observations.shape)
            dtheta = np.empty((n_ids, self._n_ids, self._n_dim))
            return self._shape(score, dpsi, dtheta, reduce, flattened)

        # Otherwise return
        score = 0
        dpsi = np.zeros(observations.shape)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        dtheta = np.zeros((n_ids, self._n_ids, self._n_dim))
        return self._shape(score, dpsi, dtheta, reduce, flattened)

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

        :param n_ids: Number of individuals.
        :type n_ids: int
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
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        For ``n_samples > 1`` the samples are randomly drawn from the ``n_ids``
        individuals.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
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

    def set_n_ids(self, n_ids, no_shortcut=False):
        """
        Sets the number of modelled individuals.

        The behaviour of most population models is the same for any number of
        individuals, in which case ``n_ids`` is ignored. However, for some
        models, e.g. :class:`HeterogeneousModel` the behaviour changes with
        ``n_ids``.

        :param n_ids: Number of individuals.
        :type n_ids: int
        :param no_shortcut: Boolean flag that prevents exiting the method
            early when n_ids are unchanged.
        :type no_shortcut: bool, optional
        """
        n_ids = int(n_ids)

        if n_ids < 1:
            raise ValueError(
                'The number of modelled individuals needs to be greater or '
                'equal to 1.')

        if (n_ids == self._n_ids) and not no_shortcut:
            return None

        self._n_ids = n_ids
        self._n_parameters = self._n_ids * self._n_dim
        self._parameter_names = []
        for _id in range(self._n_ids):
            self._parameter_names += ['ID %d' % (_id + 1)] * self._n_dim

        # Update special dims
        self._special_dims = [[0, self._n_dim, 0, self._n_parameters, False]]

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
    A population model which models parameters across individuals
    with a lognormal distribution.

    A lognormal population model assumes that a model parameter :math:`\psi`
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

    If ``centered = False`` the parametrisation is non-centered, i.e.

    .. math::
        \log \psi = \mu _{\text{log}} + \sigma _{\text{log}} \eta ,

    where :math:`\eta` models the inter-individual variability and is
    standard normally distributed.

    Extends :class:`PopulationModel`.

    :param n_dim: The dimensionality of the population model.
    :type n_dim: int, optional
    :param dim_names: Optional names of the population dimensions.
    :type dim_names: List[str], optional
    :param centered: Boolean flag indicating whether parametrisation is
        centered or non-centered.
    :type centered: bool, optional
    """

    def __init__(self, n_dim=1, dim_names=None, centered=True):
        super(LogNormalModel, self).__init__(n_dim, dim_names)

        # Set number of parameters
        self._n_parameters = 2 * self._n_dim

        # Set default parameter names
        self._parameter_names = [
            'Log mean'] * self._n_dim + ['Log std.'] * self._n_dim

        self._centered = bool(centered)

    def _compute_dpsi(self, mu, sigma, etas):
        """
        Computes the partial derivatives of psi = exp(mu + sigma eta) w.r.t.
        eta, mu and sigma.

        mu: (n_ids, n_dim)
        sigma: (n_ids, n_dim)
        etas: (n_ids, n_dim)

        rtype: np.ndarray of shape (n_ids, n_dim),
            np.ndarray of shape (n_ids, 2, n_dim)
        """
        n_ids, n_dim = etas.shape
        psi = np.exp(mu + sigma * etas)
        dpsi_deta = sigma * psi
        dpsi_dtheta = np.empty(shape=(n_ids, 2, n_dim))
        dpsi_dtheta[:, 0] = psi
        dpsi_dtheta[:, 1] = etas * psi
        return dpsi_deta, dpsi_dtheta

    @staticmethod
    def _compute_log_likelihood(mus, vars, observations):
        r"""
        Calculates the log-likelihood using.

        mus shape: (n_ids, n_dim)
        vars shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)
        """
        # Compute log-likelihood score
        with np.errstate(divide='ignore'):
            log_likelihood = - np.sum(
                np.log(2 * np.pi * vars) / 2 + np.log(observations)
                + (np.log(observations) - mus)**2 / 2 / vars)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    @staticmethod
    def _compute_non_centered_log_likelihood(observations):  # pragma: no cover
        r"""
        Calculates the log-likelihood.

        observations: (n_ids, n_dim)
        """
        # Compute log-likelihood score
        log_likelihood = - np.sum(
            np.log(2 * np.pi) / 2 + observations ** 2 / 2)

        # If score evaluates to NaN, return -infinity
        if np.isnan(log_likelihood):
            return -np.inf

        return log_likelihood

    def _compute_non_centered_sensitivities(
            self, mus, sigmas, observations, dlogp_dpsi):
        """
        Returns the log-likelihood and the sensitivities with respect to
        eta and theta.
        """
        # Copmute score
        score = self._compute_non_centered_log_likelihood(observations)

        # Compute sensitivities
        if dlogp_dpsi is None:
            dlogp_dpsi = np.zeros((1, self._n_dim))
        deta = -observations
        dpsi_deta, dpsi_dtheta = self._compute_dpsi(mus, sigmas, observations)
        dlogp_deta = dlogp_dpsi * dpsi_deta + deta
        dlogp_dtheta = dlogp_dpsi[:, np.newaxis, :] * dpsi_dtheta

        return score, dlogp_deta, dlogp_dtheta

    @staticmethod
    def _compute_pointwise_ll(mean, var, observations):  # pragma: no cover
        r"""
        Calculates the pointwise log-likelihoods using numba speed up.
        """
        # Compute log-likelihood score
        with np.errstate(divide='ignore'):
            log_psi = np.log(observations)
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
        mus shape: (n_ids, n_dim)
        vars shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)

        Returns:
        deta for non-centered of shape (n_ids, n_dim)

        and
        deta and dtheta for centered
        dtheta: np.ndarray of shape (n_ids, n_parameters, n_dim)
        """
        # Compute sensitivities
        n_ids = len(psi)
        with np.errstate(divide='ignore'):
            dpsi = - ((np.log(psi) - mus) / vars + 1) / psi
            dmus = (np.log(psi) - mus) / vars
            dstd = (-1 + (np.log(psi) - mus)**2 / vars) / np.sqrt(vars)

        # Collect sensitivities
        n_ids, n_dim = psi.shape
        dtheta = np.empty(shape=(n_ids, 2, n_dim))
        dtheta[:, 0] = dmus
        dtheta[:, 1] = dstd

        return dpsi, dtheta

    def compute_individual_parameters(
            self, parameters, eta, return_eta=False, *args, **kwargs):
        r"""
        Returns the individual parameters.

        If ``centered = True``, the model does not transform the parameters
        and ``eta`` is returned.

        If ``centered = False``, the individual parameters are computed using

        .. math::
            \psi = \mathrm{e}^{
                \mu _{\mathrm{log}} + \sigma _{\mathrm{log}} \eta},

        where :math:`\mu _{\mathrm{log}}` and :math:`\sigma _{\mathrm{log}}`
        are the model parameters and :math:`\eta` are the inter-individual
        fluctuations.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids * n_dim)`` or
            ``(n_ids, n_dim)``
        :param return_eta: A boolean flag indicating whether eta is returned
            regardless of whether the parametrisation is centered or not.
        :type return_eta: boolean, optional
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        eta = np.asarray(eta)
        if eta.ndim == 1:
            eta = eta.reshape(self._n_ids, self._n_dim)
        if self._centered or return_eta:
            return eta

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            n_parameters = parameters[np.newaxis, ...]

        mu = parameters[:, 0]
        sigma = parameters[:, 1]
        if np.any(sigma < 0):
            return np.full(shape=eta.shape, fill_value=np.nan)

        psi = np.exp(mu + sigma * eta)

        return psi

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        If ``centered = False``, the log-likelihood of the standard normal
        is returned. The contribution of the population parameters to the
        log-likelihood can be computed with the log-likelihood of the
        individual parameters, see :class:`chi.ErrorModel`.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        if not self._centered:
            return self._compute_non_centered_log_likelihood(observations)

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]
        vars = sigmas**2

        if np.any(sigmas < 0):
            # The scale. of the lognormal distribution is strictly positive
            return -np.inf

        return self._compute_log_likelihood(mus, vars, observations)

    # def compute_pointwise_ll(
    #         self, parameters, observations):  # pragma: no cover
    #     r"""
    #     Returns the pointwise log-likelihood of the model parameters for
    #     each observation.

    #     The pointwise log-likelihood of a LogNormalModel is the log-pdf
    #     evaluated at the observations

    #     .. math::
    #         L(\mu _{\text{log}}, \sigma _{\text{log}}| \psi _i) =
    #         \log p(\psi _i |
    #         \mu _{\text{log}}, \sigma _{\text{log}}) ,

    #     where
    #     :math:`\psi _i` are the "observed" parameters :math:`\psi` from
    #     individual :math:`i`.

    #     Parameters
    #     ----------
    #     parameters
    #         An array-like object with the model parameter values, i.e.
    #         [:math:`\mu _{\text{log}}`, :math:`\sigma _{\text{log}}`].
    #     observations
    #         An array like object with the parameter values for the
    #         individuals,
    #         i.e. [:math:`\psi _1, \ldots , \psi _N`].
    #     """
    #     # TODO: Needs proper research to establish which pointwise
    #     # log-likelihood makes sense for hierarchical models.
    #     # Also needs to be adapted to match multi-dimensional API.
    #     raise NotImplementedError
    #     observations = np.asarray(observations)
    #     mean, std = parameters
    #     var = std**2

    #     eps = 1E-12
    #     if (std <= 0) or (var <= eps) or np.any(observations == 0):
    #         # The standard deviation of log psi is strictly positive
    #         return np.full(shape=len(observations), fill_value=-np.inf)

    #     return self._compute_pointwise_ll(mean, var, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, reduce=False,
            flattened=True, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
            ``reduce`` is prioritised over ``flattened``.
        :type reduce: bool, optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            n_parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]
        vars = sigmas**2

        if np.any(sigmas < 0):
            # The scale of the lognormal distribution is strictly positive
            score = -np.inf
            dpsi = np.empty(observations.shape)
            dtheta = np.empty((len(observations), 2, self._n_dim))
            return self._shape(score, dpsi, dtheta, reduce, flattened)

        if not self._centered:
            score, dpsi, dtheta = self._compute_non_centered_sensitivities(
                mus, sigmas, observations, dlogp_dpsi)
            return self._shape(score, dpsi, dtheta, reduce, flattened)

        # Compute for centered parametrisation
        score = self._compute_log_likelihood(mus, vars, observations)
        dpsi, dtheta = self._compute_sensitivities(mus, vars, observations)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi

        return self._shape(score, dpsi, dtheta, reduce, flattened)

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
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If ``centered = False`` random samples from the standard normal
        distribution are returned.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
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

        # Instantiate random number generator
        rng = np.random.default_rng(seed=seed)

        # Get parameters
        mus = parameters[0]
        sigmas = parameters[1]
        if not self._centered:
            mus = np.zeros(mus.shape)
            sigmas = np.ones(sigmas.shape)
            return rng.normal(loc=mus, scale=sigmas, size=sample_shape)

        if np.any(sigmas <= 0):
            raise ValueError(
                'A log-normal distribution only accepts strictly positive '
                'standard deviations.')

        # Sample from population distribution
        # (Mean and sigma are the mean and standard deviation of
        # the log samples)
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

        # Set number of hierarchical dimensions
        self._n_hierarchical_dim = 0

        # Set special dimensions
        self._special_dims = [[0, self._n_dim, 0, self._n_parameters, True]]
        self._n_pooled_dims = self._n_dim
        self._n_hetero_dims = 0

        # Set default parameter names
        self._parameter_names = ['Pooled'] * self._n_dim

    def _shape(self, score, dpsi, dtheta, reduce, flattened):
        """
        Returns the score, dpsi and dtheta.

        If reduce is True, dpsi and dtheta are flattened according to the
        hierarchical ordering.

        If reduce is False, dpsi and dtheta are returned separately.

        If flattened is True, dtheta is flattened. If flattened is False,
        dtheta is returned in shape (n_ids, n_param_per_dim, n_dim)
        """
        if reduce or flattened:
            # Since all dtheta are zero, this is a more efficient
            # implementation than summing
            dtheta = np.zeros(self._n_parameters)
        if reduce:
            # dpsi can be non-zero
            dtheta = np.sum(dpsi, axis=0)
            return score, dtheta

        return score, dpsi, dtheta

    def compute_individual_parameters(
            self, parameters, eta, return_eta=False, *args, **kwargs):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        :param return_eta: A boolean flag indicating whether eta is returned
            regardless of whether the parametrisation is centered or not.
        :type return_eta: boolean, optional
        """
        if parameters.ndim < 3:
            parameters = np.broadcast_to(
                parameters.reshape(1, self._n_dim),
                shape=(self._n_ids, self._n_dim))
        elif parameters.ndim == 3:
            parameters = parameters[:, 0, :]

        return parameters

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(n_parameters, self._n_dim)
        elif parameters.ndim == 3:
            parameters = parameters[:, 0]

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

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, reduce=False,
            flattened=True, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
            ``reduce`` is prioritised over ``flattened``.
        :type reduce: bool, optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 3:
            parameters = parameters[:, 0]

        # Return -inf if any of the observations does not equal the pooled
        # parameter
        mask = observations != parameters
        if np.any(mask):
            score = -np.inf
            dpsi = np.empty(observations.shape)
            dtheta = np.empty((len(observations), 1, self._n_dim))
            return self._shape(score, dpsi, dtheta, reduce, flattened)

        # Otherwise return
        score = 0
        dpsi = np.zeros(observations.shape)
        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi
        dtheta = np.zeros((len(observations), 1, self._n_dim))

        return self._shape(score, dpsi, dtheta, reduce, flattened)

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

        :param n_ids: Number of individuals.
        :type n_ids: int
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
        ``n_samples`` times and are returned.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
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
        self._n_dim = population_model.n_dim()
        self._n_covariates = population_model.n_covariates()
        self._n_hierarchical_dim = population_model.n_hierarchical_dim()

    def compute_individual_parameters(self, parameters, eta, *args, **kwargs):
        """
        Returns the individual parameters.

        If the model does not transform the bottom-level parameters, ``eta`` is
        returned.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: np.ndarray of shape ``(n_ids, n_dim)``
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        return self._population_model.compute_individual_parameters(
            parameters, eta, *args, **kwargs)

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        parameters = np.asarray(parameters).flatten()
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Compute log-likelihood
        score = self._population_model.compute_log_likelihood(
            parameters, observations, *args, **kwargs)

        return score

    # def compute_pointwise_ll(self, parameters, observations):
    #     """
    #     Returns the pointwise log-likelihood of the population model
    #     parameters
    #     for each observation.

    #     Parameters
    #     ----------
    #     parameters
    #         An array-like object with the parameters of the population model.
    #     observations
    #         An array-like object with the observations of the individuals.
    #         Each
    #         entry is assumed to belong to one individual.
    #     """
    #     # TODO: Needs proper research to establish which pointwise
    #     # log-likelihood makes sense for hierarchical models.
    #     # Also needs to be adapted to match multi-dimensional API.
    #     raise NotImplementedError
    #     # # Get fixed parameter values
    #     # if self._fixed_params_mask is not None:
    #     #     self._fixed_params_values[~self._fixed_params_mask] = \
    #               parameters
    #     #     parameters = self._fixed_params_values

    #     # # Compute log-likelihood
    #     # scores = self._population_model.compute_pointwise_ll(
    #     #     parameters, observations)

    #     # return scores

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, reduce=False,
            **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
            ``reduce`` is prioritised over ``flattened``.
        :type reduce: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        parameters = np.asarray(parameters).flatten()
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Compute log-likelihood and sensitivities
        kwargs['flattened'] = True
        output = self._population_model.compute_sensitivities(
            parameters=parameters,
            observations=observations,
            dlogp_dpsi=dlogp_dpsi,
            reduce=reduce,
            **kwargs)

        if self._fixed_params_mask is None:
            return output

        # Need to filter sensitivities of fixed top-level parameters
        if not reduce:
            score, dpsi, dtheta = output
            return score, dpsi, dtheta[~self._fixed_params_mask]

        score, dscore = output
        n_bottom, _ = self._population_model.n_hierarchical_parameters(
            len(observations))
        dpsi = dscore[:n_bottom]
        dtheta = dscore[n_bottom:][~self._fixed_params_mask]

        return score, np.hstack((dpsi, dtheta))

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

    def get_dim_names(self):
        """
        Returns the names of the dimensions.
        """
        return self._population_model.get_dim_names()

    def get_special_dims(self):
        r"""
        Returns information on pooled and heterogeneously modelled dimensions.

        Returns a tuple with 3 entries:
            1. A list of lists, each entry containing 1. the start and end
                dimension of the special dimensions; 2. the associated
                start and end index of the model parameters; 3. and a boolean
                indicating whether the dimension is pooled (``True``) or
                heterogeneous (``False``).
            2. The number of pooled dimensions.
            3. The number of heterogeneous dimensions.
        """
        special_dims, n_pooled_dims, n_hetero_dims = \
            self._population_model.get_special_dims()

        if self._fixed_params_mask is None:
            return special_dims, n_pooled_dims, n_hetero_dims

        # If parameters are fixed, we need to reindex top level parameters
        s_dims = []
        for s in special_dims:
            start = s[2]
            end = s[3]

            # Shift by number of leading fixed parameters
            start -= int(np.sum(self._fixed_params_mask[:start]))
            end -= int(np.sum(self._fixed_params_mask[:end]))
            s_dims += [[
                s[0], s[1], start, end, s[4]
            ]]

        return s_dims, n_pooled_dims, n_hetero_dims

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
            n_fixed = int(np.sum(self._fixed_params_mask))
            n_pop = self._n_parameters - n_fixed

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

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :param n_samples: Number of samples. If ``None``, one sample is
            returned.
        :type n_samples: int, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int, optional
        """
        # Get fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        # Sample from population model
        sample = self._population_model.sample(
            parameters=parameters, n_samples=n_samples, seed=seed, *args,
            **kwargs)

        return sample

    def set_covariate_names(self, names=None):
        """
        Sets the names of the covariates.

        If the model has no covariates, input is ignored.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List
        """
        self._population_model.set_covariate_names(names)

    def set_dim_names(self, names=None):
        """
        Sets the names of the population model dimensions.

        :param names: A list of dimension names. If ``None``, dimension names
            are reset to defaults.
        :type names: List[str], optional
        """
        self._population_model.set_dim_names(names)

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
        self._population_model.set_n_ids(n_ids)

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


class TruncatedGaussianModel(PopulationModel):
    r"""
    A population model which models model parameters across individuals
    as Gaussian random variables which are truncated at zero.

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

        mus shape: (n_ids, n_dim)
        sigmas shape: (n_ids, n_dim)
        observations: (n_ids, n_dim)
        """
        # Return infinity if any psis are negative
        if np.any(observations < 0):
            return -np.inf

        # Compute log-likelihood score
        with np.errstate(divide='ignore'):
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
        """
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
        """
        Calculates the log-likelihood and its sensitivities.

        Expects:
        mus shape: (n_ids, n_dim)
        sigmas shape: (n_ids, n_dim)
        psi: (n_ids, n_dim)

        Returns:
        log_likelihood: float
        dpsi: (n_ids, n_dim)
        dtheta: (n_ids, n_parameters, n_dim)
        """
        # Compute log-likelihood score
        log_likelihood = self._compute_log_likelihood(mus, sigmas, psi)

        n_ids = len(psi)
        dtheta = np.empty(shape=(n_ids, self._n_parameters, self._n_dim))
        if np.isinf(log_likelihood):
            return (-np.inf, np.empty(shape=psi.shape), dtheta)

        # Compute sensitivities
        with np.errstate(divide='ignore'):
            dpsi = (mus - psi) / sigmas**2
            dtheta[:, 0] = (
                (psi - mus) / sigmas
                - _norm_pdf(mus/sigmas) / (1 - _norm_cdf(-mus/sigmas))
                ) / sigmas
            dtheta[:, 1] = (
                -1 + (psi - mus)**2 / sigmas**2
                + _norm_pdf(mus/sigmas) * mus / sigmas
                / (1 - _norm_cdf(-mus/sigmas))
                ) / sigmas

        return log_likelihood, dpsi, dtheta

    def compute_log_likelihood(
            self, parameters, observations, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :rtype: float
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]

        if np.any(sigmas <= 0):
            # Gaussians are only defined for positive sigmas.
            return -np.inf

        return self._compute_log_likelihood(mus, sigmas, observations)

    # def compute_pointwise_ll(self, parameters, observations):
    #     r"""
    #     Returns the pointwise log-likelihood of the model parameters for
    #     each observation.

    #     The pointwise log-likelihood of a truncated Gaussian distribution is
    #     the log-pdf evaluated at the observations

    #     .. math::
    #         L(\mu , \sigma | \psi _i) =
    #         \log p(\psi _i |
    #         \mu , \sigma ) ,

    #     where
    #     :math:`\psi _i` are the "observed" parameters :math:`\psi` from
    #     individual :math:`i`.

    #     Parameters
    #     ----------
    #     parameters
    #         An array-like object with the model parameter values, i.e.
    #         [:math:`\mu`, :math:`\sigma`].
    #     observations
    #         An array like object with the parameter values for the
    #         individuals,
    #         i.e. [:math:`\psi _1, \ldots , \psi _N`].
    #     """
    #     # TODO: Needs proper research to establish which pointwise
    #     # log-likelihood makes sense for hierarchical models.
    #     # Also needs to be adapted to match multi-dimensional API.
    #     raise NotImplementedError
    #     # observations = np.asarray(observations)
    #     # mean, std = parameters

    #     # if (mean <= 0) or (std <= 0):
    #     #     # The mean and std. of the Gaussian distribution are
    #     #     # strictly positive if truncated at zero
    #     #     return np.full(shape=len(observations), fill_value=-np.inf)

    #     # return self._compute_pointwise_ll(mean, std, observations)

    def compute_sensitivities(
            self, parameters, observations, dlogp_dpsi=None, reduce=False,
            flattened=True, *args, **kwargs):
        """
        Returns the log-likelihood of the population model parameters and
        its sensitivity to the population parameters as well as the
        observations.

        The sensitivities of the bottom-level log-likelihoods with respect to
        the ``observations`` (bottom-level parameters) may be provided using
        ``dlogp_dpsi``, in order to compute the sensitivities of the full
        hierarchical log-likelihood.

        The log-likelihood and sensitivities are returned as a tuple
        ``(score, deta, dtheta)``.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(n_parameters,)``,
            ``(n_param_per_dim, n_dim)`` or ``(n_ids, n_param_per_dim, n_dim)``
        :param observations: Individual model parameters.
        :type observations: np.ndarray of shape ``(n_ids, n_dim)``
        :param dlogp_dpsi: The sensitivities of the log-likelihood of the
            individual parameters with respect to the individual parameters.
        :type dlogp_dpsi: np.ndarray of shape ``(n_ids, n_dim)``,
            optional
        :param reduce: A boolean flag indicating whether the sensitivites are
            returned as an array of shape ``(n_hierarchical_parameters,)``.
            ``reduce`` is prioritised over ``flattened``.
        :type reduce: bool, optional
        :param flattened: Boolean flag that indicates whether the sensitivities
            w.r.t. the population parameters are returned as 1-dim. array. If
            ``False`` sensitivities are returned in shape
            ``(n_ids, n_param_per_dim, n_dim)``.
        :type flattened: bool, optional
        :rtype: Tuple[float, np.ndarray of shape ``(n_ids, n_dim)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations[:, np.newaxis]

        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            n_parameters = len(parameters) // self._n_dim
            parameters = parameters.reshape(1, n_parameters, self._n_dim)
        elif parameters.ndim == 2:
            parameters = parameters[np.newaxis, ...]

        # Parse parameters
        mus = parameters[:, 0]
        sigmas = parameters[:, 1]

        if np.any(sigmas <= 0):
            # Gaussians are only defined for positive sigmas.
            n_ids, n_dim = observations.shape
            score = -np.inf
            dpsi = np.empty(observations.shape)
            dtheta = np.empty((n_ids, self._n_parameters, n_dim))
            return self._shape(score, dpsi, dtheta, reduce, flattened)

        score, dpsi, dtheta = self._compute_sensitivities(
            mus, sigmas, observations)

        if dlogp_dpsi is not None:
            dpsi += dlogp_dpsi

        return self._shape(score, dpsi, dtheta, reduce, flattened)

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
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
        :rtype: np.ndarray of shape ``(p_per_dim, n_dim)``
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
        output = np.empty((self._n_parameters, self._n_dim))
        output[0] = \
            mus + sigmas * norm.pdf(mus/sigmas) / (1 - norm.cdf(-mus/sigmas))
        output[1] = np.sqrt(
            sigmas**2 * (
                1 -
                mus / sigmas * norm.pdf(mus/sigmas)
                / (1 - norm.cdf(-mus/sigmas))
                - (norm.pdf(mus/sigmas) / (1 - norm.cdf(-mus/sigmas)))**2)
            )

        return output

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

        :param n_ids: Number of individuals.
        :type n_ids: int
        """
        n_ids = int(n_ids)

        return (n_ids * self._n_dim, self._n_parameters)

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def sample(self, parameters, n_samples=None, seed=None, *args, **kwargs):
        """
        Returns random samples from the population distribution.

        The returned value is a NumPy array with shape ``(n_samples, n_dim)``.

        If ``centered = False`` random samples from the standard normal
        distribution are returned.

        :param parameters: Parameters of the population model.
        :type parameters: np.ndarray of shape ``(p,)`` or
            ``(p_per_dim, n_dim)``
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
        """
        Sets the names of the population model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List[str]
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
