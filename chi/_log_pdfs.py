#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy
import warnings

import myokit
import numpy as np
import pints

import chi


class HierarchicalLogLikelihood(object):
    r"""
    A hierarchical log-likelihood consists of structurally identical
    log-likelihoods whose parameters are related by a population model.

    A hierarchical log-likelihood takes a list of :class:`LogLikelihood`
    instances and a :class:`PopulationModel` with the same dimension as the
    number of parameters of the log-likelihoods. The hierarchical
    log-likelihood is defined as

    .. math::
        \log p(\mathcal{D}, \Psi | \theta ) =
            \sum _{ij} \log p(y_{ij} | \psi _{i} , t_{ij})
            + \sum _{ik} \log p(\psi _{ik}| \theta _k),

    where :math:`\sum _{j} \log p(y_{ij} | \psi _{i} , t_{ij})` is the
    log-likelihood of :math:`\psi _{i}` associated with individual :math:`i`
    and :math:`\sum _{ik} \log p(\psi _{ik}| \theta _k)` is the log-likelihood
    of the population parameters. :math:`\mathcal{D}=\{ \mathcal{D}_i\}` is the
    data, composed of measurements from different individuals
    :math:`\mathcal{D}_i = \{(y_{ij}, t_{ij})\}`. :math:`\Psi = \{ \psi_i\}`
    is the collection of bottom-level parameters, which in turn are the
    parameters of the individual log-likelihoods
    :math:`\psi_i = \{ \psi _{ik}\}`.
    We use :math:`i` to index individuals, :math:`j` to index measurements for
    an individual and :math:`k` to index the bottom-level parameters.
    :math:`\theta _k = \{ \theta _{kr} \}` are the population parameters
    associated with the :math:`k^{\text{th}}` bottom-level parameter.

    :param log_likelihoods: A list of log-likelihoods which are
        defined on the same parameter space with dimension ``n_parameters``.
    :type log_likelihoods: list[LogLikelihood] of length ``n_ids``
    :param population_models: A population model of dimension ``n_parameters``.
    :type population_models: PopulationModel
    :param covariates: A 2-dimensional array of with the
        individual's covariates of shape ``(n_ids, n_cov)``.
    :type covariates: np.ndarray of shape ``(n_ids, n_cov)``, optional
    """
    def __init__(
            self, log_likelihoods, population_model, covariates=None):
        super(HierarchicalLogLikelihood, self).__init__()
        for log_likelihood in log_likelihoods:
            if not isinstance(log_likelihood, chi.LogLikelihood):
                raise ValueError(
                    'The log-likelihoods have to be instances of a '
                    'chi.LogLikelihood.')

        if not isinstance(population_model, chi.PopulationModel):
            raise ValueError(
                'The population model has to be an instances of '
                'chi.PopulationModel.')

        n_parameters = population_model.n_dim()
        for log_likelihood in log_likelihoods:
            if log_likelihood.n_parameters() != n_parameters:
                raise ValueError(
                    'The dimension of the population model does not match the '
                    'the dimensions of the log-likelihood parameter space.')

        n_ids = len(log_likelihoods)
        if population_model.needs_covariates():
            if covariates is None:
                raise ValueError(
                    'The population model needs covariates, but no covariates '
                    'have been provided.')

            # Make sure covariates have correct format
            n_cov = population_model.n_covariates()
            try:
                covariates = np.asarray(covariates).reshape(n_ids, n_cov)
            except ValueError:
                raise ValueError(
                    'The covariates do not have the correct format. '
                    'Covariates need to be of shape (n_ids, n_cov).')

        # Remember models and number of individuals
        self._log_likelihoods = log_likelihoods
        self._population_model = population_model
        self._covariates = covariates
        self._n_ids = n_ids
        self._n_dim = self._population_model.n_dim()
        info = self._count_parameters()
        self._n_parameters, self._pooled_dims, self._n_pooled_dim = info[:3]
        self._hetero_dims = info[3]

        # NOTE: Heterogeneous model parameters are weird. They count are top
        # parameters, but also are counted as bottom parameters during
        # inference because they are the parameters that go into the individual
        # likelihoods directly. self._n_top + self._n_bottom IS NOT
        # n_parameters, because heterogeneous parameters are counted twice.
        n_hetero_dim = \
            np.sum([dims[1] - dims[0] for dims in self._hetero_dims])
        self._n_top = int(
            self._n_ids * n_hetero_dim + self._population_model.n_parameters())
        self._n_bottom = \
            self._n_parameters - self._population_model.n_parameters()

        # Label log-likelihoods for later convenience
        self._label_log_likelihoods()

        # Count number of observations per individual
        self._n_obs = [np.sum(ll.n_observations()) for ll in log_likelihoods]

    def __call__(self, parameters):
        """
        Returns the log-likelihood score of the model.

        Expects parameters in order of
        [psi_1, ..., psi_n, theta_1, ..., theta_k], where
        psi_i = [psi_i1, ..., psi_ik] are the bottom parameters of
        individual i.
        """
        # Split parameters into bottom- and top-level parameters
        parameters = np.asarray(parameters)
        bottom_parameters = parameters[:self._n_bottom]
        top_parameters = parameters[self._n_bottom:]

        # Broadcast pooled parameters and reshape bottom parameters to
        # (n_ids, n_dim)
        bottom_parameters = self._reshape_bottom_parameters(
            bottom_parameters, top_parameters)

        # Compute population model score
        score = self._population_model.compute_log_likelihood(
            top_parameters, bottom_parameters)

        # Return if values already lead to a rejection
        if np.isinf(score):
            return score

        # Transform bottom-level parameters
        if self._population_model.transforms_individual_parameters():
            bottom_parameters = \
                self._population_model.compute_individual_parameters(
                    top_parameters, bottom_parameters, self._covariates)

        # Evaluate individual likelihoods
        for idi, log_likelihood in enumerate(self._log_likelihoods):
            score += log_likelihood(bottom_parameters[idi])
        return score

    def _remove_pooled_duplicates(self, sensitivities):
        """
        In some sense the reverse of self._reshape_bottom_parameters.

        Reshaping the bottom parameters introduces duplicates of pooled
        parameters, so the bottom parameters can be expressed in a shape
        (n_ids, n_dim). The sensitivities are now of length (n_ids * n_dim),
        even though, so the duplicates are treated as independent. Those
        sensitivities need to be summed, because the originate from the same
        pooled input parameter.
        """
        # Check for quick solution 1: no pooled parameters
        if self._n_pooled_dim == 0:
            return sensitivities

        # Get population parameter sensitvitities
        start_pop = self._n_ids * self._n_dim
        sens = np.zeros(shape=self._n_parameters)
        sens[self._n_bottom:] = sensitivities[start_pop:]

        # Check for quick solution 2: all parameters pooled
        if self._n_pooled_dim == self._n_dim:
            # Add contributions from bottom-level parameters
            # (For fully pooled models, n_pooled = n_pop)
            sens = np.sum(
                sensitivities[:start_pop].reshape(self._n_ids, self._n_dim),
                axis=0)
            return sens

        shift = 0
        current_dim = 0
        bottom_sensitivities = sensitivities[:start_pop].reshape(
            self._n_ids, self._n_dim)
        bottom_sens = np.zeros(
            shape=(self._n_ids, self._n_dim - self._n_pooled_dim))
        for info in self._pooled_dims:
            start_dim, end_dim, start_top, end_top = info
            # Fill leading non-pooled dims
            bottom_sens[:, current_dim-shift:start_dim-shift] = \
                bottom_sensitivities[:, current_dim:start_dim]
            # Fill pooled dims
            sens[self._n_bottom+start_top:self._n_bottom+end_top] = \
                np.sum(bottom_sensitivities[:, start_dim:end_dim], axis=0)
            current_dim = end_dim
            shift += end_dim - start_dim
        # Fill trailing non-pooled dims
        bottom_sens[:, current_dim-shift:] = bottom_sensitivities[
            :, current_dim:]

        # Add bottom sensitivties
        sens[:self._n_bottom] = bottom_sens.flatten()

        return sens

    def _count_parameters(self):
        """
        Counts the parameters of the hierarchical log-likelihood.

        For convenience it also remembers the indices where pooled parameters
        have to be broadcasted and inserted (pooled parameters appear only
        once for the inference, but the computation of the log-likelihood
        requires n_ids+1 copies of the pooled parametres).
        """
        # Get elementary population models
        pop_models = [self._population_model]
        if isinstance(self._population_model, chi.ComposedPopulationModel):
            pop_models = self._population_model.get_population_models()

        n_parameters = 0
        n_pooled_dims = 0
        pooled_dims = []
        hetero_dims = []
        current_dim = 0
        current_top_index = 0
        for pop_model in pop_models:
            # Check whether dimension is pooled
            n_bottom, n_top = pop_model.n_hierarchical_parameters(self._n_ids)
            if isinstance(pop_model, chi.PooledModel):
                # Remember start and end of pooled dimensions,
                # Start and end of pooled parameter values,
                pooled_dims.append([
                    current_dim, current_dim + pop_model.n_dim(),
                    current_top_index,
                    current_top_index + n_top])
                n_pooled_dims += pop_model.n_dim()
            if isinstance(pop_model, chi.HeterogeneousModel):
                # Remember start and end of heterogeneous dimensions
                # Subtract pooled dims for later convenience
                hetero_dims.append([
                    current_dim - n_pooled_dims,
                    current_dim + pop_model.n_dim() - n_pooled_dims])

            # Count overall number of parameters
            n_parameters += n_bottom + n_top
            current_dim += pop_model.n_dim()
            current_top_index += n_top

        return n_parameters, pooled_dims, n_pooled_dims, hetero_dims

    def _label_log_likelihoods(self):
        """
        Labels log-likelihoods if they don't already have an ID.
        """
        ids = []
        for idl, log_likelihood in enumerate(self._log_likelihoods):
            _id = log_likelihood.get_id()
            if not _id:
                _id = 'Log-likelihood %d' % (idl + 1)
            if _id in ids:
                raise ValueError('Log-likelihood IDs need to be unique.')
            log_likelihood.set_id(_id)
            ids.append(_id)

    def _reshape_bottom_parameters(self, bottom_parameters, top_parameters):
        """
        Takes bottom parameters and top parameters with no duplicates and
        returns bottom parameters of shape (n_ids, n_dim), where pooled
        parameters are duplicated.
        """
        bottom_params = np.empty(shape=(self._n_ids, self._n_dim))

        # Check for quick solution 1: no pooled parameters
        if self._n_pooled_dim == 0:
            bottom_params[:, :] = bottom_parameters.reshape(
                self._n_ids, self._n_dim)
            return bottom_params

        # Check for quick solution 2: all parameters pooled
        if self._n_pooled_dim == self._n_dim:
            bottom_params[:, :] = top_parameters[np.newaxis, :]
            return bottom_params

        shift = 0
        current_dim = 0
        bottom_parameters = bottom_parameters.reshape(
            self._n_ids, self._n_dim - self._n_pooled_dim)
        for info in self._pooled_dims:
            start_dim, end_dim, start_top, end_top = info
            # Fill leading non-pooled dims
            bottom_params[:, current_dim:start_dim] = bottom_parameters[
                :, current_dim-shift:start_dim-shift]
            # Fill pooled dims
            bottom_params[:, start_dim:end_dim] = top_parameters[
                start_top:end_top]
            current_dim = end_dim
            shift += end_dim - start_dim
        # Fill trailing non-pooled dims
        bottom_params[:, current_dim:] = bottom_parameters[
            :, current_dim-shift:]

        return bottom_params

    def compute_pointwise_ll(self, parameters, per_individual=True):
        r"""
        Returns the pointwise log-likelihood scores of the parameters for
        each
        observation.

        :param parameters: A list of parameter values.
        :type parameters: list, numpy.ndarray
        :param per_individual: A boolean flag that determines whether the
            scores are computed per individual or per observation.
        :type per_individual: bool, optional
        """
        # NOTE: Some thoughts on how to do it
        # The pointwise log-likelihood for an hierarchical model can be
        # straightforwardly defined when each observation is treated as one
        # "point"

        # .. math::
        #     L(\Psi , \theta | x^{\text{obs}}_{i}) =
        #         \sum _n \log p(x^{\text{obs}}_{in} | \psi _i )
        #         + \log p(\psi _i | \theta ) ,

        # where the sum runs over all :math:`N_i` observations of individual
        # :math:`i`.

        # Alternatively, the pointwise log-likelihoods may be computed per
        # observation point, assuming that the population contribution can
        # be uniformly attributed to each observation

        # .. math::
        #     L(\Psi , \theta | x^{\text{obs}}_{in}) =
        #         \log p(x^{\text{obs}}_{in} | \psi _i )
        #         + \log p(\psi _i | \theta ) / N_i .

        # Setting the flag ``per_individual`` to ``True`` or ``False`` switches
        # between the two modi.
        raise NotImplementedError
        # TODO: Implement for covariate model
        # TODO: Think again whether this pointwise log-likelihood
        # is really meaningful, e.g. when computing LOO.
        if np.any(self._uses_eta):
            raise NotImplementedError(
                'This method is not implemented for '
                'CovariatePopulationModels.'
            )

        # Transform parameters to numpy array
        parameters = np.asarray(parameters)

        # Compute population model scores of individuals
        start = 0
        pop_scores = np.zeros(shape=self._n_ids)
        for pop_model in self._population_models:
            # Get number of individual and population level parameters
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)

            # Get parameter ranges
            end_indiv = start + n_indiv
            end_pop = end_indiv + n_pop

            # Add score, if individual parameters exist
            if n_indiv > 0:
                pop_scores += pop_model.compute_pointwise_ll(
                    parameters=parameters[end_indiv:end_pop],
                    observations=parameters[start:end_indiv])

            # Shift start index
            start = end_pop

        if per_individual is True:
            # Compute aggregated individual likelihoods
            pw_log_likelihoods = pop_scores
            for index, log_likelihood in enumerate(self._log_likelihoods):
                # Compute scores for each observation
                pw_log_likelihoods[index] += log_likelihood(
                    parameters[self._indiv_params[index]])

            return pw_log_likelihoods

        # Evaluate individual likelihoods pointwise
        pw_log_likelihoods = []
        for index, log_likelihood in enumerate(self._log_likelihoods):
            # Compute scores for each observation
            scores = log_likelihood.compute_pointwise_ll(
                parameters[self._indiv_params[index]])

            # Add population contribution
            scores += pop_scores[index] / len(scores)

            # Safe scores
            pw_log_likelihoods.append(scores)

        return np.hstack(pw_log_likelihoods)

    def evaluateS1(self, parameters):
        """
        Computes the log-likelihood of the parameters and its
        sensitivities.

        :param parameters: A list of parameter values
        :type parameters: list, numpy.ndarray
        """
        # Split parameters into bottom- and top-level parameters
        parameters = np.asarray(parameters)
        bottom_parameters = parameters[:self._n_bottom]
        top_parameters = parameters[self._n_bottom:]

        # Broadcast pooled parameters and reshape bottom parameters to
        # (n_ids, n_dim)
        bottom_parameters = self._reshape_bottom_parameters(
            bottom_parameters, top_parameters)

        # Compute population model score
        score, sensitivities = self._population_model.compute_sensitivities(
            top_parameters, bottom_parameters)

        # Return if values already lead to a rejection
        if np.isinf(score):
            return score, np.full(shape=self._n_parameters, fill_value=np.inf)

        # Transform bottom-level parameters
        dpsi = None
        if self._population_model.transforms_individual_parameters():
            bottom_parameters, dpsi = \
                self._population_model.compute_individual_sensitivities(
                    top_parameters, bottom_parameters, self._covariates)
            dpsi_deta = dpsi[0]
            dpsi_dtheta = dpsi[1:]

        # Evaluate individual likelihoods
        for idi, log_likelihood in enumerate(self._log_likelihoods):
            l, dl_dpsi = log_likelihood.evaluateS1(bottom_parameters[idi])

            # Collect score and sensitivities
            score += l
            if dpsi is None:
                sensitivities[idi*self._n_dim:(idi+1)*self._n_dim] += dl_dpsi
            else:
                sensitivities[idi*self._n_dim:(idi+1)*self._n_dim] += \
                    dl_dpsi * dpsi_deta[idi]
                sensitivities[self._n_ids*self._n_dim:] += np.sum(
                    dl_dpsi[np.newaxis, :] * dpsi_dtheta[:, idi, :],
                    axis=1)

        # Collect sensitivities of pooled parameters
        sensitivities = self._remove_pooled_duplicates(sensitivities)

        return score, sensitivities

    def get_id(self, unique=False):
        """
        Returns the IDs (prefixes) of the model parameters.

        By default the IDs of all parameters (bottom and top level) parameters
        are returned in the order of the parameter names. If ``unique``
        is set to ``True``, each ID is returned only once.
        """
        ids = []
        for log_likelihood in self._log_likelihoods:
            ids.append(log_likelihood.get_id())

        if unique:
            return ids

        # Add n_bottom / n_ids copies of ids
        n_copies = self._n_bottom // self._n_ids
        _ids = []
        for _id in ids:
            _ids += [_id] * n_copies
        ids = _ids

        # Append None's for population level parameters
        ids += [None] * self._population_model.n_parameters()

        return ids

    def get_parameter_names(
            self, exclude_bottom_level=False, include_ids=False):
        """
        Returns the names of the model.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter names are returned in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        :param include_ids: A boolean flag which determines whether the IDs
            (prefixes) of the model parameters are included.
        :type include_ids: bool, optional
        """
        # Get bottom parameter names (pooled parameters need to be filtered)
        current_dim = 0
        names = self._log_likelihoods[0].get_parameter_names()
        n = []
        for info in self._pooled_dims:
            start_dim, end_dim, _, _ = info
            n += names[current_dim:start_dim]
            current_dim = end_dim
        n += names[current_dim:]
        names = n

        # Make copies of bottom parameters and append top parameters
        names = names * self._n_ids
        names += self._population_model.get_parameter_names()

        if include_ids:
            ids = self.get_id()
            n = []
            for idn, name in enumerate(names):
                if ids[idn]:
                    name = ids[idn] + ' ' + name
                n.append(name)
            names = n

        if exclude_bottom_level:
            n = []
            for info in self._hetero_dims:
                # NOTE: Heterogeneous bottom parameters count as top parameters
                start_dim, end_dim = info
                while start_dim < end_dim:
                    n += names[start_dim:self._n_bottom:self._n_ids]
                    start_dim += 1
            # Append top-level parameters
            n += names[self._n_bottom:]
            names = n

        return names

    def get_population_model(self):
        """
        Returns the population model.
        """
        return self._population_model

    def n_log_likelihoods(self):
        """
        Returns the number of individual likelihoods.
        """
        return self._n_ids

    def n_parameters(self, exclude_bottom_level=False):
        """
        Returns the number of parameters.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter are counted in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        """
        if exclude_bottom_level:
            return self._n_top

        return self._n_parameters

    def n_observations(self):
        """
        Returns the number of observed data points per individual.
        """
        return self._n_obs


class HierarchicalLogPosterior(pints.LogPDF):
    r"""
    A hierarchical log-posterior is defined by a hierarchical log-likelihood
    and a log-prior for the population (or top-level) parameters.

    The log-posterior takes an instance of a
    :class:`HierarchicalLogLikelihood` and an instance of a
    :class:`pints.LogPrior` of the same dimensionality
    as population (or top-level) parameters in the log-likelihood.

    Formally the log-posterior is defined as

    .. math::
        \log p(\Psi , \theta | X ^{\text{obs}}) =
            \log p(\mathcal{D}, \Psi | \theta) + \log p(\theta ) +
            \text{constant},

    where :math:`\Psi` are the bottom-level parameters, :math:`\theta` are the
    top-level parameters and :math:`\mathcal{D}` is the data, see
    :class:`HierarchicalLogLikelihood`.

    Extends :class:`pints.LogPDF`.

    :param log_likelihood: A log-likelihood for the individual and population
        parameters.
    :type log_likelihood: HierarchicalLogLikelihood
    :param log_prior: A log-prior for the population (or top-level) parameters.
    :type log_prior: pints.LogPrior
    """
    def __init__(self, log_likelihood, log_prior):
        # Check inputs
        super(HierarchicalLogPosterior, self).__init__()

        # Check inputs
        if not isinstance(log_likelihood, HierarchicalLogLikelihood):
            raise TypeError(
                'The log-likelihood has to be an instance of a '
                'chi.HierarchicalLogLikelihood.')
        if not isinstance(log_prior, pints.LogPrior):
            raise TypeError(
                'The log-prior has to be an instance of a pints.LogPrior.')

        # Check dimensions
        n_top_parameters = log_likelihood.n_parameters(
            exclude_bottom_level=True)
        if log_prior.n_parameters() != n_top_parameters:
            raise ValueError(
                'The log-prior has to have as many parameters as population '
                'parameters in the log-likelihood. There are '
                '<' + str(n_top_parameters) + '> population parameters.')

        # Store prior and log_likelihood, as well as number of parameters
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood
        self._n_parameters = log_likelihood.n_parameters()

        # Create mask for top-level parameters
        self._create_top_level_mask()

    def __call__(self, parameters):
        # Convert parameters
        parameters = np.asarray(parameters)

        # Evaluate log-prior first, assuming this is very cheap
        score = self._log_prior(parameters[self._top_level_mask])
        if np.isinf(score):
            return score

        return score + self._log_likelihood(parameters)

    def _create_top_level_mask(self):
        """
        Creates a mask that can be used to mask for the top level
        parameters.

        Uses that bottom-level parameters can only be top-level parameters
        when they are heterogeneously modelled,
        and that all population parameters are top-level.
        """
        # Enumerate bottom-level parameters
        n_ids = self._log_likelihood.n_log_likelihoods()
        n_dim = self._log_likelihood.get_population_model().n_dim()
        pop_models = self._log_likelihood.get_population_model(
            ).get_population_models()
        current_dim = 0
        n_pooled_dim = 0
        n_hetero_dims = []
        for pop_model in pop_models:
            end_dim = current_dim + pop_model.n_dim()
            if isinstance(pop_model, chi.PooledModel):
                n_pooled_dim += pop_model.n_dim()
                continue
            if isinstance(pop_model, chi.HeterogeneousModel):
                n_hetero_dims.append([current_dim, end_dim])
            current_dim = end_dim
        bottom_parameters = np.arange(
            n_ids*(n_dim-n_pooled_dim)).reshape(n_ids, n_dim-n_pooled_dim)

        # Keep indices of heterogeneously modelled parameters
        bottom_params = []
        for info in n_hetero_dims:
            start_dim, end_dim = info
            bottom_params.append(bottom_parameters[:, start_dim:end_dim])
        bottom_parameters = bottom_params
        if len(bottom_parameters) > 0:
            bottom_parameters = np.hstack(bottom_parameters)

        # Append indices of population paramters
        n_pop = self._log_likelihood.get_population_model().n_parameters()
        n_parameters = self._log_likelihood.n_parameters()
        pop_parameters = np.arange(n_parameters-n_pop, n_parameters)

        self._top_level_mask = np.hstack(
            [bottom_parameters, pop_parameters]).astype(int)

    def evaluateS1(self, parameters):
        """
        Returns the log-posterior score and its sensitivities to the model
        parameters.

        :param parameters: An array-like object with parameter values.
        :type parameters: List[float], numpy.ndarray
        """
        # Convert parameters
        parameters = np.asarray(parameters)

        # Evaluate log-prior first, assuming this is very cheap
        score, sens = self._log_prior.evaluateS1(
            parameters[self._top_level_mask])

        if np.isinf(score):
            return score, np.full(shape=len(parameters), fill_value=np.inf)

        # Add log-likelihood score and sensitivities
        ll_score, sensitivities = self._log_likelihood.evaluateS1(
            parameters)

        score += ll_score
        sensitivities[self._top_level_mask] += sens

        return score, sensitivities

    def get_log_likelihood(self):
        """
        Returns the log-likelihood.
        """
        return self._log_likelihood

    def get_log_prior(self):
        """
        Returns the log-prior.
        """
        return self._log_prior

    def get_id(self, unique=False):
        """
        Returns the ids of the log-posterior's parameters. If the ID is
        ``None`` corresponding parameter is defined on the population level.

        :param unique: A boolean flag which indicates whether each ID is only
            returned once, or whether the IDs of all paramaters are returned.
        :type unique: bool, optional
        """
        return self._log_likelihood.get_id(unique)

    def get_parameter_names(
            self, exclude_bottom_level=False, include_ids=False):
        """
        Returns the names of the parameters.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter names are returned in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        :param include_ids: A boolean flag which determines whether the IDs
            (prefixes) of the model parameters are included.
        :type include_ids: bool, optional
        """
        # Get parameter names
        names = self._log_likelihood.get_parameter_names(
            exclude_bottom_level, include_ids)

        return names

    def n_parameters(self, exclude_bottom_level=False):
        """
        Returns the number of parameters.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter are counted in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        """
        return self._log_likelihood.n_parameters(exclude_bottom_level)


class LogLikelihood(pints.LogPDF):
    r"""
    A log-likelihood quantifies how likely a model for a set of
    parameters is to explain some observed biomarker values.

    A log-likelihood takes an instance of a :class:`MechanisticModel` and one
    instance of an :class:`ErrorModel` for each mechanistic model output. This
    defines a time-dependent distribution of observable biomarkers
    equivalent to a :class:`PredictiveModel`

    .. math::
        p(x | t; \psi ),

    which is centered at the mechanistic model output and has a variance
    according to the error model. Here, :math:`x` are the observable biomarker
    values at time :math:`t`, and :math:`\psi` are the model parameters of the
    mechanistic model and the error model. For multiple outputs of the
    mechanistic model, :math:`p` will be a multivariate distribution.

    The log-likelihood for observations
    :math:`(x^{\text{obs}}, t^{\text{obs}})` is given by

    .. math::
        L(\psi | x^{\text{obs}}) = \sum _{i=1}^n
        \log p(x^{\text{obs}}_i | t^{\text{obs}}_i; \psi),

    where :math:`n` is the total number of observations. Note that for
    notational ease we omitted the conditioning on the observation times
    :math:`t^{\text{obs}}` on the left hand side, and will also often drop
    it elsewhere in the documentation.

    .. note::
        For notational ease we omitted that the log-likelihood also is
        conditional on the dosing regimen associated with the observations.
        The appropriate regimen can be set with
        :meth:`PharmacokineticModel.set_dosing_regimen`

    Extends :class:`pints.LogPDF`.

    :param mechanistic_model: A mechanistic model that models the
        simplified behaviour of the biomarkers.
    :type mechanistic_model: MechanisticModel
    :param error_model:
        One error model for each output of the mechanistic model. For multiple
        ouputs the error models are expected to be ordered according to the
        outputs.
    :type error_model: ErrorModel, list[ErrorModel]
    :param observations: A list of one dimensional array-like objects with
        measured values of the biomarkers. The list is expected to be ordered
        in the same way as the mechanistic model outputs.
    :type observations: list[float], list[list[float]]
    :param times: A list of one dimensional array-like objects with measured
        times associated to the observations.
    :type times: list[float], list[list[float]]
    :param outputs: A list of output names, which sets the mechanistic model
        outputs. If ``None`` the currently set outputs of the mechanistic model
        are assumed.
    :type outputs: list[str], optional

    Example
    -------

    ::

        import chi

        # Define mechanistic and error model
        sbml_file = chi.ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(sbml_file)
        error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()

        # Define observations
        observations = [1, 2, 3, 4]
        times = [0, 0.5, 1, 2]

        # Create log-likelihood
        log_likelihood = chi.LogLikelihood(
            mechanistic_model,
            error_model,
            observations,
            times)

        # Compute log-likelihood score
        parameters = [1, 1, 1, 1, 1, 1, 1]
        score = log_likelihood(parameters)  # -5.4395320556329265
    """
    def __init__(
            self, mechanistic_model, error_model, observations, times,
            outputs=None):
        super(LogLikelihood, self).__init__()

        # Check inputs
        if not isinstance(mechanistic_model, chi.MechanisticModel):
            raise TypeError(
                'The mechanistic model as to be an instance of a '
                'chi.MechanisticModel.')

        if not isinstance(error_model, list):
            error_model = [error_model]

        # Copy mechanistic model
        mechanistic_model = mechanistic_model.copy()

        # Set outputs
        if outputs is not None:
            mechanistic_model.set_outputs(outputs)

        n_outputs = mechanistic_model.n_outputs()
        if len(error_model) != n_outputs:
            raise ValueError(
                'One error model has to be provided for each mechanistic '
                'model output.')

        for em in error_model:
            if not isinstance(
                    em, (chi.ErrorModel, chi.ReducedErrorModel)):
                raise TypeError(
                    'The error models have to instances of a '
                    'chi.ErrorModel.')

        if n_outputs == 1:
            # For single-output problems the observations can be provided as a
            # simple one dimensional list / array. To match the multi-output
            # scenario wrap values by a list
            if len(observations) != n_outputs:
                observations = [observations]

            if len(times) != n_outputs:
                times = [times]

        if len(observations) != n_outputs:
            raise ValueError(
                'The observations have the wrong length. For a '
                'multi-output problem the observations are expected to be '
                'a list of array-like objects with measurements for each '
                'of the mechanistic model outputs.')

        if len(times) != n_outputs:
            raise ValueError(
                'The times have the wrong length. For a multi-output problem '
                'the times are expected to be a list of array-like objects '
                'with the measurement time points for each of the mechanistic '
                'model outputs.')

        # Transform observations and times to read-only arrays
        observations = [pints.vector(obs) for obs in observations]
        times = [pints.vector(ts) for ts in times]

        # Make sure times are strictly increasing
        for ts in times:
            if np.any(ts < 0):
                raise ValueError('Times cannot be negative.')
            if np.any(ts[:-1] > ts[1:]):
                raise ValueError('Times must be increasing.')

        # Make sure that the observation-time pairs match
        for output_id, output_times in enumerate(times):
            if observations[output_id].shape != output_times.shape:
                raise ValueError(
                    'The observations and times have to be of the same '
                    'dimension.')

            # Sort times and observations
            order = np.argsort(output_times)
            times[output_id] = output_times[order]
            observations[output_id] = observations[output_id][order]

        # Copy error models, such that renaming doesn't affect input models
        error_model = [
            copy.deepcopy(em) for em in error_model]

        # Remember models and observations
        self._mechanistic_model = mechanistic_model
        self._error_models = error_model
        self._observations = observations
        self._n_obs = [len(obs) for obs in observations]

        self._arange_times_for_mechanistic_model(times)

        # Set parameter names and number of parameters
        self._set_error_model_parameter_names()
        self._set_number_and_parameter_names()

        # Set default ID
        self._id = None

    def __call__(self, parameters):
        """
        Computes the log-likelihood score of the parameters.
        """
        # Check that mechanistic model has sensitivities disabled
        # (Simply for performance)
        if self._mechanistic_model.has_sensitivities():
            self._mechanistic_model.enable_sensitivities(False)

        # Solve the mechanistic model
        try:
            outputs = self._mechanistic_model.simulate(
                parameters=parameters[:self._n_mechanistic_params],
                times=self._times)
        except (myokit.SimulationError, Exception) as e:  # pragma: no cover
            warnings.warn(
                'An error occured while solving the mechanistic model: \n'
                + str(e) + '.\n A score of -infinity is returned.',
                RuntimeWarning)
            return -np.infty

        # Remember only error parameters
        parameters = parameters[self._n_mechanistic_params:]

        # Compute log-likelihood score
        score = 0
        start = 0
        for output_id, error_model in enumerate(self._error_models):
            # Get relevant mechanistic model outputs and parameters
            output = outputs[output_id, self._obs_masks[output_id]]
            end = start + self._n_error_params[output_id]

            # Compute log-likelihood score for this output
            score += error_model.compute_log_likelihood(
                parameters=parameters[start:end],
                model_output=output,
                observations=self._observations[output_id])

            # Shift start index
            start = end

        return score

    def _arange_times_for_mechanistic_model(self, times):
        """
        Sets the evaluation time points for the mechanistic model.

        The challenge is to avoid solving the mechanistic model multiple
        times for each observed output separately. So here we define a
        union of all time points and track which time points correspond
        to observations.
        """
        # Get unique times and sort them
        unique_times = []
        for output_times in times:
            unique_times += list(output_times)
        unique_times = set(unique_times)
        unique_times = sorted(unique_times)
        unique_times = pints.vector(unique_times)

        # Create a container for the observation masks
        n_outputs = len(times)
        n_unique_times = len(unique_times)
        obs_masks = np.zeros(shape=(n_outputs, n_unique_times), dtype=bool)

        # Find relevant time points for each output
        for output_id, output_times in enumerate(times):
            if np.array_equal(output_times, unique_times):
                n_times = len(output_times)
                obs_masks[output_id] = np.ones(shape=n_times, dtype=bool)

                # Continue to the next iteration
                continue

            for time in output_times:
                # If time is in unique times, flip position to True
                if time in unique_times:
                    mask = unique_times == time
                    obs_masks[output_id, mask] = True

        self._times = pints.vector(unique_times)
        self._obs_masks = obs_masks

    def _set_error_model_parameter_names(self):
        """
        Resets the error model parameter names and prepends the output name
        if more than one output exists.
        """
        # Reset error model parameter names to defaults
        for error_model in self._error_models:
            error_model.set_parameter_names(None)

        # Rename error model parameters, if more than one output
        n_outputs = self._mechanistic_model.n_outputs()
        if n_outputs > 1:
            # Get output names
            outputs = self._mechanistic_model.outputs()

            for output_id, error_model in enumerate(self._error_models):
                # Get original parameter names
                names = error_model.get_parameter_names()

                # Prepend output name
                output = outputs[output_id]
                names = [output + ' ' + name for name in names]

                # Set new parameter names
                error_model.set_parameter_names(names)

    def _set_number_and_parameter_names(self):
        """
        Sets the number and names of the free model parameters.
        """
        # Get mechanistic model parameters
        parameter_names = self._mechanistic_model.parameters()

        # Get error model parameters
        n_error_params = []
        for error_model in self._error_models:
            parameter_names += error_model.get_parameter_names()
            n_error_params.append(error_model.n_parameters())

        # Update number and names
        self._parameter_names = parameter_names
        self._n_parameters = len(self._parameter_names)

        # Get number of mechanistic and error model parameters
        self._n_mechanistic_params = self._mechanistic_model.n_parameters()
        self._n_error_params = n_error_params

    def compute_pointwise_ll(self, parameters):
        """
        Returns the pointwise log-likelihood scores of the parameters for
        each observation.

        :param parameters: A list of parameter values
        :type parameters: list, numpy.ndarray
        """
        # Check that mechanistic model has sensitivities disabled
        # (Simply for performance)
        if self._mechanistic_model.has_sensitivities():
            self._mechanistic_model.enable_sensitivities(False)

        # Solve the mechanistic model
        outputs = self._mechanistic_model.simulate(
            parameters=parameters[:self._n_mechanistic_params],
            times=self._times)

        # Remember only error parameters
        parameters = parameters[self._n_mechanistic_params:]

        # Compute the pointwise log-likelihood score
        start = 0
        pointwise_ll = []
        for output_id, error_model in enumerate(self._error_models):
            # Get relevant mechanistic model outputs and parameters
            output = outputs[output_id, self._obs_masks[output_id]]
            end = start + self._n_error_params[output_id]

            # Compute pointwise log-likelihood scores for this output
            pointwise_ll.append(
                error_model.compute_pointwise_ll(
                    parameters=parameters[start:end],
                    model_output=output,
                    observations=self._observations[output_id]))

            # Shift start indices
            start = end

        return np.hstack(pointwise_ll)

    def evaluateS1(self, parameters):
        """
        Computes the log-likelihood of the parameters and its
        sensitivities.

        :param parameters: A list of parameter values
        :type parameters: list, numpy.ndarray
        """
        # Check that mechanistic model has sensitivities enabled
        if not self._mechanistic_model.has_sensitivities():
            self._mechanistic_model.enable_sensitivities(True)

        # Solve the mechanistic model
        try:
            outputs, senss = self._mechanistic_model.simulate(
                parameters=parameters[:self._n_mechanistic_params],
                times=self._times)
        except (myokit.SimulationError, Exception) as e:  # pragma: no cover
            warnings.warn(
                'An error occured while solving the mechanistic model: \n'
                + str(e) + '.\n A score of -infinity is returned.',
                RuntimeWarning)
            n_parameters = len(parameters)
            return -np.infty, np.full(shape=n_parameters, fill_value=np.infty)

        # Remember only error parameters
        parameters = parameters[self._n_mechanistic_params:]

        # Compute log-likelihood score
        start = 0
        score = 0
        n_mech = self._n_mechanistic_params
        sensitivities = np.zeros(shape=self._n_parameters)
        for output_id, error_model in enumerate(self._error_models):
            # Get relevant mechanistic model outputs and sensitivities
            output = outputs[output_id, self._obs_masks[output_id]]
            sens = senss[self._obs_masks[output_id], output_id, :]
            end = start + self._n_error_params[output_id]

            # Compute log-likelihood score for this output
            l, s = error_model.compute_sensitivities(
                parameters=parameters[start:end],
                model_output=output,
                model_sensitivities=sens,
                observations=self._observations[output_id])

            # Aggregate Log-likelihoods and sensitivities
            score += l
            sensitivities[:n_mech] += s[:n_mech]
            sensitivities[n_mech+start:n_mech+end] += s[n_mech:]

            # Shift start index
            start = end

        return score, sensitivities

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``
        sets the parameter free again.

        :param name_value_dict: A dictionary with model parameter names as
            keys, and parameter value as values.
        :type name_value_dict: dict[str, float]
        """
        # Check type of dictionanry
        try:
            name_value_dict = dict(name_value_dict)
        except (TypeError, ValueError):
            raise ValueError(
                'The name-value dictionary has to be convertable to a python '
                'dictionary.')

        # Get submodels
        mechanistic_model = self._mechanistic_model
        error_models = self._error_models

        # Convert models to reduced models
        if not isinstance(mechanistic_model, chi.ReducedMechanisticModel):
            mechanistic_model = chi.ReducedMechanisticModel(mechanistic_model)
        for model_id, error_model in enumerate(error_models):
            if not isinstance(error_model, chi.ReducedErrorModel):
                error_models[model_id] = chi.ReducedErrorModel(error_model)

        # Fix model parameters
        mechanistic_model.fix_parameters(name_value_dict)
        for error_model in error_models:
            error_model.fix_parameters(name_value_dict)

        # If no parameters are fixed, get original model back
        if mechanistic_model.n_fixed_parameters() == 0:
            mechanistic_model = mechanistic_model.mechanistic_model()

        for model_id, error_model in enumerate(error_models):
            if error_model.n_fixed_parameters() == 0:
                error_model = error_model.get_error_model()
                error_models[model_id] = error_model

        # Safe reduced models
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models

        # Update names and number of parameters
        self._set_number_and_parameter_names()

    def get_id(self, *args, **kwargs):
        """
        Returns the ID of the log-likelihood. If not set, ``None`` is returned.

        The ID is used as meta data to identify the origin of the data.
        """
        return self._id

    def get_parameter_names(self):
        """
        Returns the parameter names of the predictive model.
        """
        return copy.copy(self._parameter_names)

    def get_submodels(self):
        """
        Returns the submodels of the log-likelihood in form of a dictionary.

        .. warning::
            The returned submodels are only references to the models used by
            the log-likelihood. Changing e.g. the dosing regimen of the
            :class:`MechanisticModel` will therefore also change the dosing
            regimen of the log-likelihood!
        """
        # Get original submodels
        mechanistic_model = self._mechanistic_model
        if isinstance(mechanistic_model, chi.ReducedMechanisticModel):
            mechanistic_model = mechanistic_model.mechanistic_model()

        error_models = []
        for error_model in self._error_models:
            # Get original error model
            if isinstance(error_model, chi.ReducedErrorModel):
                error_model = error_model.get_error_model()

            error_models.append(error_model)

        submodels = dict({
            'Mechanistic model': mechanistic_model,
            'Error models': error_models})

        return submodels

    def n_parameters(self):
        """
        Returns the number of parameters.
        """
        return self._n_parameters

    def n_observations(self):
        """
        Returns the number of observed data points for each output.
        """
        return self._n_obs

    def set_id(self, label):
        """
        Sets the ID of the log-likelihood.

        The ID is used as meta data to identify the origin of the data.

        :param label: Integer value which is used as ID for the
            log-likelihood.
        :type label: str
        """
        if isinstance(label, float):
            # Just in case floats are used as labels
            label = int(label)

        # Construct ID as <ID: #> for convenience
        self._id = str(label)


class LogPosterior(pints.LogPDF):
    r"""
    A log-posterior constructed from a log-likelihood and a log-prior.

    The log-posterior takes an instance of a :class:`LogLikelihood` and
    an instance of a :class:`pints.LogPrior` of the same dimensionality
    as parameters in the log-likelihood.

    Formally the log-posterior is given by the sum of the input log-likelihood
    :math:`L(\psi | x^{\text{obs}})` and the input log-prior
    :math:`\log p(\psi )` up to an additive constant

    .. math::
        \log p(\psi | x ^{\text{obs}}) \sim
        L(\psi | x^{\text{obs}}) + \log p(\psi ),

    where :math:`\psi` are the parameters of the log-likelihood and
    :math:`x ^{\text{obs}}` are the observed data. The additive constant
    is the normalisation of the log-posterior and is in general not known.

    Extends :class:`pints.LogPDF`.

    :param log_likelihood: A log-likelihood for the model parameters.
    :type log_likelihood: LogLikelihood
    :param log_prior: A log-prior for the model parameters. The log-prior
        has to have the same dimensionality as the log-likelihood.
    :type log_prior: pints.LogPrior

    Example
    -------

    ::

        import chi
        import pints

        # Define mechanistic and error model
        sbml_file = chi.ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(sbml_file)
        error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()

        # Define observations
        observations = [1, 2, 3, 4]
        times = [0, 0.5, 1, 2]

        # Create log-likelihood
        log_likelihood = chi.LogLikelihood(
            mechanistic_model,
            error_model,
            observations,
            times)

        # Define log-prior
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.HalfCauchyLogPrior(0, 1),
            pints.HalfCauchyLogPrior(0, 1))

        # Create log-posterior
        log_posterior = chi.LogPosterior(log_likelihood, log_prior)

        # Compute log-posterior score
        parameters = [1, 1, 1, 1, 1, 1, 1]
        score = log_posterior(parameters)  # -14.823684493355092
    """
    def __init__(self, log_likelihood, log_prior):
        # Check inputs
        super(LogPosterior, self).__init__()

        # Check inputs
        if not isinstance(log_likelihood, LogLikelihood):
            raise TypeError(
                'The log-likelihood has to extend chi.LogLikelihood.')
        if not isinstance(log_prior, pints.LogPrior):
            raise TypeError(
                'The log-prior has to extend pints.LogPrior.')

        # Check dimensions
        n_parameters = log_prior.n_parameters()
        if log_likelihood.n_parameters() != n_parameters:
            raise ValueError(
                'The log-prior and the log-likelihood must have same '
                'dimension.')

        # Store prior and log_likelihood, as well as number of parameters
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood
        self._n_parameters = n_parameters

    def __call__(self, parameters):
        # Evaluate log-prior first, assuming this is very cheap
        score = self._log_prior(parameters)
        if np.isinf(score):
            return score

        return score + self._log_likelihood(parameters)

    def evaluateS1(self, parameters):
        """
        Returns the log-posterior score and its sensitivities to the model
        parameters.

        :param parameters: An array-like object with parameter values.
        :type parameters: List[float], numpy.ndarray
        """
        # Evaluate log-prior first, assuming this is very cheap
        score, sensitivities = self._log_prior.evaluateS1(parameters)
        if np.isinf(score):
            return score, sensitivities

        # Compute log-likelihood and sensitivities
        l, s = self._log_likelihood.evaluateS1(parameters)

        # Aggregate scores
        score += l
        sensitivities += s

        return score, sensitivities

    def get_log_likelihood(self):
        """
        Returns the log-likelihood.
        """
        return self._log_likelihood

    def get_log_prior(self):
        """
        Returns the log-prior.
        """
        return self._log_prior

    def get_id(self):
        """
        Returns the id of the log-posterior. If no id is set, ``None`` is
        returned.
        """
        return self._log_likelihood.get_id()

    def get_parameter_names(self):
        """
        Returns the names of the model parameters. By default the parameters
        are enumerated and assigned with the names 'Param #'.
        """
        # Get parameter names
        names = self._log_likelihood.get_parameter_names()

        return names

    def n_parameters(self, *args, **kwargs):
        """
        Returns the number of parameters of the posterior.
        """
        return self._n_parameters


class ReducedLogPDF(pints.LogPDF):
    """
    A wrapper for a :class:`pints.LogPDF` to fix the values of a subset of
    model parameters.

    This allows to reduce the parameter dimensionality of the log-pdf
    at the cost of fixing some parameters at a constant value.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    log_pdf
        An instance of a :class:`pints.LogPDF`.
    mask
        A boolean array of the length of the number of parameters. ``True``
        indicates that the parameter is fixed at a constant value, ``False``
        indicates that the parameter remains free.
    values
        A list of values the parameters are fixed at.
    """

    def __init__(self, log_pdf, mask, values):
        super(ReducedLogPDF, self).__init__()

        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError(
                'The log-pdf has to be an instance of a pints.LogPDF.')

        self._log_pdf = log_pdf

        if len(mask) != self._log_pdf.n_parameters():
            raise ValueError(
                'Length of mask has to match the number of log-pdf '
                'parameters.')

        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise ValueError(
                'Mask has to be a boolean array.')

        n_fixed = int(np.sum(mask))
        if n_fixed != len(values):
            raise ValueError(
                'There have to be as many value inputs as the number of '
                'fixed parameters.')

        # Create a parameter array for later calls of the log-pdf
        self._parameters = np.empty(shape=len(mask))
        self._parameters[mask] = np.asarray(values)

        # Allow for updating the 'free' number of parameters
        self._mask = ~mask
        self._n_parameters = int(np.sum(self._mask))

    def __call__(self, parameters):
        # Fill in 'free' parameters
        self._parameters[self._mask] = np.asarray(parameters)

        return self._log_pdf(self._parameters)

    def n_parameters(self):
        """
        Returns the number of free parameters of the log-posterior.
        """
        return self._n_parameters
