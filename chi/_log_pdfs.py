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

        # Get number of parameters as well as pooled or heterogen. dimensions
        self._population_model.set_n_ids(self._n_ids)
        info = self._count_parameters()
        self._n_parameters, self._special_dims = info[:2]
        self._n_pooled_dim, self._n_hetero_dim = info[2:]
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
        n_hetero_dims = 0
        special_dims = []
        current_dim = 0
        current_top_index = 0
        for pop_model in pop_models:
            # Check whether dimension is pooled
            n_bottom, n_top = pop_model.n_hierarchical_parameters(self._n_ids)
            n_dim = pop_model.n_dim()
            is_pooled = isinstance(pop_model, chi.PooledModel)
            is_heterogen = isinstance(pop_model, chi.HeterogeneousModel)
            if is_pooled or is_heterogen:
                # Remember start and end of special dimensions,
                # Start and end of parameter values,
                # and whether it's pooled or heterogeneous
                special_dims.append([
                    current_dim, current_dim + n_dim,
                    current_top_index,
                    current_top_index + n_top,
                    is_pooled])
            if is_pooled:
                n_pooled_dims += n_dim
            if is_heterogen:
                n_hetero_dims += n_dim

            # Count overall number of parameters
            n_parameters += n_bottom + n_top
            current_dim += n_dim
            current_top_index += n_top

        return n_parameters, special_dims, n_pooled_dims, n_hetero_dims

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

    def _remove_duplicates(self, sensitivities):
        """
        In some sense the reverse of self._reshape_bottom_parameters.

        1. Pooled bottom parameters need to be added to population parameter
        2. Heterogeneous bottom parameters need to added to population
            parameters
        """
        # Check for quick solution 1: no pooled parameters and no heterogeneous
        # parameters
        if (self._n_pooled_dim == 0) and (self._n_hetero_dim == 0):
            return sensitivities

        # Get population parameter sensitvitities
        start_pop = self._n_ids * self._n_dim
        sens = np.zeros(shape=self._n_parameters)
        sens[self._n_bottom:] = sensitivities[start_pop:]

        # Check for quick solution 2: all parameters heterogen.
        if self._n_hetero_dim == self._n_dim:
            # Add contributions from bottom-level parameters
            # (Population sens. are actually zero, so we can just replace them)
            sens = sensitivities[:start_pop]
            return sens

        # Check for quick solution 3: all parameters pooled
        if self._n_pooled_dim == self._n_dim:
            # Add contributions from bottom-level parameters
            # (Population sens. are actually zero, so we can just replace them)
            sens = np.sum(
                sensitivities[:start_pop].reshape(self._n_ids, self._n_dim),
                axis=0)
            return sens

        shift = 0
        current_dim = 0
        bottom_sensitivities = sensitivities[:start_pop].reshape(
            self._n_ids, self._n_dim)
        bottom_sens = np.zeros(shape=(
            self._n_ids,
            self._n_dim - self._n_pooled_dim - self._n_hetero_dim))
        for info in self._special_dims:
            start_dim, end_dim, start_top, end_top, is_pooled = info
            # Fill leading regular dims
            bottom_sens[:, current_dim-shift:start_dim-shift] = \
                bottom_sensitivities[:, current_dim:start_dim]
            # Fill special dims
            if is_pooled:
                sens[self._n_bottom+start_top:self._n_bottom+end_top] = \
                    np.sum(bottom_sensitivities[:, start_dim:end_dim], axis=0)
            else:
                sens[self._n_bottom+start_top:self._n_bottom+end_top] = \
                    bottom_sensitivities[:, start_dim:end_dim].flatten()
            current_dim = end_dim
            shift += end_dim - start_dim
        # Fill trailing regular dims
        bottom_sens[:, current_dim-shift:] = bottom_sensitivities[
            :, current_dim:]

        # Add bottom sensitivties
        sens[:self._n_bottom] = bottom_sens.flatten()

        return sens

    def _reshape_bottom_parameters(self, bottom_parameters, top_parameters):
        """
        Takes bottom parameters and top parameters with no duplicates and
        returns bottom parameters of shape (n_ids, n_dim), where pooled
        parameters are duplicated.
        """
        bottom_params = np.empty(shape=(self._n_ids, self._n_dim))

        # Check for quick solution 1: no pooled parameters and no heterogen.
        if (self._n_pooled_dim == 0) and (self._n_hetero_dim == 0):
            bottom_params[:, :] = bottom_parameters.reshape(
                self._n_ids, self._n_dim)
            return bottom_params

        # Check for quick solution 2: all parameters pooled
        if self._n_pooled_dim == self._n_dim:
            bottom_params[:, :] = top_parameters[np.newaxis, :]
            return bottom_params

        # Check for quick solution 3: all parameters heterogen.
        if self._n_hetero_dim == self._n_dim:
            bottom_params[:, :] = top_parameters.reshape(
                self._n_ids, self._n_dim)
            return bottom_params

        shift = 0
        current_dim = 0
        bottom_parameters = bottom_parameters.reshape(
            self._n_ids,
            self._n_dim - self._n_pooled_dim - self._n_hetero_dim)
        for info in self._special_dims:
            start_dim, end_dim, start_top, end_top, is_pooled = info
            # Fill leading non-pooled dims
            bottom_params[:, current_dim:start_dim] = bottom_parameters[
                :, current_dim-shift:start_dim-shift]
            # Fill special dims
            dims = end_dim - start_dim
            if is_pooled:
                bottom_params[:, start_dim:end_dim] = top_parameters[
                    start_top:end_top]
            else:
                bottom_params[:, start_dim:end_dim] = top_parameters[
                    start_top:end_top].reshape(self._n_ids, dims)
            current_dim = end_dim
            shift += dims
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
        # if np.any(self._uses_eta):
        #     raise NotImplementedError(
        #         'This method is not implemented for '
        #         'CovariatePopulationModels.'
        #     )

        # # Transform parameters to numpy array
        # parameters = np.asarray(parameters)

        # # Compute population model scores of individuals
        # start = 0
        # pop_scores = np.zeros(shape=self._n_ids)
        # for pop_model in self._population_models:
        #     # Get number of individual and population level parameters
        #     n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)

        #     # Get parameter ranges
        #     end_indiv = start + n_indiv
        #     end_pop = end_indiv + n_pop

        #     # Add score, if individual parameters exist
        #     if n_indiv > 0:
        #         pop_scores += pop_model.compute_pointwise_ll(
        #             parameters=parameters[end_indiv:end_pop],
        #             observations=parameters[start:end_indiv])

        #     # Shift start index
        #     start = end_pop

        # if per_individual is True:
        #     # Compute aggregated individual likelihoods
        #     pw_log_likelihoods = pop_scores
        #     for index, log_likelihood in enumerate(self._log_likelihoods):
        #         # Compute scores for each observation
        #         pw_log_likelihoods[index] += log_likelihood(
        #             parameters[self._indiv_params[index]])

        #     return pw_log_likelihoods

        # # Evaluate individual likelihoods pointwise
        # pw_log_likelihoods = []
        # for index, log_likelihood in enumerate(self._log_likelihoods):
        #     # Compute scores for each observation
        #     scores = log_likelihood.compute_pointwise_ll(
        #         parameters[self._indiv_params[index]])

        #     # Add population contribution
        #     scores += pop_scores[index] / len(scores)

        #     # Safe scores
        #     pw_log_likelihoods.append(scores)

        # return np.hstack(pw_log_likelihoods)

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
        sensitivities = self._remove_duplicates(sensitivities)

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
        # Get bottom parameter names
        # (pooled and heterogenous parameters count as top parameters)
        current_dim = 0
        names = self._log_likelihoods[0].get_parameter_names()
        n = []
        for info in self._special_dims:
            start_dim, end_dim, _, _, _ = info
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
            names = names[self._n_bottom:]

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
            return self._population_model.n_parameters()

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
        n_top = log_likelihood.n_parameters(
            exclude_bottom_level=True)
        if log_prior.n_parameters() != n_top:
            raise ValueError(
                'The log-prior has to have as many parameters as population '
                'parameters in the log-likelihood. There are '
                '<' + str(n_top) + '> population parameters.')

        # Store prior and log_likelihood, as well as number of parameters
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood
        self._n_parameters = log_likelihood.n_parameters()
        self._n_bottom = self._n_parameters - n_top

    def __call__(self, parameters):
        # Convert parameters
        parameters = np.asarray(parameters)

        # Evaluate log-prior first, assuming this is very cheap
        score = self._log_prior(parameters[self._n_bottom:])
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
        # Convert parameters
        parameters = np.asarray(parameters)

        # Evaluate log-prior first, assuming this is very cheap
        score, sens = self._log_prior.evaluateS1(parameters[self._n_bottom:])

        if np.isinf(score):
            return score, np.full(shape=len(parameters), fill_value=np.inf)

        # Add log-likelihood score and sensitivities
        ll_score, sensitivities = self._log_likelihood.evaluateS1(
            parameters)

        score += ll_score
        sensitivities[self._n_bottom:] += sens

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

    def get_id(self, *args, **kwargs):
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


class PopulationFilterLogPosterior(HierarchicalLogPosterior):
    r"""
    A population filter log-posterior approximates a hierarchical
    log-posterior.

    Population filter log-posteriors can be used to approximate hierarchical
    log-posteriors when exact hierarchical inference becomes numerically
    intractable. The canonical use case for population filter inference is the
    inference from time series snapshot data.

    The population filter log-posterior is defined by a population filter,
    a mechanistic model, an error model, a population model and the data

    .. math::
        \log p(\theta , \tilde{\Psi}, \tilde{Y} | \mathcal{D}) =
            \sum _{ij} \log p (y_{ij} | \tilde{Y}_j) +
            \sum _{sj} \log p (\tilde{y}_{sj} | \tilde{\psi}_s, t_j) +
            \sum _{sk} \log p (\tilde{psi}_{sk} | \theta _k) +
            \sum _{k} \log p (\theta _k) + \mathrm{constant}.

    The first term is the population filter contribution which estimates the
    log-likelihood of simulated measurements,
    :math:`\tilde{Y}_j = \{ \tilde{y}_{sj}\}`, at time
    :math:`t_j` to come from the same distribution as the observations,
    :math:`Y_j = \{ y_{ij}\}`. Here, :math:`s` indexes a simulated individual
    and :math:`i` indexes an individual from the dataset.
    The second term is the contribution from the
    log-likelihood of the simulated individual parameters
    :math:`\tilde{\Psi} = \{ \tilde{\psi} _s\}` for the simulated
    measurements. This log-likelihood is defined by the mechanistic
    model and the error model. The third term is the contribution from the
    log-likelihood of the population parameters
    :math:`\theta = \{ \theta _k \}` to govern the distribution of the
    individual parameters. The final contribution is from the log-prior
    of the population parameters.

    Note that the choice of population filter makes assumptions about the
    distributional shape of the measurements which can influence the inference
    results.

    :param population_filter: The population filter which connects the
        observations to the simulated measurements.
    :type population_filter: chi.PopulationFilter
    :param times: Measurement time points of the data.
    :type times: np.ndarray of shape ``(n_times,)``
    :param mechanistic_model: A mechanistic model for the dynamics. The outputs
        of the mechanistic model are expected to be in the same order as the
        observables in ``observations``.
    :type mechanistic_model: chi.MechanisticModel
    :param population_model: A population model with the same dimensionality
        as the number of mechanistic model parameters. The dimensions are
        expected to be in the same order as the model parameters.
    :type population_models: chi.PopulationModel
    :param log_prior: Log-prior for the population level parameters.
        The prior dimensions are expected to be in the order of the population
        models.
    :type log_prior: pints.LogPrior
    :param sigma: Standard deviation of the Gaussian error model. If ``None``
        the parameter is inferred from the data.
    :type sigma: List[float] of length ``(n_observables)``, optional
    :param error_on_log_scale: A boolean flag indicating whether the error
        model models the residuals of the mechanistic model directly or on
        a log scale.
    :type error_on_log_scale: bool, optional
    """
    def __init__(
            self, population_filter, times, mechanistic_model,
            population_model, log_prior, sigma=None, error_on_log_scale=False,
            n_samples=100):

        # Check filter
        if not isinstance(population_filter, chi.PopulationFilter):
            raise TypeError(
                'The population filter has to be an instance of '
                'chi.PopulationFilter.')
        self._filter = copy.deepcopy(population_filter)
        self._n_times = population_filter.n_times()
        self._n_observables = population_filter.n_observables()

        # Check times
        if len(times) != len(np.unique(times)):
            raise ValueError(
                'The measurement times in times have to be unique.')

        if len(times) != self._n_times:
            raise ValueError(
                'The length of times does not match the time dimension of '
                'observations.')
        self._filter.sort_times(np.argsort(times))
        self._times = np.sort(times)

        # Check mechanistic model
        if not isinstance(mechanistic_model, chi.MechanisticModel):
            raise TypeError(
                'The mechanistic model has to be an instance of '
                'chi.MechanisticModel.')
        if mechanistic_model.n_outputs() != self._n_observables:
            raise ValueError(
                'The number of mechanistic model outputs does not match the '
                'number of observables.')
        self._mechanistic_model = mechanistic_model.copy()

        # Check population model
        if not isinstance(population_model, chi.PopulationModel):
            raise TypeError(
                'The population model has to be an instance of '
                'chi.PopulationModel.')
        if population_model.n_dim() != self._mechanistic_model.n_parameters():
            raise ValueError(
                'The number of population model dimensions does not match the '
                'number of mechanistic model parameters.')
        # TODO: Currently, no support for covariate population models
        if population_model.transforms_individual_parameters():
            raise ValueError(
                'Population models that transform the individual parameters '
                'are currently not supported. This feature will be added in a '
                'future release.')
        self._population_model = population_model
        self._n_dim = self._population_model.n_dim()
        self._n_top = self._population_model.n_parameters()
        self._population_model.set_dim_names(mechanistic_model.parameters())

        # Check error model
        if sigma is not None:
            # Make sure sigma is a list (integers and floats wrapped)
            try:
                sigma = list(sigma)
            except TypeError:
                sigma = [sigma]
            # Make sure that one sigma for each model output exists
            if len(sigma) != self._n_observables:
                raise ValueError(
                    'One sigma for each observable has to provided.')
            # Make sure sigmas assume valid values
            sigma = np.array([float(s) for s in sigma])
            if np.any(sigma < 0):
                raise ValueError(
                    'The elements of sigma have to be greater or equal '
                    'to zero.')

            # Reshape for later convenience
            sigma = sigma.reshape(1, self._n_observables, 1)
        self._sigma = sigma

        # Get parameter names and update n_top if sigma has not been fixed
        names = self._population_model.get_parameter_names()
        if self._sigma is None:
            names += [
                'Sigma %s' % name for name in mechanistic_model.outputs()]
            self._n_top += self._n_observables

        if not isinstance(log_prior, pints.LogPrior):
            raise TypeError(
                'The log-prior has to be an instance of pints.LogPrior.')
        if log_prior.n_parameters() != self._n_top:
            raise ValueError(
                'The dimensionality of the log-prior does not match the '
                'number of population parameters. The population parameters '
                'are <' + str(names) + '>.')
        self._log_prior = log_prior

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(
                'The number of samples of the population filter has to be '
                'greater than zero.')
        self._n_samples = n_samples

        self._error_on_log_scale = bool(error_on_log_scale)
        self._top_names = names
        self._n_parameters = \
            self._n_top + self._n_samples * (self._n_dim + self._n_times)

    def __call__(self, parameters):
        """
        Returns the log-likelihood of the model parameters with respect
        to the filtered data.

        The order of the input parameters is expected to be
            1. Population parameters in order of the population models
            2. The sampled bottom-level parameters for the simulated
                individuals in order of the population models.
            3. The sampled residual error of the mechanistic model.

        The order of the parameters can be more explicitly checked with
        :meth:`get_parameter_names`.

        :param parameters: Parameters of the inference model.
        :type parameters: np.ndarray of shape (n_parameters,)
        """
        # Parse parameters into top parameters, bottom parameters and noise
        # realisations.
        # (top parameters inlcudes sigma, if sigma is not fixed)
        parameters = np.asarray(parameters)
        n_pop = self._population_model.n_parameters()
        pop_parameters = parameters[:n_pop]
        sigma = self._sigma
        if self._sigma is None:
            # Sigma is not fixed
            sigma = parameters[n_pop:self._n_top].reshape(
                1, self._n_observables, 1)
        end_bottom = self._n_top+self._n_samples*self._n_dim
        bottom_parameters = parameters[self._n_top:end_bottom].reshape(
            self._n_samples, self._n_dim)
        epsilon = parameters[end_bottom:].reshape(
            self._n_samples, self._n_observables, self._n_times)

        # Compute log-prior contribution to score
        score = self._log_prior(parameters[:self._n_top])
        if np.isinf(score):
            return score

        # Add population contribution to the score
        # (Slicing makes sure that if sigma is not set, it's filtered now)
        score += self._population_model.compute_log_likelihood(
            parameters=pop_parameters,
            observations=bottom_parameters)
        if np.isinf(score):
            return score

        # Add noise contribution
        score += \
            - self._n_samples * self._n_observables * np.log(2 * np.pi) / 2 \
            - np.sum(np.log(sigma) + epsilon**2 / sigma**2 / 2)

        # Check that mechanistic model has sensitivities disabled
        # (Simply for performance)
        if self._mechanistic_model.has_sensitivities():
            self._mechanistic_model.enable_sensitivities(False)

        # Solve mechanistic model for bottom parameters
        y = np.empty(
            shape=(self._n_samples, self._n_observables, self._n_times))
        for ids, individual_parameters in enumerate(bottom_parameters):
            try:
                y[ids] = self._mechanistic_model.simulate(
                    parameters=individual_parameters, times=self._times)
            except (myokit.SimulationError, Exception
                    ) as e:  # pragma: no cover
                warnings.warn(
                    'An error occured while solving the mechanistic model: \n'
                    + str(e) + '.\n A score of -infinity is returned.',
                    RuntimeWarning)
                return -np.infty

        # Add noise to simulate measurements
        if self._error_on_log_scale:
            y *= np.exp(sigma * epsilon)
        else:
            y += sigma * epsilon

        # Use population filter to compute log-likelihood of bottom-level
        # parameters
        score += self._filter.compute_log_likelihood(y)

        return score

    def evaluateS1(self, parameters):
        """
        Returns the log-posterior score and its sensitivities to the model
        parameters.

        :param parameters: An array-like object with parameter values.
        :type parameters: List[float], numpy.ndarray of length ``n_parameters``
        """
        # Parse parameters into top parameters, bottom parameters and noise
        # realisations.
        # (top parameters inlcude sigma, if sigma is not fixed)
        parameters = np.asarray(parameters)
        n_pop = self._population_model.n_parameters()
        pop_parameters = parameters[:n_pop]
        sigma = self._sigma
        if self._sigma is None:
            # Sigma is not fixed
            sigma = parameters[n_pop:self._n_top].reshape(
                1, self._n_observables, 1)
        end_bottom = self._n_top+self._n_samples*self._n_dim
        bottom_parameters = parameters[self._n_top:end_bottom].reshape(
            self._n_samples, self._n_dim)
        epsilon = parameters[end_bottom:].reshape(
            self._n_samples, self._n_observables, self._n_times)

        # Compute log-prior contribution to score
        sensitivities = np.empty(shape=self._n_parameters)
        score, sensitivities[:self._n_top] = self._log_prior.evaluateS1(
            parameters[:self._n_top])
        if np.isinf(score):
            return score, sensitivities

        # Add population contribution to the score
        # (Slicing makes sure that if sigma is not set, it's filtered now)
        s, sens = self._population_model.compute_sensitivities(
            parameters=pop_parameters,
            observations=bottom_parameters)
        score += s
        sensitivities[:n_pop] += sens[-n_pop:]
        sensitivities[self._n_top:end_bottom] = sens[:-n_pop]
        if np.isinf(score):
            return score, sensitivities

        # Add noise contribution
        score += \
            - self._n_samples * self._n_observables * np.log(2 * np.pi) / 2 \
            - np.sum(np.log(sigma) + epsilon**2 / sigma**2 / 2)
        sensitivities[end_bottom:] = - (epsilon / sigma**2).flatten()

        # Check that mechanistic model has sensitivities enabled
        if not self._mechanistic_model.has_sensitivities():
            self._mechanistic_model.enable_sensitivities(True)

        # Solve mechanistic model for bottom parameters
        n_parameters = self._mechanistic_model.n_parameters()
        y = np.empty(
            shape=(self._n_samples, self._n_observables, self._n_times))
        dybar_dpsi = np.empty(shape=(
            self._n_samples, self._n_times, self._n_observables, n_parameters))
        for ids, individual_parameters in enumerate(bottom_parameters):
            try:
                y[ids], dybar_dpsi[ids] = self._mechanistic_model.simulate(
                    parameters=individual_parameters, times=self._times)
            except (myokit.SimulationError, Exception
                    ) as e:  # pragma: no cover
                warnings.warn(
                    'An error occured while solving the mechanistic model: \n'
                    + str(e) + '.\n A score of -infinity is returned.',
                    RuntimeWarning)
                return -np.infty, sensitivities

        # Add noise to simulate measurements
        if self._error_on_log_scale:
            y *= np.exp(sigma * epsilon)
        else:
            y += sigma * epsilon

        # Use population filter to compute log-likelihood of bottom-level
        # parameters
        s, ds_y = self._filter.compute_sensitivities(simulated_obs=y)

        # Add filter contributions
        # (ds_dy * dy_dybar * dybar_dpsi, ds_dy * dy_dybar * dy_depsilon)
        score += s
        if self._error_on_log_scale:
            sensitivities[self._n_top:end_bottom] += np.sum(
                (ds_y * np.exp(sigma * epsilon))[..., np.newaxis]
                * np.swapaxes(dybar_dpsi, 1, 2), axis=(1, 2)).flatten()
            sensitivities[end_bottom:] += np.sum(
                ds_y * y * sigma, axis=1).flatten()
        else:
            sensitivities[self._n_top:end_bottom] += np.sum(
                ds_y[..., np.newaxis]
                * np.swapaxes(dybar_dpsi, 1, 2), axis=(1, 2)).flatten()
            sensitivities[end_bottom:] += np.sum(
                ds_y * sigma, axis=1).flatten()

        # Add sigma sensitivities, if sigma is not fixed
        # ds_dsigma = derror_model_dsigma + ds_dy * dy_dsigma
        if self._sigma is None:
            # Error model contribution
            sensitivities[n_pop:self._n_top] += np.sum(
                -1 / sigma + epsilon**2 / sigma**3, axis=(0, 2))
            # Pop. filter contribution
            if self._error_on_log_scale:
                sensitivities[n_pop:self._n_top] += np.sum(
                    ds_y * epsilon * y, axis=(0, 2))
            else:
                sensitivities[n_pop:self._n_top] += np.sum(
                    ds_y * epsilon, axis=(0, 2))

        return score, sensitivities

    def get_log_likelihood(self):
        """
        Returns the log-likelihood.

        For the population filter log-posterior the population filter is
        returned.
        """
        return copy.deepcopy(self._filter)

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
        if unique:
            return [None] + [
                'Sim. %d' % (_id + 1) for _id in range(self._n_samples)]

        ids = [None] * self._n_top
        for _id in range(self._n_samples):
            ids += ['Sim. %d' % (_id + 1)] * self._n_dim
        for _id in range(self._n_samples):
            ids += ['Sim. %d' % (_id + 1)] * self._n_times

        return ids

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
        names = copy.copy(self._top_names)
        if not exclude_bottom_level:
            bottom_names = self._mechanistic_model.parameters()
            names += bottom_names * self._n_samples
            epsilon_names = [
                'Epsilon time %d' % (idt + 1) for idt in range(self._n_times)]
            names += epsilon_names * self._n_samples

        if include_ids:
            ids = self.get_id()
            for idn, name in enumerate(names):
                if ids[idn] is not None:
                    names[idn] = ids[idn] + ' ' + name

        return names

    def n_parameters(self, exclude_bottom_level=False):
        """
        Returns the number of parameters.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter are counted in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        """
        n_parameters = self._n_top
        if exclude_bottom_level:
            return n_parameters

        n_bottom = self._mechanistic_model.n_parameters()

        return n_parameters + self._n_samples * (n_bottom + self._n_times)


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
