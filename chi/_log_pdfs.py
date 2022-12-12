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
    log-likelihoods whose parameters are governed by a population model.

    A hierarchical log-likelihood is defined by a list of
    :class:`LogLikelihood` instances and a :class:`PopulationModel`

    .. math::
        \log p(\mathcal{D}, \Psi | \theta ) =
            \sum _{ij} \log p(y_{ij} | \psi _{i} , t_{ij})
            + \sum _{i} \log p(\psi _{i}| \theta),

    where the first term is the sum over the log-likelihoods and the second
    term is the log-likelihood of the population model parameters.
    :math:`\mathcal{D}=\{ (y_{ij}, t_{ij})\}` is the
    data, where :math:`(y_{ij}, t_{ij})` is the :math:`j^{\mathrm{th}}`
    measurement of log-likelihood :math:`i`. :math:`\Psi = \{ \psi_i\}`
    denotes the parameters across the individual log-likelihoods.

    :param log_likelihoods: A list of log-likelihoods which are
        defined on the same parameter space with dimension ``n_parameters``.
    :type log_likelihoods: list[LogLikelihood] of length ``n_ids``
    :param population_models: A population model of dimension ``n_parameters``.
    :type population_models: PopulationModel
    :param covariates: A 2-dimensional array of with the
        individual's covariates.
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
        if population_model.n_covariates() > 0:
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
        self._n_parameters = np.sum(
            self._population_model.n_hierarchical_parameters(self._n_ids))
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
        bottom_parameters = \
            self._population_model.compute_individual_parameters(
                parameters=top_parameters,
                eta=bottom_parameters,
                covariates=self._covariates,
                return_eta=True
            )

        # Compute population model score
        score = self._population_model.compute_log_likelihood(
            top_parameters, bottom_parameters, covariates=self._covariates)

        # Return if values already lead to a rejection
        if np.isinf(score):
            return score

        # Transform bottom-level parameters
        # Identity, if model does not tranform parameters
        bottom_parameters = \
            self._population_model.compute_individual_parameters(
                top_parameters, bottom_parameters, self._covariates)

        # Evaluate individual likelihoods
        for idi, log_likelihood in enumerate(self._log_likelihoods):
            score += log_likelihood(bottom_parameters[idi])
        return score

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
        bottom_parameters = \
            self._population_model.compute_individual_parameters(
                parameters=top_parameters,
                eta=bottom_parameters,
                covariates=self._covariates,
                return_eta=True
            )

        # Make sure bottom parameters are psi
        # (parameters of the individual log-likelihoods)
        psi = self._population_model.compute_individual_parameters(
            parameters=top_parameters,
            eta=bottom_parameters,
            covariates=self._covariates
        )

        # Evaluate individual likelihoods
        score = 0
        dlogp_dpsi = np.empty(shape=(self._n_ids, self._n_dim))
        for idi, log_likelihood in enumerate(self._log_likelihoods):
            l, dl_dpsi = log_likelihood.evaluateS1(psi[idi])
            score += l
            dlogp_dpsi[idi] = dl_dpsi

        # Compute population model score
        s, dscore = self._population_model.compute_sensitivities(
            top_parameters, bottom_parameters, covariates=self._covariates,
            dlogp_dpsi=dlogp_dpsi, reduce=True)
        score += s

        return score, dscore

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
        special_dims, _, _ = self._population_model.get_special_dims()
        for info in special_dims:
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
        \log p(\Psi , \theta | \mathcal{D}) =
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

    def get_population_model(self):
        """
        Returns the population model.
        """
        return self._log_likelihood.get_population_model()

    def n_ids(self):
        """
        Returns the number of modelled individuals.
        """
        return self._log_likelihood.n_log_likelihoods()

    def n_parameters(self, exclude_bottom_level=False):
        """
        Returns the number of parameters.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter are counted in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        """
        return self._log_likelihood.n_parameters(exclude_bottom_level)

    def sample_initial_parameters(self, n_samples=1, seed=None):
        """
        Samples top-level parameters from the log-prior and bottom-level
        parameters from the population model using the top-level samples.

        These parameter samples may be used to initialise inference algorithms.

        :param n_samples: Number of samples.
        :type n_samples: int, optional
        :param seed: Seed for random number generator.
        :type seed: int, optional
        :rtype: np.ndarray of shape ``(n_samples, n_parameters)``
        """
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(
                'The number of samples has to be greater or equal to 1.')

        initial_params = np.empty(shape=(n_samples, self._n_parameters))

        # Sample initial values for top-level parameters
        np.random.seed(seed)
        n_top = self._log_prior.n_parameters()
        n_bottom = self._n_parameters - n_top
        initial_params[:, n_bottom:] = self._log_prior.sample(n_samples)

        # Sample bottom-level parameters
        if n_bottom == 0:
            return initial_params

        # Transform seed to random number generator, so seed is propagated
        # across models
        if seed is not None:
            seed += 1
        rng = np.random.default_rng(seed=seed)

        n_ids = self._log_likelihood.n_log_likelihoods()
        covariates = self._log_likelihood._covariates
        bottom_parameters = []
        population_model = self._log_likelihood.get_population_model()
        for sample_id in range(n_samples):
            bottom_parameters.append(population_model.sample(
                parameters=initial_params[sample_id, n_bottom:],
                n_samples=n_ids, seed=rng, covariates=covariates))

        # Remove pooled dimensions
        # (Pooled and heterogen. dimensions do not count as bottom parameters)
        dims = []
        current_dim = 0
        if isinstance(population_model, chi.ReducedPopulationModel):
            population_model = population_model.get_population_model()
        try:
            pop_models = population_model.get_population_models()
        except AttributeError:
            pop_models = [population_model]
        for pop_model in pop_models:
            n_dim = pop_model.n_dim()
            if isinstance(
                    pop_model, (chi.PooledModel, chi.HeterogeneousModel)):
                current_dim += n_dim
                continue
            end_dim = current_dim + n_dim
            dims += list(range(current_dim, end_dim))
            current_dim = end_dim
        for idx, bottom_params in enumerate(bottom_parameters):
            bottom_parameters[idx] = bottom_params[:, dims].flatten()

        initial_params[:, :n_bottom] = np.vstack(bottom_parameters)

        return initial_params


class LogLikelihood(pints.LogPDF):
    r"""
    A log-likelihood that quantifies the likelihood of parameter values to
    capture the measurements within the model approximation of the
    data-generating process.

    A log-likelihood is defined by an instance of a :class:`MechanisticModel`,
    one :class:`ErrorModel` for each mechanistic model output and measurements
    defined by ``observations`` and ``times``

    .. math::
        p(\psi | \mathcal{D}) = \sum _{j=1}
        \log p(y_j | \psi, t_j),

    where :math:`\mathcal{D} = \{(y_j , t_j)\}` denotes the measurements.

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
        \log p(\psi | \mathcal{D}) =
            \log p(\mathcal{D} | \psi) + \log p(\psi ) + \mathrm{constant},

    where :math:`\psi` are the parameters of the log-likelihood and
    :math:`\mathcal{D}` are the observed data. The additive constant
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

    def get_parameter_names(self, *args, **kwargs):
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

    def sample_initial_parameters(self, n_samples=1, seed=None):
        """
        Samples parameters from the log-prior which may be used to initialise
        inference algorithms.

        :param n_samples: Number of samples.
        :type n_samples: int, optional
        :param seed: Seed for random number generator.
        :type seed: int, optional
        :rtype: np.ndarray of shape ``(n_samples, n_parameters)``
        """
        np.random.seed(seed)
        return self._log_prior.sample(n_samples)


class PopulationFilterLogPosterior(HierarchicalLogPosterior):
    r"""
    A population filter log-posterior approximates a hierarchical
    log-posterior.

    Population filter log-posteriors can be used to approximate hierarchical
    log-posteriors when exact hierarchical inference becomes numerically
    intractable. The canonical application of population filter inference is
    the inference from time series snapshot data.

    The population filter log-posterior is defined by a population filter,
    a mechanistic model, an error model, a population model and the data

    .. math::
        \log p(\theta , \tilde{Y}, \Psi| \mathcal{D}) =&
            \sum _{ij} \log p (y_{ij} | \tilde{Y}_j) +
            \sum _{sj} \log p (\tilde{y}_{sj} | \psi_s, t_j) +
            \sum _{s} \log p (\psi_{s} | \theta)& \\
            &+ \log p (\theta) + \mathrm{constant},&

    where the data :math:`\mathcal{D} = \{ (Y_j , t_j)\}` are measurements
    over time with :math:`Y_j = \{ y_{ij} \}` denoting the measurements at time
    point :math:`t_j` across individuals.
    Here, we use :math:`i` to index individuals from the dataset.
    The first term of the log-posterior is the population filter
    contribution which estimates the log-likelihood that the virtual
    measurements, :math:`\tilde{Y}_j = \{ \tilde{y}_{sj}\}`, come from the same
    distribution as the measurements, :math:`Y_j`.
    The quality of the log-likelihood estimate is subject to the
    appropriateness of the population filter [ref]. We use :math:`s` to index
    virtual individuals.
    The second term of the log-posterior is the
    log-likelihood of the simulated parameters
    :math:`\Psi = \{ \psi _s\}`
    with respect to the virtual measurements. Each simulated parameter
    corresponds to a virtual individual.
    The log-likelihood of a set of simulated parameters is defined
    by the mechanistic model and the error model, as well as the simulated
    measurements for that individual.
    The third term is the log-likelihood that the population parameters
    :math:`\theta = \{ \theta _k \}` govern the distribution of the
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
    :type sigma: List[float] of length ``n_observables``, optional
    :param error_on_log_scale: A boolean flag indicating whether the error
        model models the residuals of the mechanistic model directly or on
        a log scale.
    :type error_on_log_scale: bool, optional
    :param n_samples: Number of simulated individuals per evaluation.
    :type n_samples: int, optional.
    :param covariates: Covariates of the simulated individuals.
    :type covariates: np.ndarray of shape ``(n_cov,)`` or
        ``(n_samples, n_cov)``, optional
    """
    def __init__(
            self, population_filter, times, mechanistic_model,
            population_model, log_prior, sigma=None, error_on_log_scale=False,
            n_samples=100, covariates=None):

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
        self._population_model = copy.deepcopy(population_model)

        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError(
                'The number of samples of the population filter has to be '
                'greater than zero.')
        self._n_samples = n_samples

        self._covariates = None
        if self._population_model.n_covariates() > 0:
            covariates = np.array(covariates)
            if covariates.ndim == 1:
                covariates = covariates[np.newaxis, :]
            _, n_c = covariates.shape
            if n_c != self._population_model.n_covariates():
                raise ValueError(
                    'Invalid covariates. The provided covariates do not match '
                    'the number of covariates.')
            try:
                covariates = np.broadcast_to(covariates, (n_samples, n_c))
            except ValueError:
                raise ValueError(
                    'Invalid covariates. The provided covariates cannot be '
                    'broadcasted to the shape (n_samples, n_cov).')
            self._covariates = covariates

        self._population_model.set_n_ids(self._n_samples)
        self._n_hdim = self._population_model.n_hierarchical_dim()
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

        self._error_on_log_scale = bool(error_on_log_scale)
        self._top_names = names
        self._n_parameters = \
            self._n_top + self._n_samples \
            * (self._n_hdim + self._n_times * self._n_observables)
        self._end_bottom = self._n_top + self._n_samples * self._n_hdim

        # Get dimensions that need to be treated differently during inference
        self._special_dims, self._n_pooled_dim, self._n_heterogen_dim = \
            self._get_special_dims()

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
        bottom_parameters = parameters[self._n_top:self._end_bottom].reshape(
            self._n_samples, self._n_hdim)
        epsilon = parameters[self._end_bottom:].reshape(
            self._n_samples, self._n_observables, self._n_times)

        # Compute log-prior contribution to score
        score = self._log_prior(parameters[:self._n_top])
        if np.isinf(score):
            return score

        # Add population contribution to the score
        bottom_parameters = self._reshape_bottom_parameters(
            bottom_parameters, pop_parameters)
        score += self._population_model.compute_log_likelihood(
            parameters=pop_parameters,
            observations=bottom_parameters,
            covariates=self._covariates)
        if np.isinf(score):
            return score

        # Add noise contribution
        score += \
            - self._n_samples * self._n_observables * np.log(2 * np.pi) / 2 \
            - np.sum(epsilon**2) / 2

        # Check that mechanistic model has sensitivities disabled
        # (Simply for performance)
        if self._mechanistic_model.has_sensitivities():
            self._mechanistic_model.enable_sensitivities(False)

        # Transform bottom parameters into mechanistic model space
        bottom_parameters = \
            self._population_model.compute_individual_parameters(
                parameters=pop_parameters,
                eta=bottom_parameters,
                covariates=self._covariates)

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

    def _get_special_dims(self):
        """
        Counts the number of pooled and heterogeneous dimensions.
        """
        # Get elementary population models
        pop_models = [self._population_model]
        if isinstance(self._population_model, chi.ComposedPopulationModel):
            pop_models = self._population_model.get_population_models()

        n_pooled_dims = 0
        n_hetero_dims = 0
        special_dims = []
        current_dim = 0
        current_top_index = 0
        for pop_model in pop_models:
            # Check whether dimension is pooled
            _, n_top = pop_model.n_hierarchical_parameters(self._n_samples)
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

            current_dim += n_dim
            current_top_index += n_top

        return special_dims, n_pooled_dims, n_hetero_dims

    def _remove_duplicates(self, sensitivities, dbottom):
        """
        In some sense the reverse of self._reshape_bottom_parameters.

        1. Pooled bottom parameters need to be added to population parameter
        2. Heterogeneous bottom parameters need to added to population
            parameters

        sensitivities: shape (n_parameters,)
        dbottom: shape (n_ids, n_bottom, n_dim)
        """
        # Check for quick solution 1: no pooled parameters and no heterogeneous
        # parameters
        n_dim = self._population_model.n_dim()
        if self._population_model.n_hierarchical_dim() == n_dim:
            sensitivities[self._n_top:self._end_bottom] = dbottom.flatten()
            return sensitivities

        # Check for quick solution 2: all parameters heterogen.
        if self._n_heterogen_dim == n_dim:
            # Add contributions from bottom-level parameters to top-level
            # parameters
            sensitivities[:self._n_top] += dbottom.flatten()
            return sensitivities

        # Check for quick solution 3: all parameters pooled
        if self._n_pooled_dim == n_dim:
            # Add contributions from bottom-level parameters
            # (Population sens. are actually zero, so we can just replace them)
            sensitivities[:self._n_top] += np.sum(dbottom, axis=0)
            return sensitivities

        shift = 0
        current_dim = 0
        bottom_sens = np.empty(shape=(self._n_samples, self._n_hdim))
        for info in self._special_dims:
            start_dim, end_dim, start_top, end_top, is_pooled = info
            # Fill leading regular dims
            bottom_sens[:, current_dim-shift:start_dim-shift] = \
                dbottom[:, current_dim:start_dim]
            # Fill special dims
            if is_pooled:
                sensitivities[start_top:end_top] += \
                    np.sum(dbottom[:, start_dim:end_dim], axis=0)
            else:
                sensitivities[start_top:end_top] += \
                    dbottom[:, start_dim:end_dim].flatten()
            current_dim = end_dim
            shift += end_dim - start_dim
        # Fill trailing regular dims
        bottom_sens[:, current_dim-shift:] = dbottom[:, current_dim:]

        # Add bottom sensitivties
        sensitivities[self._n_top:self._end_bottom] = bottom_sens.flatten()

        return sensitivities

    def _reshape_bottom_parameters(self, bottom_parameters, top_parameters):
        """
        Takes bottom parameters and top parameters with no duplicates and
        returns bottom parameters of shape (n_ids, n_dim), where pooled
        and heterogenous parameters are duplicated.

        Bottom parameters have currently shape (n_ids, n_hierarchical_dim) and
        they will be returned in shape (n_ids, n_dim).
        """
        # Check for quick solution 1: no pooled parameters and no heterogen.
        n_dim = self._population_model.n_dim()
        if self._population_model.n_hierarchical_dim() == n_dim:
            return bottom_parameters

        # Check for quick solution 2: all parameters pooled
        bottom_params = np.empty(shape=(self._n_samples, n_dim))
        if self._n_pooled_dim == n_dim:
            bottom_params[:, :] = top_parameters[np.newaxis, :]
            return bottom_params

        # Check for quick solution 3: all parameters heterogen.
        if self._n_heterogen_dim == n_dim:
            bottom_params[:, :] = top_parameters.reshape(
                self._n_samples, n_dim)
            return bottom_params

        shift = 0
        current_dim = 0
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
                    start_top:end_top].reshape(self._n_samples, dims)
            current_dim = end_dim
            shift += dims
        # Fill trailing non-pooled dims
        bottom_params[:, current_dim:] = bottom_parameters[
            :, current_dim-shift:]

        return bottom_params

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
        bottom_parameters = parameters[self._n_top:self._end_bottom].reshape(
            self._n_samples, self._n_hdim)
        epsilon = parameters[self._end_bottom:].reshape(
            self._n_samples, self._n_observables, self._n_times)

        # Initialise sensitivities
        sensitivities = np.empty(shape=self._n_parameters)

        # Compute log-prior contribution to score
        score, sensitivities[:self._n_top] = self._log_prior.evaluateS1(
            parameters[:self._n_top])
        if np.isinf(score):
            return score, sensitivities

        # Add noise contribution
        score += \
            - self._n_samples * self._n_observables * np.log(2 * np.pi) / 2 \
            - np.sum(epsilon**2) / 2
        sensitivities[self._end_bottom:] = -epsilon.flatten()

        # Check that mechanistic model has sensitivities enabled
        if not self._mechanistic_model.has_sensitivities():
            self._mechanistic_model.enable_sensitivities(True)

        # Transform bottom-parameters
        bottom_parameters = self._reshape_bottom_parameters(
            bottom_parameters, pop_parameters)
        psi = self._population_model.compute_individual_parameters(
            parameters=pop_parameters,
            eta=bottom_parameters,
            covariates=self._covariates)

        # Solve mechanistic model for bottom parameters
        n_parameters = self._mechanistic_model.n_parameters()
        y = np.empty(
            shape=(self._n_samples, self._n_observables, self._n_times))
        dybar_dpsi = np.empty(shape=(
            self._n_samples, self._n_times, self._n_observables, n_parameters))
        for ids, individual_parameters in enumerate(psi):
            try:
                y[ids], dybar_dpsi[ids] = self._mechanistic_model.simulate(
                    parameters=individual_parameters, times=self._times)
            except (myokit.SimulationError, Exception
                    ) as e:  # pragma: no cover
                warnings.warn(
                    'An error occured while solving the mechanistic model: \n'
                    + str(e) + '.\n A score of -infinity is returned.',
                    RuntimeWarning)
                return -np.infty, sensitivities[:self._n_parameters]

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
            ds_dpsi = np.sum(
                (ds_y * np.exp(sigma * epsilon))[..., np.newaxis]
                * np.swapaxes(dybar_dpsi, 1, 2), axis=(1, 2))
            sensitivities[self._end_bottom:] += (ds_y * y * sigma).flatten()
        else:
            ds_dpsi = np.sum(
                ds_y[..., np.newaxis]
                * np.swapaxes(dybar_dpsi, 1, 2), axis=(1, 2))
            sensitivities[self._end_bottom:] += (ds_y * sigma).flatten()

        # Add sigma sensitivities, if sigma is not fixed
        # ds_dsigma = derror_model_dsigma + ds_dy * dy_dsigma
        if self._sigma is None:
            if self._error_on_log_scale:
                sensitivities[n_pop:self._n_top] += np.sum(
                    ds_y * epsilon * y, axis=(0, 2))
            else:
                sensitivities[n_pop:self._n_top] += np.sum(
                    ds_y * epsilon, axis=(0, 2))

        # Add population contribution to the score
        s, dbottom, dtheta = self._population_model.compute_sensitivities(
            parameters=pop_parameters,
            observations=bottom_parameters,
            covariates=self._covariates,
            dlogp_dpsi=ds_dpsi)
        score += s
        sensitivities[:n_pop] += dtheta
        if np.isinf(score):
            return score, sensitivities[:self._n_parameters]

        # Collect sensitivities
        sensitivities = self._remove_duplicates(sensitivities, dbottom)

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
            return [
                'Sim. %d' % (_id + 1) for _id in range(self._n_samples)]

        ids = [None] * self._n_top
        for _id in range(self._n_samples):
            ids += ['Sim. %d' % (_id + 1)] * self._n_hdim
        for _id in range(self._n_samples):
            ids += [
                'Sim. %d' % (_id + 1)] * self._n_observables * self._n_times

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
        # Get top parameter names
        names = copy.copy(self._top_names)

        # Get bottom parameter names
        # (pooled and heterogenous parameters count as top parameters)
        current_dim = 0
        bottom_names = self._mechanistic_model.parameters()
        n = []
        for info in self._special_dims:
            start_dim, end_dim, _, _, _ = info
            n += bottom_names[current_dim:start_dim]
            current_dim = end_dim
        n += bottom_names[current_dim:]
        bottom_names = n

        # Make copies of bottom parameters and append to top parameters
        names += bottom_names * self._n_samples

        # Append epsilon parameter names
        epsilon_names = []
        for output in self._mechanistic_model.outputs():
            name = output + ' Epsilon time '
            epsilon_names += [
                name + '%d' % (idt + 1) for idt in range(self._n_times)]
        names += epsilon_names * self._n_samples

        if include_ids:
            ids = self.get_id()
            n = []
            for idn, name in enumerate(names):
                if ids[idn]:
                    name = ids[idn] + ' ' + name
                n.append(name)
            names = n

        if exclude_bottom_level:
            names = names[:self._n_top]

        return names

    def get_population_model(self):
        """
        Returns the population model.
        """
        return self._population_model

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

        return self._n_parameters

    def n_samples(self):
        """
        Returns the number of simulated individuals per posterior evaluation.
        """
        return self._n_samples

    def sample_initial_parameters(self, n_samples=1, seed=None):
        """
        Samples top-level parameters from the log-prior and bottom-level
        parameters from the population model using the top-level samples.
        The noise realisations are sampled from a standard Gaussian
        distribution.

        These parameter samples may be used to initialise inference algorithms.

        :param n_samples: Number of samples.
        :type n_samples: int, optional
        :param seed: Seed for random number generator.
        :type seed: int, optional
        :rtype: np.ndarray of shape ``(n_samples, n_parameters)``
        """
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError('The number of samples has to be at least 1.')

        initial_params = np.empty(shape=(n_samples, self._n_parameters))

        # Sample initial values for top-level parameters
        np.random.seed(seed)
        initial_params[:, :self._n_top] = self._log_prior.sample(n_samples)

        # Transform seed to random number generator, so seed is propagated
        # across models
        if seed is not None:
            seed += 1
        rng = np.random.default_rng(seed=seed)

        # Sample bottom-level parameters
        n_pop = self._population_model.n_parameters()
        bottom_parameters = []
        for sample_id in range(n_samples):
            bottom_parameters.append(self._population_model.sample(
                parameters=initial_params[sample_id, :n_pop],
                covariates=self._covariates,
                n_samples=self._n_samples, seed=rng))

        # Remove pooled dimensions
        # (Pooled and heterogen. dimensions do not count as bottom parameters)
        dims = []
        current_dim = 0
        try:
            pop_models = self._population_model.get_population_models()
        except AttributeError:
            pop_models = [self._population_model]
        for pop_model in pop_models:
            n_dim = pop_model.n_dim()
            if pop_model.n_hierarchical_dim() == 0:
                current_dim += n_dim
                continue
            end_dim = current_dim + n_dim
            dims += list(range(current_dim, end_dim))
            current_dim = end_dim
        for idx, bottom_params in enumerate(bottom_parameters):
            bottom_parameters[idx] = bottom_params[:, dims].flatten()

        initial_params[:, self._n_top:self._end_bottom] = \
            np.vstack(bottom_parameters)

        # Sample epsilons
        initial_params[:, self._end_bottom:] = rng.normal(
            loc=0, scale=1,
            size=(
                n_samples,
                self._n_samples * self._n_times * self._n_observables
            )
        )

        return initial_params
