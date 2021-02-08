#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy

import numpy as np
import pints

import erlotinib as erlo


class HierarchicalLogLikelihood(object):
    r"""
    A hierarchical log-likelihood consists of structurally identical
    log-likelihoods whose parameters are coupled by population models.

    A hierarchical log-likelihood takes a list of :class:`LogLikelihood`
    instances, and a list of :class:`PopulationModel` instances. Each
    :class:`LogLikelihood` in the list has to have the same number of
    parameters. For each parameter of the :class:`LogLikelihood`, a
    :class:`PopulationModel` has to be provided which models the
    distribution of the respective parameter across individuals in the
    population.

    Formally the hierarchical log-likelihood is constructed by the product
    of the individual likelihoods :math:`p(x_i | \psi _i)`
    and the population distribution :math:`p(\Psi | \theta )`

    .. math::
        L(\Psi , \theta | X^{\text{obs}}) =
        \sum _{i=1}^n \log p(x^{\text{obs}}_i | \psi _i )
        + \log p(\Psi | \theta ).

    Here, :math:`\Psi = (\psi _1, \ldots , \psi _n )` are the
    bottom-level parameters of the individual likelihoods,
    :math:`\theta` are the top-level parameters of the population models,
    and
    :math:`X^{\text{obs}} = (x^{\text{obs}}_1, \ldots , x^{\text{obs}}_n)`
    are the observations for each individual.

    .. note::
        The number of parameters of an hierarchical log-likelihood is
        larger than the number of parameters of the corresponding
        :class:`PredictivePopulationModel`,
        as the integral over the individual parameters
        :math:`\Psi` can in general not be solved analytically.

    :param log_likelihoods: A list of log-likelihoods defined on the same
        parameter space.
    :type log_likelihoods: list[LogLikelihood]
    :param population_models: A list of population models with one
        population model for each parameter of the log-likelihoods.
    :type population_models: list[PopulationModel]

    Example
    -------

    ::

        import erlotinib as erlo

        # Define log-likelihoods
        log_likelihood_1 = erlo.LogLikelihood(
            mechanistic_model,
            error_models,
            observations_1,
            times_1)
        log_likelihood_2 = erlo.LogLikelihood(
            mechanistic_model,
            error_models,
            observations_2,
            times_2)

        # Define population models
        # (Assumes likelihoods have 3 parameters each)
        population_models = [
            erlo.LogNormalModel(),
            erlo.PooledModel(),
            erlo.HeterogeneousModel()]

        # Create hierarchical log-likelihood
        hierarch_log_likelihood = erlo.HierarchicalLogLikelihood(
            log_likelihoods=[log_likelihood_1, log_likelihood_2]
            population_models=population_models)
    """
    def __init__(self, log_likelihoods, population_models):
        super(HierarchicalLogLikelihood, self).__init__()

        for log_likelihood in log_likelihoods:
            if not isinstance(log_likelihood, LogLikelihood):
                raise ValueError(
                    'The log-likelihoods have to be instances of a '
                    'erlotinib.LogLikelihood.')

        n_parameters = log_likelihoods[0].n_parameters()
        for log_likelihood in log_likelihoods:
            if log_likelihood.n_parameters() != n_parameters:
                raise ValueError(
                    'The number of parameters of the log-likelihoods differ. '
                    'All log-likelihoods have to be defined on the same '
                    'parameter space.')

        names = log_likelihoods[0].get_parameter_names()
        for log_likelihood in log_likelihoods:
            if not np.array_equal(log_likelihood.get_parameter_names(), names):
                raise ValueError(
                    'The parameter names of the log-likelihoods differ.'
                    'All log-likelihoods have to be defined on the same '
                    'parameter space.')

        if len(population_models) != n_parameters:
            raise ValueError(
                'Wrong number of population models. One population model has '
                'to be provided for each model parameters.')

        for pop_model in population_models:
            if not isinstance(pop_model, erlo.PopulationModel):
                raise ValueError(
                    'The population models have to be instances of '
                    'erlotinib.PopulationModel')

        # Remember models and number of individuals
        self._log_likelihoods = log_likelihoods
        self._population_models = population_models
        self._n_ids = len(log_likelihoods)

        # Set IDs
        self._set_ids()

        # Set parameter names and number of parameters
        self._set_number_and_parameter_names()

        # Construct mask for top-level parameters
        self._create_top_level_mask()

    def __call__(self, parameters):
        """
        Returns the log-likelihood score of the model.
        """
        # Transform parameters to numpy array
        parameters = np.asarray(parameters)

        # Compute population model scores
        score = 0
        start = 0
        for pop_model in self._population_models:
            # Get number of individual and population level parameters
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)

            # Get parameter ranges
            end_indiv = start + n_indiv
            end_pop = end_indiv + n_pop

            # Add score
            score += pop_model.compute_log_likelihood(
                parameters=parameters[end_indiv:end_pop],
                observations=parameters[start:end_indiv])

            # Shift start index
            start = end_pop

        # Return if values already lead to a rejection
        if np.isinf(score):
            return score

        # Create container for individual parameters
        individual_params = np.empty(
            shape=(self._n_ids, self._n_indiv_params))

        # Fill conrainer with parameter values
        for param_id, indices in enumerate(self._indiv_params):
            start, end = indices
            if start == end:
                # This parameter is pooled, set all parameters to the same
                # value
                individual_params[:, param_id] = parameters[start]
                continue

            # Set the parameter values to the input values
            individual_params[:, param_id] = parameters[
                start:end]

        # Evaluate individual likelihoods
        for index, log_likelihood in enumerate(self._log_likelihoods):
            score += log_likelihood(individual_params[index, :])

        return score

    def _create_top_level_mask(self):
        """
        Creates a mask that can be used to mask for the top level
        parameters.
        """
        # Create conatainer with all False
        # (False for not top-level)
        top_level_mask = np.zeros(shape=self._n_parameters, dtype=bool)

        # Flip entries to true if top-level parameter
        start = 0
        for pop_model in self._population_models:
            # Get number of hierarchical parameters
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)

            if isinstance(pop_model, erlo.HeterogeneousModel):
                # For heterogeneous models the individual parameters are the
                # top-level parameters
                end = start + n_indiv
                top_level_mask[start: end] = ~top_level_mask[start: end]

            # Add the population parameters as top-level parameters
            # (Heterogeneous model has 0 population parameters)
            start += n_indiv
            end = start + n_pop
            top_level_mask[start: end] = ~top_level_mask[start: end]

            # Shift start to end
            start = end

        # Store mask
        self._top_level_mask = top_level_mask

    def _set_ids(self):
        """
        Sets the IDs of the hierarchical model.

        IDs for population model parameters are ``None``.
        """
        # Get IDs of individual log-likelihoods
        indiv_ids = []
        for index, log_likelihood in enumerate(self._log_likelihoods):
            _id = log_likelihood.get_id()

            # If ID not set, give some arbitrary ID
            if _id is None:
                _id = 'automatic-id-%d' % (index + 1)

            indiv_ids.append(_id)

        # Construct IDs (prefixes) for hierarchical model
        ids = []
        for pop_model in self._population_models:
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)

            # If population model has individual parameters, add IDs
            if n_indiv > 0:
                ids += indiv_ids

            # If population model has population model parameters, add them as
            # prefixes.
            ids += [None] * n_pop

        # Remember IDs
        self._ids = ids

    def _set_number_and_parameter_names(self):
        """
        Sets the number and names of the parameters.

        The model parameters are arranged by keeping the order of the
        parameters of the individual log-likelihoods and expanding them such
        that the parameters associated with individuals come first and the
        the population parameters.

        Example:
        Parameters of hierarchical log-likelihood:
        [
        log-likelihood 1 parameter 1, ..., log-likelihood N parameter 1,
        population model 1 parameter 1, ..., population model 1 parameter K,
        log-likelihood 1 parameter 2, ..., log-likelihood N parameter 2,
        population model 2 parameter 1, ..., population model 2 parameter L,
        ...
        ]
        where N is the number of parameters of the individual log-likelihoods,
        and K and L are the varying numbers of parameters of the respective
        population models.
        """
        # Get individual parameter names
        indiv_names = self._log_likelihoods[0].get_parameter_names()

        # Construct parameter names
        start = 0
        indiv_params = []
        parameter_names = []
        for param_id, pop_model in enumerate(self._population_models):
            # Get number of hierarchical parameters
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)

            # Add a copy of the parameter name for each individual parameter
            name = indiv_names[param_id]
            parameter_names += [name] * n_indiv

            # Add the population parameter name, composed of the population
            # name and the parameter name
            if n_pop > 0:
                # (Reset population parameter names first)
                pop_model.set_parameter_names(None)
                pop_names = pop_model.get_parameter_names()
                parameter_names += [
                    pop_name + ' ' + name for pop_name in pop_names]

            # Remember positions of individual parameters
            end = start + n_indiv
            indiv_params.append([start, end])

            # Shift start index
            start += n_indiv + n_pop

        # Remember parameter names and number of parameters
        self._parameter_names = parameter_names
        self._n_parameters = len(parameter_names)
        self._n_indiv_params = len(indiv_names)

        # Remember positions of individual parameters
        self._indiv_params = indiv_params

    def get_id(self):
        """
        Returns the IDs (prefixes) of the model parameters.
        """
        return self._ids

    def get_parameter_names(
            self, exclude_bottom_level=False, include_ids=False):
        """
        Returns the parameter names of the predictive model.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter names are returned in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        :param include_ids: A boolean flag which determines whether the IDs
            (prefixes) of the model parameters are included.
        :type include_ids: bool, optional
        """
        if include_ids is False:
            # Return names without ids
            if exclude_bottom_level is False:
                return self._parameter_names

            # Exclude bottom level parameters
            names = np.asarray(self._parameter_names)
            names = names[self._top_level_mask]
            return list(names)

        # Construct parameters names as <ID> <Name>
        names = []
        for index in range(self._n_parameters):
            _id = self._ids[index]
            name = self._parameter_names[index]

            # Prepend ID for non-population parameters
            if _id is None:
                names.append(name)
            else:
                names.append(_id + ' ' + name)

        if exclude_bottom_level is True:
            names = np.asarray(names)
            names = names[self._top_level_mask]
            return list(names)

        return names

    def get_population_models(self):
        """
        Returns the population models.
        """
        return self._population_models

    def n_log_likelihoods(self):
        """
        Returns the number of individual likelihoods.
        """
        return self._n_ids

    def n_parameters(self, exclude_bottom_level=False):
        """
        Returns the number of parameters of the log-likelihood.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter are counted in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        """
        if exclude_bottom_level:
            return int(np.sum(self._top_level_mask))

        return self._n_parameters


class HierarchicalLogPosterior(pints.LogPDF):
    r"""
    A log-posterior constructed from a :class:`HierarchicalLoglikelihood`
    and a :class:`LogPrior`.

    A hierarchical log-posterior takes a hierarchical log-likelihood and
    a log-prior of the same dimensionality as top-level parameters in
    the log-likelihood.

    Formally the posterior is constructed for a hierarchical
    log-likelihood of the form :math:`p(x | \Psi , \theta )`
    and a prior :math:`p(\theta )` as

    .. math::
        p(\Psi , \theta | x) =
        p(x | \Psi , \theta )\, p(\theta )

    where :math:`\Psi = (\psi _1, \ldots , \psi _n)` are the bottom-level
    parameters, :math:`\theta` are the top-level parameters and
    :math:`x` are the observations.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    log_likelihood
        An instance of a :class:`erlotinib.HierarchicalLogLikelihood`.
    log_prior
        An instance of a :class:`pints.LogPrior` which represents the prior
        probability distributions for the population parameters of the
        log-likelihood.
    """
    def __init__(self, log_likelihood, log_prior):
        # Check inputs
        super(HierarchicalLogPosterior, self).__init__()

        # Check inputs
        if not isinstance(log_likelihood, HierarchicalLogLikelihood):
            raise ValueError(
                'Log-likelihood has to extend pints.LogPDF.')
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError(
                'Log-prior has to extend pints.LogPrior.')

        # Check dimensions
        n_top_parameters = log_likelihood.n_parameters(
            exclude_bottom_level=True)
        if log_prior.n_parameters() != n_top_parameters:
            raise ValueError(
                'The log-prior has to have as many parameters as population '
                'parameters in the log-likelihood.')

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
        """
        # Create conatainer with all False
        # (False for not top-level)
        top_level_mask = np.zeros(shape=self._n_parameters, dtype=bool)

        # Flip entries to true if top-level parameter
        start = 0
        n_ids = self._log_likelihood.n_log_likelihoods()
        population_models = self._log_likelihood.get_population_models()
        for pop_model in population_models:
            # Get number of hierarchical parameters
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(n_ids)

            if isinstance(pop_model, erlo.HeterogeneousModel):
                # For heterogeneous models the individual parameters are the
                # top-level parameters
                end = start + n_indiv
                top_level_mask[start: end] = ~top_level_mask[start: end]

            # Add the population parameters as top-level parameters
            # (Heterogeneous model has 0 population parameters)
            start += n_indiv
            end = start + n_pop
            top_level_mask[start: end] = ~top_level_mask[start: end]

            # Shift start to end
            start = end

        # Store mask
        self._top_level_mask = top_level_mask

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
        Returns the id of the log-posterior. If no ID is set, ``None`` is
        returned.
        """
        return self._log_likelihood.get_id()

    def get_parameter_names(
            self, exclude_bottom_level=False, include_ids=False):
        """
        Returns the parameter names of the predictive model.

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
        Returns the number of parameters of the log-likelihood.

        :param exclude_bottom_level: A boolean flag which determines whether
            the bottom-level parameter are counted in addition to the
            top-level parameters.
        :type exclude_bottom_level: bool, optional
        """
        return self._log_likelihood.n_parameters(exclude_bottom_level)


class LogLikelihood(pints.LogPDF):
    r"""
    A log-likelihood quantifies how likely a set of mechanistic and error model
    parameters are to have produced some observations.

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
        L(\psi | x^{\text{obs}}) = \sum _{i=1}^N
        \log p(x^{\text{obs}}_i | t^{\text{obs}}_i; \psi),

    where :math:`N` is the total number of observations. Note that for
    notational ease we omitted the conditioning on the observation times
    :math:`t^{\text{obs}}` on the left hand side, and will also often drop
    it elsewhere in the documentation

    .. note::
        For notational ease we omitted that the log-likelihood also is
        conditional on the dosing regimen associated with the observations.
        The appropriate regimen can be set with
        :meth:`PharmacokineticModel.set_dosing_regimen`

    Example
    -------

    ::

        # Create log-likelihood
        log_likelihood = erlotinib.LogLikelihood(
            mechanistic_model,
            error_models,
            observations,
            times)

        # Compute log-likelihood score
        score = log_likelihood(parameters)

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
        measured values of the biomarkers. The list is expected to ordered in
        the same way as the mechanistic model outputs.
    :type observations: list[float], list[list[float]]
    :param times: A list of one dimensional array-like objects with measured
        times associated to the observations.
    :type times: list[float], list[list[float]]
    :param outputs: A list of output names, which sets the mechanistic model
        outputs. If ``None`` the currently set outputs of the mechanistic model
        are assumed.
    :type outputs: list[str], optional
    """
    def __init__(
            self, mechanistic_model, error_model, observations, times,
            outputs=None):
        super(LogLikelihood, self).__init__()

        # Check inputs
        if not isinstance(
                mechanistic_model,
                (erlo.MechanisticModel, erlo.ReducedMechanisticModel)):
            raise TypeError(
                'The mechanistic model as to be an instance of a '
                'erlotinib.MechanisticModel.')

        if not isinstance(error_model, list):
            error_model = [error_model]

        # Copy mechanistic model
        mechanistic_model = copy.deepcopy(mechanistic_model)

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
                    em, (erlo.ErrorModel, erlo.ReducedErrorModel)):
                raise TypeError(
                    'The error models have to instances of a '
                    'erlotinib.ErrorModel.')

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
        # Solve the mechanistic model
        outputs = self._mechanistic_model.simulate(
            parameters=parameters[:self._n_mechanistic_params],
            times=self._times)

        # Remember only error parameters
        parameters = parameters[self._n_mechanistic_params:]

        # Compute log-likelihood score
        score = 0
        start = 0
        for output_id, error_model in enumerate(self._error_models):
            # Get relevant mechanistic model outputs and parameters
            output = outputs[output_id, self._obs_masks[output_id]]
            end = start + self._n_error_params[output_id]
            params = parameters[start:end]

            # Compute log-likelihood score for this output
            score += error_model.compute_log_likelihood(
                parameters=params,
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

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``
        sets the parameter free again.

        :param name_value_dict: A dictionary with model parameter names as keys,
            and parameter value as values.
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
        if not isinstance(mechanistic_model, erlo.ReducedMechanisticModel):
            mechanistic_model = erlo.ReducedMechanisticModel(mechanistic_model)
        for model_id, error_model in enumerate(error_models):
            if not isinstance(error_model, erlo.ReducedErrorModel):
                error_models[model_id] = erlo.ReducedErrorModel(error_model)

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

    def get_id(self):
        """
        Returns the ID of the log-likelihood. If not set, ``None`` is returned.

        The ID is used as meta data to identify the origin of the data.
        """
        return self._id

    def get_parameter_names(self):
        """
        Returns the parameter names of the predictive model.
        """
        return self._parameter_names

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
        if isinstance(mechanistic_model, erlo.ReducedMechanisticModel):
            mechanistic_model = mechanistic_model.mechanistic_model()

        error_models = []
        for error_model in self._error_models:
            # Get original error model
            if isinstance(error_model, erlo.ReducedErrorModel):
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

    def set_id(self, label):
        """
        Sets the ID of the log-likelihood.

        The ID is used as meta data to identify the origin of the data.

        :param label: Integer value which is used as ID for the
            log-likelihood.
        :type label: str
        """
        label = int(label)

        # Construct ID as <ID: #> for convenience
        self._id = 'ID ' + str(label)


class LogPosterior(pints.LogPDF):
    r"""
    A log-posterior class which can be used with the
    :class:`OptimisationController` or the :class:`SamplingController`
    to find either the maximum a posteriori (MAP)
    estimates of the model parameters, or to sample from the posterior
    probability distribution of the model parameters directly.

    The log-posterior is constructed by the sum of the input log-likelihood
    :math:`\log p(x ^{\text{obs}} | \psi )` and the input log-prior
    :math:`\log p(\psi )` up to an additive constant

    .. math::
        \log p(\psi | x ^{\text{obs}}) =
        \log p(x ^{\text{obs}} | \psi ) + \log p(\psi ) + \text{constant},

    where :math:`\psi` are the parameters of the log-likelihood and
    :math:`x ^{\text{obs}}` are the observed data. The additive constant
    is the normalisation of the log-posterior and is generally not known.

    Extends :class:`pints.LogPDF`.

    :param log_likelihood: A log-likelihood for the model parameters.
    :type log_likelihood: LogLikelihood
    :param log_prior: A log-prior for the model parameters. The log-prior
        has to have the same dimensionality as the log-likelihood.
    :type log_prior: pints.LogPrior
    """
    def __init__(self, log_likelihood, log_prior):
        # Check inputs
        super(LogPosterior, self).__init__()

        # Check inputs
        if not isinstance(log_likelihood, LogLikelihood):
            raise ValueError(
                'Log-likelihood has to extend erlotinib.LogLikelihood.')
        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError(
                'Log-prior has to extend pints.LogPrior.')

        # Check dimensions
        n_parameters = log_prior.n_parameters()
        if log_likelihood.n_parameters() != n_parameters:
            raise ValueError(
                'Given log_prior and log_likelihood must have same dimension.')

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
        # Get ID of likelihood
        try:
            _id = self._log_likelihood.get_id()
        except AttributeError:
            # If a pints likelihood is used, it won't have an ID
            _id = None

        return _id

    def get_parameter_names(self):
        """
        Returns the names of the model parameters. By default the parameters
        are enumerated and assigned with the names 'Param #'.
        """
        # Get parameter names
        names = self._log_likelihood.get_parameter_names()

        return names

    def n_parameters(self):
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
