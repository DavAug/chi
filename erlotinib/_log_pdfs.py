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


class HierarchicalLogLikelihood(pints.LogPDF):
    """
    A hierarchical log-likelihood class which can be used for population-level
    inference.

    A hierarchical log-likelihood takes a list of :class:`LogLikelihood`
    instances, and a list of :class:`PopulationModel` instances. Each
    :class:`LogLikelihood` in the list is expected to model an independent
    dataset, and must be defined on the same parameter space. For example,
    one likelihood is defined for each patient in a clinical trial.

    For each parameter of the :class:`LogLikelihood`, a
    :class:`PopulationModel` has to be provided which models the
    distribution of the respective parameter across individuals in the
    population.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------

    log_likelihoods
        A list of :class:`LogLikelihood` instances defined on the same
        parameter space.
    population_models
        A list of :class:`PopulationModel` instances with one
        population model for each parameter of the log-likelihoods.
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

    def __call__(self, parameters):
        """
        Returns the log-likelihood score of the model.
        """
        # Transform parameters to numpy array
        parameters = np.asarray(parameters)

        # Compute population model scores
        score = 0
        start_index = 0
        for pop_model in self._population_models:
            # Get number of individual and population level parameters
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)

            # Get parameter ranges
            end_indiv = start_index + n_indiv
            end_pop = end_indiv + n_pop

            # Add score
            score += pop_model.compute_log_likelihood(
                parameters=parameters[end_indiv:end_pop],
                observations=parameters[start_index:end_indiv])

            # Shift start index
            start_index = end_pop

        # Return if values already lead to a rejection
        if score == -np.inf:
            return score

        # Create container for individual parameters
        individual_params = np.empty(
            shape=(self._n_ids, self._n_indiv_params))

        # Fill conrainer with parameter values
        for param_id, indices in enumerate(self._indiv_params):
            start_index = indices[0]
            end_index = indices[1]

            if start_index == end_index:
                # This parameter is pooled, set all parameters to the same
                # value
                individual_params[:, param_id] = parameters[start_index]
                continue

            # Set the parameter values to the input values
            individual_params[:, param_id] = parameters[
                start_index:end_index]

        # Evaluate individual likelihoods
        for index, log_likelihood in enumerate(self._log_likelihoods):
            score += log_likelihood(individual_params[index, :])

        return score

    def _set_ids(self):
        """
        Sets the IDs of the hierarchical model.

        The IDs can also be interpreted as prefix of the model parameters,
        which allow to distingish parameters of the same name.
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
            if n_pop > 0:
                # Reset parameter names to original names
                pop_model.set_parameter_names(None)

                # Add population parameters
                names = pop_model.get_parameter_names()
                ids += names

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
            n_parameters = n_indiv + n_pop

            # Add a copy of the parameter name for each hierarchical parameter
            parameter_names += [indiv_names[param_id]] * n_parameters

            # Remember positions of individual parameters
            end = start + n_indiv
            indiv_params.append([start, end])

            # Shift start index
            start += n_parameters

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

    def get_parameter_names(self, include_ids=False):
        """
        Returns the parameter names of the predictive model.

        Parameters
        ----------
        include_ids
            A boolean flag which determines whether the IDs (prefixes) of the
            model parameters are included.
        """
        if not include_ids:
            # Return names without ids
            return self._parameter_names

        # Construct parameters names as <ID> <Name>
        names = []
        for index in range(self._n_parameters):
            _id = self._ids[index]
            name = self._parameter_names[index]
            names.append(_id + ' ' + name)

        return names

    def get_population_models(self):
        """
        Returns the population models.
        """
        return copy.copy(self._population_models)

    def n_log_likelihoods(self):
        """
        Returns the number of individual likelihoods.
        """
        return len(self._log_likelihoods)

    def n_parameters(self):
        """
        Returns the number of parameters of the hierarchical log-likelihood.
        """
        return self._n_parameters


class LogLikelihood(pints.LogPDF):
    r"""
    A class which defines the log-likelihood of the model parameters.

    A log-likelihood takes an instance of a :class:`MechanisticModel` and one
    instance of a :class:`ErrorModel` for each mechanistic model output. These
    submodels define a time-dependent distribution of observable biomarkers
    equivalent to a :class:`PredictiveModel`

    .. math::
        p(x | t; \psi _{\text{m}}, \psi _{\text{e}}),

    where :math:`p` is the probability density of the observable biomarker
    :math:`x` at time :math:`t`. :math:`\psi _{\text{m}}` and
    :math:`\psi _{\text{e}}` are the model parameters of the mechanistic model
    and the error model respectively. For multiple outputs of the mechanistic
    model, this distribution will be multi-dimensional.

    The log-likelihood for given observations and times is the given by
    the sum of :math:`\log p` evaluated at the observations

    .. math::
        L(\psi _{\text{m}}, \psi _{\text{e}}) = \sum _{i=1}^N
        \log p(x^{\text{obs}}_i | t^{\text{obs}}_i;
        \psi _{\text{m}}, \psi _{\text{e}}),

    where :math:`N` is the total number of observations, and
    :math:`x^{\text{obs}}` and :math:`t^{\text{obs}}` the observed biomarker
    values and times.

    The error models are expected to be in the same order as the mechanistic
    model outputs :meth:`MechanisticModel.outputs`. The observations and times
    are equally expected to order in the same way as the model outputs.

    Calling the log-likelihood for some parameters returns the unnormalised
    log-likelihood score for those paramters.

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

    .. note::
        The parameters are expected to be ordered according to the mechanistic
        model and error models, where the mechanistic model parameters are
        first, then the parameters of the first error model, then the
        parameters of the second error model, etc.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    mechanistic_model
        An instance of a :class:`MechanisticModel`.
    error_models
        A list of instances of a :class:`ErrorModel`. The error models are
        expected to be ordered in the same way as the mechanistic model
        outputs.
    observations
        A list of one dimensional array-like objects with measured values of
        the biomarkers. The list is expected to ordered in the same way as the
        mechanistic model outputs.
    times
        A list of one dimensional array-like objects with measured times
        associated to the observations.
    outputs
        A list of output names, which sets the mechanistic model outputs. If
        ``None`` the previously set outputs are assumed.
    """
    def __init__(
            self, mechanistic_model, error_models, observations, times,
            outputs=None):
        super(LogLikelihood, self).__init__()

        # Check inputs
        if not isinstance(
                mechanistic_model,
                (erlo.MechanisticModel, erlo.ReducedMechanisticModel)):
            raise TypeError(
                'The mechanistic model as to be an instance of a '
                'erlotinib.MechanisticModel.')

        if not isinstance(error_models, list):
            error_models = [error_models]

        # Copy mechanistic model
        mechanistic_model = copy.deepcopy(mechanistic_model)

        # Set outputs
        if outputs is not None:
            mechanistic_model.set_outputs(outputs)

        n_outputs = mechanistic_model.n_outputs()
        if len(error_models) != n_outputs:
            raise ValueError(
                'One error model has to be provided for each mechanistic '
                'model output.')

        for error_model in error_models:
            if not isinstance(
                    error_model, (erlo.ErrorModel, erlo.ReducedErrorModel)):
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
        error_models = [
            copy.deepcopy(error_model) for error_model in error_models]

        # Remember models and observations
        # (Mechanistic model needs to be copied, such that it's dosing regimen
        # cannot be altered by the input model.)
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models
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
        Sets the evaluation time points for the mechanistic time points.

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
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        Parameters
        ----------
        name_value_dict
            A dictionary with model parameter names as keys, and parameter
            value as values.
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
        Returns the submodels of the predictive model in form of a dictionary.
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

        Parameters
        ----------
        label
            Integer value which is used as ID for the log-likelihood.
        """
        label = int(label)

        # Construct ID as <ID: #> for convenience
        self._id = 'ID ' + str(label)


class LogPosterior(pints.LogPosterior):
    """
    A log-posterior class which can be used with the
    :class:`OptimisationController` or the :class:`SamplingController`
    to find either the maximum a posteriori (MAP)
    estimates of the model parameters, or to sample from the posterior
    probability distribution of the model parameters directly.

    Extends :class:`pints.LogPosterior`.

    Parameters
    ----------

    log_likelihood
        An instance of a :class:`pints.LogPDF`.
    log_prior
        An instance of a :class:`pints.LogPrior` which represents the prior
        probability distributions for the parameters of the log-likelihood.
    """

    def __init__(self, log_likelihood, log_prior):
        super(LogPosterior, self).__init__(log_likelihood, log_prior)

        # Set defaults
        n_params = self._n_parameters
        self._default_names = ['Parameter %d' % (n+1) for n in range(n_params)]

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
        try:
            names = self._log_likelihood.get_parameter_names()
        except AttributeError:
            # If a pints likelihood is used, it won't have an parameter names
            names = self._default_names

        return names


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
