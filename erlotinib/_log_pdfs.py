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

    A hierarchical log-likelihood takes a list of :class:`pints.LogPDF`
    instances, and a list of :class:`erlotinib.PopulationModel` instances. Each
    :class:`pints.LogPDF` in the list is expected to model an independent
    dataset, and must be defined on the same parameter space. For each
    parameter of the :class:`pints.LogPDF` instances, a
    :class:`erlotinib.PopulationModel` has to be provided which models the
    distribution of the respective parameter across individuals in the
    population.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------

    log_likelihoods
        A list of :class:`pints.LogPDF` instances defined on the same
        parameter space.
    population_models
        A list of :class:`erlotinib.PopulationModel` instances with one
        population model for each parameter of the log-likelihoods.
    """

    def __init__(self, log_likelihoods, population_models):
        super(HierarchicalLogLikelihood, self).__init__()

        for log_likelihood in log_likelihoods:
            if not isinstance(log_likelihood, pints.LogPDF):
                raise ValueError(
                    'The log-likelihoods have to be instances of a '
                    'pints.LogPDF.')

        n_parameters = log_likelihoods[0].n_parameters()
        for log_likelihood in log_likelihoods:
            if log_likelihood.n_parameters() != n_parameters:
                raise ValueError(
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

        self._log_likelihoods = log_likelihoods
        self._population_models = population_models

        # Save number of ids and number of parameters per likelihood
        self._n_individual_params = n_parameters

        # Save total number of parameters
        self._n_ids = len(self._log_likelihoods)
        n_parameters = 0
        for pop_model in self._population_models:
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)
            n_parameters += n_indiv + n_pop

        self._n_parameters = n_parameters

        # Construct a mask for the indiviudal parameters
        # Parameters will be ordered as
        # [[psi_i0], [theta _0k], [psi_i1], [theta _1k], ...]
        # where [psi_ij] is the jth parameter in the ith likelihood, and
        # theta_jk is the corresponding kth parameter of the population model.
        indices = []
        start_index = 0
        for pop_model in population_models:
            # Get number of individual and population level parameters
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(self._n_ids)

            # Get end index for individual parameters
            end_index = start_index + n_indiv

            # Save start and end index
            indices.append([start_index, end_index])

            # Increment start index by total number of parameters
            start_index += n_indiv + n_pop

        self._individual_params = indices

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
            shape=(self._n_ids, self._n_individual_params))

        # Fill conrainer with parameter values
        for param_id, indices in enumerate(self._individual_params):
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

    def get_log_likelihoods(self):
        """
        Returns the log-likelihoods.
        """
        return copy.copy(self._log_likelihoods)

    def get_population_models(self):
        """
        Returns the population models.
        """
        return copy.copy(self._population_models)

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
    """
    def __init__(self, mechanistic_model, error_models, observations, times):
        super(LogLikelihood, self).__init__()

        # Check inputs
        if not isinstance(mechanistic_model, erlo.MechanisticModel):
            raise TypeError(
                'The mechanistic model as to be an instance of a '
                'erlotinib.MechanisticModel.')

        if not isinstance(error_models, list):
            error_models = [error_models]

        n_outputs = mechanistic_model.n_outputs()
        if len(error_models) != n_outputs:
            raise ValueError(
                'One error model has to be provided for each mechanistic '
                'model output.')

        for error_model in error_models:
            if not isinstance(error_model, erlo.ErrorModel):
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

        # Transform observations and times to numpy arrays
        observations = [np.array(obs) for obs in observations]
        times = [np.array(t) for t in times]

        # Make sure that the observation-time pairs match
        for output_id, output_times in enumerate(times):
            if observations[output_id].shape != output_times.shape:
                raise ValueError(
                    'The observations and times have to be of the same '
                    'dimension.')

            if observations[output_id].ndim != 1:
                raise ValueError(
                    'The observations for each output have to be provided '
                    'as a one dimensional array-like object.')

            # Sort times and observations
            order = np.argsort(output_times)
            times[output_id] = output_times[order]
            observations[output_id] = observations[output_id][order]

        self._mechanistic_model = copy.deepcopy(mechanistic_model)
        self._error_models = error_models
        self._observations = observations

        self._arange_times_for_mechanistic_model(times)

        # Get number of parameters
        self._n_mechanistic_params = self._mechanistic_model.n_parameters()
        self._n_error_params = [em.n_parameters() for em in error_models]
        self._n_parameters = int(
            self._n_mechanistic_params + np.sum(self._n_error_params))

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
        unique_times = np.array(unique_times)

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

        self._times = unique_times
        self._obs_masks = obs_masks

    def get_error_models(self):
        """
        Returns the error models.
        """
        return copy.copy(self._error_models)

    def get_mechanistic_model(self):
        """
        Returns the mechanistic model.
        """
        return copy.deepcopy(self._mechanistic_model)

    def n_parameters(self):
        """
        Returns the number of parameters.
        """
        return self._n_parameters


class LogPosterior(pints.LogPosterior):
    """
    A log-posterior class which can be used with the
    :class:`OptimisationController` or the :class:`SamplingController`
    to find either the maximum a posteriori
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
        self._id = None
        n_params = self._n_parameters
        self._parameter_names = ['Param %d' % (n+1) for n in range(n_params)]

    def get_id(self):
        """
        Returns the id of the log-posterior. If no id is set, ``None`` is
        returned.
        """
        return self._id

    def get_parameter_names(self):
        """
        Returns the names of the model parameters. By default the parameters
        are enumerated and assigned with the names 'Param #'.
        """
        return self._parameter_names

    def set_id(self, posterior_id):
        """
        Sets the posterior id(s).

        This can be used to tag the log-posterior to distinguish it from
        other structurally identical log-posteriors, e.g. when the same
        model is used to describe the PKPD of different individuals.

        Alternatively a list of IDs may be provided which sets the ID for
        each model parameter individually. This may be useful for
        log-posteiors that are derived from a
        :class:`HierarchicalLoglikelihood`.

        Parameters
        ----------
        posterior_id
            An ID (or a list of IDs) that can be used to identify the
            log-posterior. A valid ID has to be convertable to a string
            object, or be a list of length of IDs of length ``n_parameters``.
        """
        if isinstance(posterior_id, list):
            if len(posterior_id) != self.n_parameters():
                raise ValueError(
                    'If a list of IDs is provided, it needs to be of the same '
                    'length as the number of parameters.')

            self._id = [str(label) for label in posterior_id]

        else:
            self._id = str(posterior_id)

    def set_parameter_names(self, names):
        """
        Sets the names of the model parameters.

        The list of parameters has to match the length of the number of
        parameters. The first parameter name in the list is assigned to the
        first parameter, the second name in the list is assigned to second
        parameter, and so on.

        Parameters
        ----------

        names
            A list of string-convertable objects that is used to assign names
            to the model parameters.
        """
        if len(names) != self._n_parameters:
            raise ValueError(
                'The list of parameter names has to match the number of model '
                'parameters.')
        self._parameter_names = [str(name) for name in names]


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
