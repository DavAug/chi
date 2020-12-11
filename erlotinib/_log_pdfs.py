#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

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
    dataset, and must be defined on the same parameter space. For parameter of
    the :class:`pints.LogPDF` instances, a :class:`erlotinib.PopulationModel`
    has to be provided which models the distribution of the respective
    parameter across individuals in the population.

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
                    'The log-likelihoods have to be instances of pints.LogPDF.'
                )

        n_parameters = log_likelihoods[0].n_parameters()
        for log_likelihood in log_likelihoods:
            if log_likelihood.n_parameters() != n_parameters:
                raise ValueError(
                    'All log-likelihoods have to be defined on the same '
                    'parameter space.')

        for pop_model in population_models:
            if not isinstance(pop_model, erlo.PopulationModel):
                raise ValueError(
                    'The population models have to be instances of '
                    'erlotinib.PopulationModel')

        n_ids = len(log_likelihoods)
        for pop_model in population_models:
            if pop_model.n_ids() != n_ids:
                raise ValueError(
                    "Population models' n_ids have to coincide with the "
                    'number of log-likelihoods.')

        n_params_per_id = 0
        for pop_model in population_models:
            n_params_per_id += pop_model.n_parameters_per_id()

        if n_params_per_id != n_parameters:
            raise ValueError(
                'Each likelihood parameter has to be assigned to a population '
                'model. The cumulative number of parameters per individual of '
                'the population models does however not sum up to the number '
                'of parameters per likelihood.')

        self._log_likelihoods = log_likelihoods
        self._population_models = population_models

        # Save number of ids and number of parameters per likelihood
        self._n_ids = n_ids
        self._n_individual_params = n_parameters

        # Save total number of parameters
        n_parameters = 0
        for pop_model in self._population_models:
            n_parameters += pop_model.n_parameters()

        self._n_parameters = n_parameters

        # Construct a mask for the indiviudal parameters
        # Parameters will be order as
        # [[psi_i0], [theta _0k], [psi_i1], [theta _1k], ...]
        # where [psi_ij] is the jth parameter in the ith likelihood, and
        # theta_jk is the corresponding kth parameter of the population model.
        indices = []
        start_index = 0
        for pop_model in population_models:
            # Get end index for individual parameters
            end_index = start_index + pop_model.n_bottom_parameters()

            # If parameter is pooled, i.e. mark end_index as None for later
            # convenience
            if isinstance(pop_model, erlo.PooledModel):
                end_index = None

            # Save start and end index
            indices.append([start_index, end_index])

            # Increment start index by total number of population model
            # parameters
            start_index += pop_model.n_parameters()

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
        for pop_models in self._population_models:
            # Compute likelihood score
            end_index = start_index + pop_models.n_parameters()
            score += pop_models(parameters[start_index:end_index])

            # Shift start index
            start_index += end_index

        # TODO: return if values are already broken

        # Create container for individual values
        individual_params = np.empty(
            shape=(self._n_ids, self._n_individual_params))

        # Fill conrainer with parameter values
        for param_id, indices in enumerate(self._individual_params):
            start_index = indices[0]
            end_index = indices[1]

            if end_index is None:
                # This parameter is pooled. Leverage broadcasting
                individual_params[:, param_id] = parameters[start_index]
            # TODO: temporarily commented out, as long no other pop models
            # exist.
            # else:
            #     individual_params[:, param_id] = parameters[
            #         start_index:end_index]

        # Evaluate individual likelihoods
        for index, log_likelihood in enumerate(self._log_likelihoods):
            score += log_likelihood(individual_params[index, :])

        return score

    def n_parameters(self):
        """
        Returns the number of parameters of the hierarchical log-likelihood.
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
        Sets the posterior id.

        This can be used to tag the log-posterior to distinguish it from
        other structurally identical log-posteriors, e.g. when the same
        model is used to describe the PKPD of different individuals.

        Parameters
        ----------

        posterior_id
            An ID that can be used to identify the log-posterior. A valid ID
            has to be convertable to a string object.
        """
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
