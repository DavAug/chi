#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#
# The InverseProblem class is based on the SingleOutputProblem and
# MultiOutputProblem classes of PINTS (https://github.com/pints-team/pints/),
# which is distributed under the BSD 3-clause license.
#

import numpy as np
import pandas as pd
import pints

import erlotinib as erlo


class ProblemModellingController(object):
    """
    A controller class for the modelling of PKPD datasets.

    - data
    - mechanistic PK, PD, or PKPD model
    - error model
    - (optional) Population model
    """

    def __init__(
            self, data, id_key='ID', time_key='Time', biom_keys=['Biomarker']):
        super(ProblemModellingController, self).__init__()

        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                'Data has to be pandas.DataFrame.')

        keys = [id_key, time_key] + biom_keys
        for key in keys:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        self._id_key, self._time_key = keys[:2]
        self._biom_keys = keys[2:]
        self._data = data[keys]

        # Set defaults
        self._mechanistic_model = None
        self._error_model = None
        self._population_model = None
        self._log_prior = None
        self._n_parameters = None
        self._parameter_names = None
        self._fixed_params_mask = None
        self._fixed_params_values = None

    def _create_error_model(self, log_likelihoods):
        """
        Returns a list of log-likelihoods, one for each individual in the
        dataset.
        """
        # Raise error if the model has more than one output.
        # How we compose likelihoods with possibly different error models for
        # different outputs is not yet implemented.
        if len(log_likelihoods) > 1:
            raise NotImplementedError(
                'Fitting multiple outputs has not been implemented yet.')

        # This only works for single output problem!
        biom_key = self._biom_keys[0]
        log_likelihood = log_likelihoods[0]

        # Create a likelihood for each individual
        error_model = []
        ids = self._data[self._id_key].unique()
        for individual in ids:
            # Get data
            # TODO: what happens if data includes nans? Should we exclude those
            # rows?
            mask = self._data[self._id_key] == individual
            times = self._data[self._time_key][mask].to_numpy()
            biomarker = self._data[biom_key][mask].to_numpy()

            # Create inverse problem
            problem = InverseProblem(self._mechanistic_model, times, biomarker)
            try:
                error_model.append(log_likelihood(problem))
            except ValueError:
                raise ValueError(
                    'Only error models for which all parameters can be '
                    'inferred are compatible.')

        return error_model

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them from
        the list of model parameters.

        Fixing the value of a parameter at ``None``, sets the parameter free
        again.

        If a parameter name in the name-value dictionary cannot be found in the
        model, it is ignored.

        Fixing model parameters resets the log-prior to ``None``.

        Parameters
        ----------

        name_value_dict
            A dictionary with model parameters as keys, and the value to be
            fixed at as values.
        """
        if self._mechanistic_model is None:
            raise ValueError(
                'The mechanistic model has not been set.')

        if self._error_model is None:
            raise ValueError(
                'The error model has not been set.')

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

    def log_posteriors(self):
        """
        Returns a list of :class:`erlotinib.LogPosterior` instances, defined by
        the dataset, the mechanistic model, the error model, the log-prior,
        and optionally the population model and the fixed model parameters.

        If a population model has been set, the list will contain only a single
        log-posterior for the populational inference. If no population model
        has been set, the list contains a log-posterior for each individual
        separately.

        This method raises an error if the mechanistic model, the error
        model, or the log-prior has not been set. They can be set with
        :meth:`set_mechanistic_model`, :meth:`set_error_model` and
        :meth:`set_log_prior`.
        """
        if self._mechanistic_model is None:
            raise ValueError(
                'The mechanistic model has not been set.')

        if self._error_model is None:
            raise ValueError(
                'The error model has not been set.')
        log_likelihoods = self._error_model

        if self._log_prior is None:
            raise ValueError(
                'The log-prior has not been set.')

        # if self._population_model is not None:
        #     # Overwrites the log-likelihoods
        #     raise NotImplementedError

        if self._fixed_params_values is not None:
            # Fix model parameters
            values = self._fixed_params_values[self._fixed_params_mask]
            for index, log_likelihood in enumerate(log_likelihoods):
                log_likelihoods[index] = erlo.ReducedLogPDF(
                    log_pdf=log_likelihood,
                    mask=self._fixed_params_mask,
                    values=values)

        # Compose the log-posterior
        log_posterior = []
        for log_likelihood in log_likelihoods:
            log_posterior.append(
                pints.LogPosterior(log_likelihood, self._log_prior))

            # Set parameter names (with ID tag)

        return log_posterior

    def n_parameters(self, exclude_pop_model=False):
        """
        Returns the number of free parameters of the structural model, i.e. the
        mechanistic model, the error model and, if set, the population model.

        Any parameters that have been fixed to a constant value will not be
        included in the number of model parameters.

        If the mechanistic model or the error model have not been set ``None``
        is returned. If the population model has not been set, only the number
        of parameters for one structural model is returned, as the models are
        structurally the same across individuals.
        """
        # if exclude_pop_model:
        #     # Get number of mechanistic model and error model parameters
        #     n_parameters = self._error_model[0].n_parameters()

        #     # If no parameters have been fixed, return
        #     if self._fixed_params_mask is None:
        #         return n_parameters

        #     # If parameters have been fixed, subtract fixed number
        #     n_fixed_params = int(np.sum(
        #         self._fixed_params_mask[:n_parameters]))

        #     return n_parameters - n_fixed_params

        # Return all free model parameters (including population model)
        if self._fixed_params_mask is None:
            return self._n_parameters

        # Subtract the number of fixed parameters
        n_fixed_params = int(np.sum(self._fixed_params_mask))

        return self._n_parameters - n_fixed_params

    def parameter_names(self, exclude_pop_model=False):
        """
        Returns the names of the free structural model parameters, i.e. the
        parameters of the mechanistic model, the error model and optionally
        the population model.

        Any parameters that have been fixed to a constant value will not be
        included in the list of model parameters.

        If the mechanistic model or the error model have not been set ``None``
        is returned. If the population model has not been set, only the names
        of parameters for one structural model is returned, as the models are
        structurally the same across individuals.
        """
        # TODO: figure out a way to return pop names and withour piop
        if self._fixed_params_mask is None:
            return self._parameter_names

        # Remove fixed parameters
        parameter_names = np.array(self._parameter_names)[
            ~self._fixed_params_mask]

        return list(parameter_names)

    def set_error_model(self, log_likelihoods, outputs=None):
        """
        Sets the error model for each observed biomarker.

        An error model is a subclass of a
        :class:`pints.ProblemLogLikelihood`.

        The error models capture the deviations of the measured biomarkers
        from the outputs of the mechanistic model in form of a
        probabilistic model.

        Setting the error model, resets the population model, the prior
        probability distributions for the model parameters, any fixed model
        parameters, and parameter names.

        Parameters
        ----------

        log_likelihoods
            A list of :class:`pints.ProblemLikelihood` daughter classes which
            specify the error model for each measured biomarker.
        outputs
            A list of the model outputs, which maps the error models to the
            model outputs. By default the error models are assumed to be
            listed in the same order as the model outputs.
        """
        if self._mechanistic_model is None:
            raise ValueError(
                'Before setting an error model for the mechanistic model '
                'outputs, a mechanistic model has to be set.')

        for log_likelihood in log_likelihoods:
            if not issubclass(log_likelihood, pints.ProblemLogLikelihood):
                raise ValueError(
                    'The log-likelihoods are not subclasses of '
                    'pints.ProblemLogLikelihood.')

        if len(log_likelihoods) != self._mechanistic_model.n_outputs():
            raise ValueError(
                'The number of log-likelihoods does not match the number of '
                'model outputs.')

        if outputs is not None:
            model_outputs = self._mechanistic_model.outputs()
            if sorted(list(outputs)) != sorted(model_outputs):
                raise ValueError(
                    'The specified outputs do not match the model outputs.')

            # Sort likelihoods according to outputs
            ordered = []
            for output in model_outputs:
                index = outputs.index(output)
                ordered.append(log_likelihoods[index])

            log_likelihoods = ordered

        # Create one log-likelihood for each individual
        # (likelihoods are identical except for the data)
        self._error_model = self._create_error_model(log_likelihoods)

        # Update number of parameters and names for each likelihood
        self._n_parameters = self._error_model[0].n_parameters()
        n_noise_parameters = self._n_parameters \
            - self._mechanistic_model.n_parameters()
        self._parameter_names = self._mechanistic_model.parameters() + [
            'Noise param %d' % (n+1) for n in range(n_noise_parameters)]

        # Reset other settings that depend on the error model
        self._population_model = None
        self._log_prior = None
        self._fixed_params_mask = None
        self._fixed_params_values = None

    def set_log_prior(self, log_priors, param_names=None):
        """
        Sets the log-prior probability distribution of the model parameters.

        The log-priors input is a list of :class:`pints.LogPrior` instances of
        the same length as the number of parameters, :meth:`n_parameters`.

        Dependence between model parameters is currently not supported. Each
        model parameter is assigned with an independent prior distribution,
        i.e. the joint log-prior for the model parameters is assumed to be a
        product of the marginal log-priors.

        If a population model has not been set, the provided log-prior is used
        for the parameters across all individuals.

        By default the log-priors are assumed to be ordered according to
        :meth:`parameter_names`. Alternatively, the mapping of the log-priors
        can be specified explicitly with `param_names`.

        Parameters:
        ----------

        log_priors
            A list of :class:`pints.LogPrior` of the length
            :meth:`n_parameters`.
        param_names
            A list of model parameter names, which is used to map the
            log-priors to the model parameters.
        """
        # Check inputs
        if self._mechanistic_model is None:
            raise ValueError(
                'Before setting log-priors for the model parameters, a '
                'mechanistic model has to be set.')

        if self._error_model is None:
            raise ValueError(
                'Before setting log-priors for the model parameters, an '
                'error model has to be set.')

        for log_prior in log_priors:
            if not isinstance(log_prior, pints.LogPrior):
                raise ValueError(
                    'All marginal log-priors have to be instances of a '
                    'pints.LogPrior.')

        if len(log_priors) != self.n_parameters():
            raise ValueError(
                'One marginal log-prior has to be provided for each parameter.'
            )

        n_parameters = 0
        for log_prior in log_priors:
            n_parameters += log_prior.n_parameters()

        if n_parameters != self.n_parameters():
            raise ValueError(
                'The joint log-prior does not match the dimensionality of the '
                'problem. At least one of the marginal log-priors has to be '
                'multi-dimensional.')

        if param_names is not None:
            if sorted(list(param_names)) != sorted(self.parameter_names()):
                raise ValueError(
                    'The specified parameter names do not match the model '
                    'parameter names.')

            # Sort log-priors according to parameter names
            ordered = []
            names = self.parameter_names()
            for name in names:
                index = param_names.index(name)
                ordered.append(log_priors[index])

            log_priors = ordered

        self._log_prior = pints.ComposedLogPrior(*log_priors)

    def set_mechanistic_model(self, model, output_biom_map=None):
        """
        Sets the mechanistic model of the PKPD modelling problem.

        A mechanistic model is either an instance of a
        :class:`PharmacokineticModel`, a :class:`PharmacodynamicModel`, or a
        :class:`PKPDModel`.

        The model outputs are mapped to the measured biomarkers in the dataset.
        By default the first output is mapped to the first biomarker, the
        second output to the second biomarker, and so on. The output-biomarker
        map can alternatively also be explicitly specified by a dictionary,
        with the model output names as keys, and the biomarker keys as values.

        Setting the mechanistic model, resets the error model, the population
        model, the prior probability distributions for the model parameters,
        any fixed model parameters, and parameter names.

        Parameters
        ----------

        model
            A mechanistic model of the pharmacokinetics and/or the
            pharmacodynamics in form of a :class:`PharmacokineticModel`, a
            :class:`PharmacodynamicModel`, or a :class:`PKPDModel`.
        output_biom_map
            A dictionary that maps the model outputs to the measured
            biomarkers. The keys of the dictionary identify the output names
            of the model, and the values the corresponding biomarker key in
            the dataset.
        """
        if not isinstance(
                model, (erlo.PharmacokineticModel, erlo.PharmacodynamicModel)):
            raise ValueError(
                'The model has to be an instance of a '
                'erlotinib.PharmacokineticModel, a '
                'erlotinib.PharmacodynamicModel or a erlotinib.PKPDModel')

        if output_biom_map is not None:
            try:
                outputs = list(output_biom_map.keys())
                biomarkers = list(output_biom_map.values())
            except AttributeError:
                raise ValueError(
                    'The output-biomarker map has to be a dictionary.')

            if sorted(biomarkers) != sorted(self._biom_keys):
                raise ValueError(
                    'The provided output-biomarker map does not map model '
                    'outputs to all biomarkers in the dataset.')

            # Set new model outputs
            model.set_outputs(outputs)

            # Overwrite biomarker keys to have the same order as model outputs
            self._biom_keys = biomarkers

        # Make sure that there is one model output for each biomarker
        n_biom = len(self._biom_keys)
        if model.n_outputs() < n_biom:
            raise ValueError(
                'The model does not have enough outputs to model all '
                'biomarkers in the dataset.')

        # If the model has too many outputs, return only the relevant number
        # of outputs.
        outputs = model.outputs()
        model.set_outputs(outputs[:n_biom])

        # Set mechanistic model
        self._mechanistic_model = model

        # Reset other settings that depend on the mechanistic model
        self._error_model = None
        self._population_model = None
        self._log_prior = None
        self._n_parameters = None
        self._parameter_names = None
        self._fixed_params_mask = None
        self._fixed_params_values = None

    '''
    def set_population_model(self, pop_models, params=None):
        """
        Sets the population model for each model parameter.

        A population model is an instance of a :class:`PopulationModel`. A
        population model specifies how a model parameter varies across
        individuals.

        The population models ``pop_model`` are mapped to the model parameters.
        By default the first population model is mapped to the first model
        parameter in :meth:`parameter_names`, the second population model to
        the second model parameter, and so on. By default, one population model
        has to be provided for each model parameter.

        If not all model parameters need to be modelled by a population model,
        and can vary independently between individuals, a list of parameter
        names can be specified with ``params``. The list of parameters has to
        be of the same length as the list of population models, and specifies
        the population model-model parameter map.

        Parameters
        ----------

        pop_models
            A list of :class:`PopulationModel` instances that specifies the
            variation of model parameters between individuals. By default
            the list has to be of the same length as the number of mechanistic
            and error model parameters. If ``params`` is not ``None``, the list
            of population models has to be of the same length as ``params``.
        params
            A list of model parameter names, which map the population models
            to the parameter names.
        """
        # Check inputs
        if self._mechanistic_model is None:
            raise ValueError(
                'Before setting a population model for the model parameters, '
                'a mechanistic model has to be set.')

        if self._error_model is None:
            raise ValueError(
                'Before setting a population model for the model parameters, '
                'an error model has to be set.')

        for pop_model in pop_models:


        parameter_names = self.parameter_names(exclude_pop_model=True)
    '''


class InverseProblem(object):
    """
    Represents an inference problem where a model is fit to a
    one-dimensional or multi-dimensional time series, such as measured in a
    PKPD study.

    Parameters
    ----------
    model
        An instance of a :class:`PharmacokineticModel`,
        :class:`PharmacodynamicModel`, or :class:`PKPDModel`.
    times
        A sequence of points in time. Must be non-negative and increasing.
    values
        A sequence of single- or multi-valued measurements. Must have shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of points in
        ``times`` and ``n_outputs`` is the number of outputs in the model. For
        ``n_outputs = 1``, the data can also have shape ``(n_times, )``.
    """

    def __init__(self, model, times, values):

        # Check model
        if not isinstance(model, erlo.PharmacodynamicModel):
            raise ValueError(
                'Model has to be an instance of erlotinib.Pharmacodynamic.'
            )
        self._model = model

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be increasing.')

        # Check values, copy so that they can no longer be changed
        values = np.asarray(values)
        if values.ndim == 1:
            np.expand_dims(values, axis=1)
        self._values = pints.matrix2d(values)

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)

        # Check for correct shape
        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values as a NumPy array of shape ``(n_times, n_outputs)``.
        """
        output = self._model.simulate(parameters, self._times)

        # The erlotinib.Model.simulate method returns the model output as
        # (n_outputs, n_times). We therefore need to transponse the result.
        return output.transpose()

    def evaluateS1(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        The returned data is a tuple of NumPy arrays ``(y, y')``, where ``y``
        has shape ``(n_times, n_outputs)``, while ``y'`` has shape
        ``(n_times, n_outputs, n_parameters)``.
        *This method only works for problems whose model implements the
        :class:`ForwardModelS1` interface.*
        """
        raise NotImplementedError

    def n_outputs(self):
        """
        Returns the number of outputs for this problem.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._n_parameters

    def n_times(self):
        """
        Returns the number of sampling points, i.e. the length of the vectors
        returned by :meth:`times()` and :meth:`values()`.
        """
        return self._n_times

    def times(self):
        """
        Returns this problem's times.
        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.
        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._values
