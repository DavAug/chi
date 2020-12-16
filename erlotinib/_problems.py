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

import myokit
import numpy as np
import pandas as pd
import pints

import erlotinib as erlo


class InverseProblem(object):
    """
    Represents an inference problem where a model is fit to a
    one-dimensional or multi-dimensional time series, such as measured in a
    PKPD study.

    Parameters
    ----------
    model
        An instance of a :class:`MechanisticModel`.
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
        if not isinstance(model, erlo.MechanisticModel):
            raise ValueError(
                'Model has to be an instance of a erlotinib.Model.'
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


class ProblemModellingController(object):
    """
    A controller class which simplifies the modelling process of a PKPD
    dataset.

    The class is instantiated with a PKPD dataset in form of a
    :class:`pandas.DataFrame`. This dataframe is expected to have an ID column,
    a time column, possibly several biomarker columns, and optionally a dose
    column. By default the keys for the columns are assumed to be ``ID``,
    ``Time``, and (for just one biomarker) ``Biomarker``. The dose key is by
    default ``None``, indicating that no dose information is contained in the
    dataset. If the keys in the dataset deviate from the defaults, they can be
    specified with the respective key arguments.

    The ProblemModellingController simplifies the process of generating a
    :class:`LogPosterior` for parameters of a model that describes the PKPD
    dataset. Such a model consists of a mechanistic model, an error model, and
    optionally a population model.

    Parameters
    ----------
    data
        A :class:`pandas.DataFrame` with the time series data in form of
        an ID, time, and (multiple) biomarker columns.
    id_key
        Key label of the :class:`DataFrame` which specifies the ID column.
        The ID refers to the identity of an individual. Defaults to
        ``'ID'``.
    time_key
        Key label of the :class:`DataFrame` which specifies the time
        column. Defaults to ``'Time'``.
    biom_keys
        A list of key labels of the :class:`DataFrame` which specifies the PK
        or PD biomarker columns. Defaults to ``['Biomarker']``.
    dose_key
        Key label of the :class:`DataFrame` which specifies the dose amount
        column. If ``None`` no dose is administered. Defaults to ``None``.
    """

    def __init__(
            self, data, id_key='ID', time_key='Time', biom_keys=['Biomarker'],
            dose_key=None):
        super(ProblemModellingController, self).__init__()

        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                'Data has to be pandas.DataFrame.')

        keys = [id_key, time_key] + biom_keys
        if dose_key is not None:
            keys += [dose_key]

        for key in keys:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        self._id_key, self._time_key = [id_key, time_key]
        self._biom_keys = biom_keys
        self._data = data[keys]

        # Make sure data is formatted correctly
        self._clean_data(dose_key)
        self._ids = self._data[self._id_key].unique()

        # Extract dosing regimens
        self._applied_regimens = None
        if dose_key is not None:
            self._applied_regimens = self._extract_dosing_regimens(dose_key)

        # Set defaults
        self._mechanistic_model = None
        self._log_likelihoods = None
        self._population_models = None
        self._log_prior = None
        self._n_parameters = None
        self._parameter_names = None
        self._inidividual_parameter_names = None
        self._fixed_params_mask = None
        self._fixed_params_values = None

    def _clean_data(self, dose_key):
        """
        Makes sure that the data is formated properly.

        1. ids can be converted to strings
        2. time are numerics or NaN
        3. biomarker are numerics or NaN
        4. dose are numerics or NaN
        """
        # Create container for data
        columns = [self._id_key, self._time_key] + self._biom_keys
        if dose_key is not None:
            columns += [dose_key]
        data = pd.DataFrame(columns=columns)

        # Convert IDs to strings
        data[self._id_key] = self._data[self._id_key].astype(
            "string")

        # Convert times to numerics
        data[self._time_key] = pd.to_numeric(self._data[self._time_key])

        # Convert biomarkers to numerics
        for biom_key in self._biom_keys:
            data[biom_key] = pd.to_numeric(self._data[biom_key])

        if dose_key is not None:
            data[dose_key] = pd.to_numeric(
                self._data[dose_key])

        self._data = data

    def _create_log_likelihoods(self, error_models):
        """
        Returns a dict of log-likelihoods, one for each individual in the
        dataset. The keys are the individual IDs and the values are the
        log-likelihoods.
        """
        # Create a likelihood for each individual
        log_likelihoods = dict()
        for individual in self._ids:
            # Set dosing regimen
            try:
                self._mechanistic_model._sim.set_protocol(
                    self._applied_regimens[individual])
            except TypeError:
                # TypeError is raised when applied regimens is still None,
                # i.e. no doses were defined by the datasets.
                pass

            log_likelihoods[individual] = self._create_inverse_problem(
                individual, error_models)

        return log_likelihoods

    def _create_inverse_problem(self, label, error_models):
        """
        Ties the observations to the structural model, creates an
        erlotinib.InverseProblem, and returns the corresponding log-likelihood.
        """
        # Get ouputs of the model
        outputs = self._mechanistic_model.outputs()

        # Get individuals data
        mask = self._data[self._id_key] == label
        data = self._data[mask][[self._time_key] + self._biom_keys]

        log_likelihoods = []
        for output_id, error_model in enumerate(error_models):
            # Filter times and observations for non-NaN entries
            biom_key = self._biom_keys[output_id]
            mask = data[biom_key].notnull()
            biomarker_data = data[[self._time_key, biom_key]][mask]
            mask = biomarker_data[self._time_key].notnull()
            biomarker_data = biomarker_data[mask]

            times = biomarker_data[self._time_key].to_numpy()
            biomarker = biomarker_data[biom_key].to_numpy()

            # Set model output to relevant output
            self._mechanistic_model.set_outputs([outputs[output_id]])

            # Create inverse problem
            problem = InverseProblem(self._mechanistic_model, times, biomarker)
            try:
                log_likelihoods.append(error_model(problem))
            except TypeError:
                raise ValueError(
                    'Pints.ProblemLoglikelihoods with other arguments than the'
                    ' problem are not supported.')

        # Reset model outputs to all biomarker-mapped outputs
        self._mechanistic_model.set_outputs(outputs)

        # If only one model output, return log-likelihood
        if len(log_likelihoods) == 1:
            return log_likelihoods[0]

        # Pool structural model parameters across output-likelihoods
        n_struc_params = self._mechanistic_model.n_parameters()
        n_error_params = log_likelihoods[0].n_parameters() - n_struc_params
        mask = [True] * n_struc_params + [False] * n_error_params
        try:
            log_likelihood = pints.PooledLogPDF(log_likelihoods, mask)
        except ValueError:
            raise ValueError(
                'Only structurally identical error models across outputs are '
                'supported.')

        return log_likelihood

    def _create_reduced_log_pdfs(self, log_pdfs):
        """
        Returns the log-pdfs with fixed model parameters according to the
        self._fixed_params_values and self._fixed_params_mask.

        The inputs log-pdfs is expected to be a dictionary of ID keys and
        pints.LogPDF values.
        """
        values = self._fixed_params_values[self._fixed_params_mask]
        for index, log_pdf in log_pdfs.items():
            log_pdfs[index] = erlo.ReducedLogPDF(
                log_pdf=log_pdf,
                mask=self._fixed_params_mask,
                values=values)

        return log_pdfs

    def _extract_dosing_regimens(self, dose_key):
        """
        Converts the dosing regimens defined by the pandas.DataFrame into
        myokit.Protocols, and returns them as a dictionary with individual
        IDs as keys, and regimens as values.

        For each dose entry in the dataframe a (bolus) dose event is added
        to the myokit.Protocol.
        """
        regimens = dict()
        for label in self._ids:
            # Filter times and dose events for non-NaN entries
            mask = self._data[self._id_key] == label
            data = self._data[[self._time_key, dose_key]][mask]
            mask = data[dose_key].notnull()
            data = data[mask]
            mask = data[self._time_key].notnull()
            data = data[mask]

            # Add dose events to dosing regimen
            regimen = myokit.Protocol()
            for _, row in data.iterrows():
                duration = 0.01  # Only support bolus at this point
                dose_rate = row[dose_key] / duration
                time = row[self._time_key]
                regimen.add(myokit.ProtocolEvent(dose_rate, time, duration))

            regimens[label] = regimen

        return regimens

    def _set_population_model_parameter_names(self):
        """
        Sets the parameter names of the (complete) population model.

        Parameters are named as
            ID: <ID> <mechanistic-error model param name>,
                if n_bottom_parameter = nids
            <top_parameter_name> <mechanistic-error model param name>
                for any top parameter
        """
        # Get the number of individuals
        n_ids = len(self._ids)

        # Construct parameter names
        parameter_names = []
        for pop_model in self._population_models:
            # Get mechanistic/error model parameter name
            name = pop_model.get_bottom_parameter_name()

            # Create names for individual parameters
            if pop_model.n_bottom_parameters() == n_ids:
                names = ['ID: %s %s' % (n, name) for n in self._ids]
                parameter_names += names

            # Create names for population-level parameters
            if pop_model.n_top_parameters() > 0:
                top_names = pop_model.get_top_parameter_names()
                names = [
                    '%s %s' % (pop_prefix, name) for pop_prefix in top_names]
                parameter_names += names

        # Save parameter names
        self._parameter_names = parameter_names

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

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

        if self._log_likelihoods is None:
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

    def get_dosing_regimens(self):
        """
        Returns a dictionary of dosing regimens in form of myokit.Protocols.

        The dosing regimens are extracted from the dataset if a dose key is
        provided. If no dose key is provided ``None`` is returned.
        """
        return self._applied_regimens

    def get_log_posteriors(self):
        """
        Returns a list of :class:`LogPosterior` instances, defined by
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

        if self._log_likelihoods is None:
            raise ValueError(
                'The error model has not been set.')
        log_likelihoods = self._log_likelihoods

        if self._log_prior is None:
            raise ValueError(
                'The log-prior has not been set.')

        if self._population_modelsis not None:
            # TODO:
            # Compose HierarchicalLogLikelihood

            # Overwrites the log-likelihoods
            raise NotImplementedError

        if self._fixed_params_values is not None:
            log_likelihoods = self._create_reduced_log_pdfs(log_likelihoods)

        # Compose the log-posteriors
        log_posteriors = []
        for index, log_likelihood in log_likelihoods.items():
            # Create log-posterior
            log_posterior = erlo.LogPosterior(log_likelihood, self._log_prior)

            # Tag posterior and name parameters
            log_posterior.set_id(index)
            log_posterior.set_parameter_names(self.get_parameter_names())

            log_posteriors.append(log_posterior)

        return log_posteriors

    def get_parameter_names(self, exclude_pop_model=False):
        """
        Returns the names of the free structural model parameters, i.e. the
        free parameters of the mechanistic model, the error model and
        optionally the population model.

        Any parameters that have been fixed to a constant value will not be
        included in the list of model parameters.

        If the mechanistic model or the error model have not been set, ``None``
        is returned. If the population model has not been set, only the names
        of parameters for one structural model is returned, as the models are
        structurally the same across individuals.

        If a population model has been set and not all parameters are pooled,
        the mechanistic and error model parameters appear multiple times (once
        for each individual). To get the mechanistic and error model parameters
        prior to setting a population model the ``exlude_pop_model`` flag can
        be set to ``True``.

        Parameters
        ----------
        exclude_pop_model
            A boolean flag which determines whether the parameter names of the
            full model are returned, or just the mechanistic and error model
            parameters prior to setting a population model.
        """
        # If `True`, return the parameter names of an individual model
        if exclude_pop_model and (self._population_models is not None):
            return self._individual_parameter_names

        if self._fixed_params_mask is None:
            return self._parameter_names

        # Remove fixed parameters
        parameter_names = np.array(self._parameter_names)[
            ~self._fixed_params_mask]

        return list(parameter_names)

    def n_parameters(self, exclude_pop_model=False):
        """
        Returns the number of free parameters of the structural model, i.e. the
        mechanistic model, the error model and, if set, the population model.

        Any parameters that have been fixed to a constant value will not be
        included in the number of model parameters.

        If the mechanistic model or the error model have not been set, ``None``
        is returned. If the population model has not been set, only the number
        of parameters for one structural model is returned, as the models are
        structurally the same across individuals.

        If a population model has been set and not all parameters are pooled,
        the mechanistic and error model parameters are counted multiple times
        (once for each individual). To get the number of mechanistic and error
        model parameters prior to setting a population model the
        ``exlude_pop_model`` flag can be set to ``True``.

        Parameters
        ----------
        exclude_pop_model
            A boolean flag which determines whether the number of parameters
            of the full model are returned, or just number of parameters of the
            mechanistic and error model, prior to setting an error model.
        """
        if exclude_pop_model and (self._population_models is not None):
            return len(self._individual_parameter_names)

        # Return all free model parameters (including population model, if set)
        if self._fixed_params_mask is None:
            return self._n_parameters

        # Subtract the number of fixed parameters
        n_fixed_params = int(np.sum(self._fixed_params_mask))

        return self._n_parameters - n_fixed_params

    def set_error_model(self, error_models, outputs=None):
        """
        Sets the error model for each observed biomarker-model output pair.

        An error model is a subclass of a
        :class:`pints.ProblemLogLikelihood`. The error models capture the
        deviations of the measured biomarkers from the outputs of the
        mechanistic model in form of a probabilistic model.

        Setting the error model, resets the population model, the prior
        probability distributions for the model parameters, any fixed model
        parameters, and parameter names.

        Parameters
        ----------

        error_models
            A list of :class:`pints.ProblemLikelihood` subclasses which
            specify the error model for each measured biomarker-model output
            pair.
        outputs
            A list of the model outputs, which maps the error models to the
            model outputs. By default the error models are assumed to be
            listed in the same order as the model outputs.
        """
        if self._mechanistic_model is None:
            raise ValueError(
                'Before setting an error model for the mechanistic model '
                'outputs, a mechanistic model has to be set.')

        for error_model in error_models:
            if not issubclass(error_model, pints.ProblemLogLikelihood):
                raise ValueError(
                    'The log-likelihoods are not subclasses of '
                    'pints.ProblemLogLikelihood.')

        if len(error_models) != self._mechanistic_model.n_outputs():
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
                ordered.append(error_models[index])

            error_models = ordered

        # Create one log-likelihood for each individual
        # (likelihoods are identical except for the data)
        self._log_likelihoods = self._create_log_likelihoods(error_models)

        # Update number of parameters and names for each likelihood
        log_likelihood = next(iter(self._log_likelihoods.values()))
        self._n_parameters = log_likelihood.n_parameters()
        n_noise_parameters = self._n_parameters \
            - self._mechanistic_model.n_parameters()
        self._parameter_names = self._mechanistic_model.parameters() + [
            'Noise param %d' % (n+1) for n in range(n_noise_parameters)]

        # Reset other settings that depend on the error model
        self._population_models = None
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
        :meth:`get_parameter_names`. Alternatively, the mapping of the
        log-priors can be specified explicitly with ``param_names``.

        Parameters
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

        if self._log_likelihoods is None:
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
            if sorted(list(param_names)) != sorted(self.get_parameter_names()):
                raise ValueError(
                    'The specified parameter names do not match the model '
                    'parameter names.')

            # Sort log-priors according to parameter names
            ordered = []
            names = self.get_parameter_names()
            for name in names:
                index = param_names.index(name)
                ordered.append(log_priors[index])

            log_priors = ordered

        self._log_prior = pints.ComposedLogPrior(*log_priors)

    def set_mechanistic_model(self, model, output_biom_map=None):
        """
        Sets the mechanistic model of the PKPD modelling problem.

        A mechanistic model is an instance of a
        :class:`MechanisticModel`.

        The model outputs are mapped to the measured biomarkers in the dataset.
        By default the first output is mapped to the first biomarker, the
        second output to the second biomarker, and so on. The output-biomarker
        map can alternatively also be explicitly specified by a dictionary,
        with the model output names as keys, and the dataframe's biomarker keys
        as values.

        Setting the mechanistic model, resets the error model, the population
        model, the prior probability distributions for the model parameters,
        any fixed model parameters, and parameter names.

        Parameters
        ----------

        model
            A mechanistic model of the pharmacokinetics and/or the
            pharmacodynamics in form of an instance of a
            :class:`MechanisticModel`.
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
                    'The provided output-biomarker map does not assign '
                    'a model output to each biomarker in the dataset.')

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
        self._log_likelihoods = None
        self._population_models = None
        self._log_prior = None
        self._n_parameters = None
        self._parameter_names = None
        self._fixed_params_mask = None
        self._fixed_params_values = None

    def set_population_model(self, pop_models, params=None):
        """
        Sets the population model for each model parameter.

        A population model is a :class:`PopulationModel` class. A
        population model specifies how a model parameter varies across
        individuals.

        The population models ``pop_models`` are mapped to the model
        parameters. By default the first population model is mapped to the
        first model parameter in :meth:`get_parameter_names`, the second
        population model to the second model parameter, and so on. One
        population model has to be provided for each model parameter.

        If not all model parameters need to be modelled by a population model,
        and can vary independently between individuals, a list of parameter
        names can be specified with ``params``. The list of parameters has to
        be of the same length as the list of population models, and specifies
        the population model-model parameter map.

        Parameters
        ----------
        pop_models
            A list of :class:`PopulationModel` classes that specifies the
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

        if self._log_likelihoods is None:
            raise ValueError(
                'Before setting a population model for the model parameters, '
                'an error model has to be set.')

        for pop_model in pop_models:
            if not issubclass(pop_model, erlo.PopulationModel):
                raise ValueError(
                    'The provided population models have to be '
                    'erlotinib.PopulationModel classes.')

        # Get free individual parameter names
        parameter_names = self._parameter_names
        if self._fixed_params_mask is not None:
            parameter_names = list(np.array(parameter_names)[
                ~self._fixed_params_mask])
        if self._population_models is not None:
            # If population model has been set before, parameter names are
            # not correct
            parameter_names = self._individual_parameter_names

        # Sort inputs according to `params` and fill blanks
        n_individual_parameters = len(parameter_names)
        if params is not None:
            # Create default population model container
            default_pop_models = [
                erlo.HeterogeneousModel] * n_individual_parameters

            # Substitute population models for provided parameter names
            for param_id, name in enumerate(params):
                try:
                    index = parameter_names.index(name)
                except ValueError:
                    raise ValueError(
                        'The provided parameter names could not be identified '
                        'in the model')
                default_pop_models[index] = pop_models[index]

            pop_models = default_pop_models

        # Make sure that each parameter is assigned to a population model
        if len(pop_models) != n_individual_parameters:
            raise ValueError(
                'If no parameter names are specified, one population model has'
                ' to be provided for each free parameter.')

        # Fix individual parameters permanently, and reset mask again so
        # fix_parameters may be used for the hierarchical model
        if self._fixed_params_mask is not None:
            # Fix parameters
            self._log_likelihoods = self._create_reduced_log_pdfs(
                self._log_likelihoods)

            # Reset mask
            self._fixed_params_values = None
            self._fixed_params_mask = None

        # Save individual parameter names
        self._individual_parameter_names = parameter_names

        # Instantiate population models and set parameter names
        n_ids = len(self._log_likelihoods)
        for model_id, pop_model in enumerate(pop_models):
            # Instantiate population model
            model = pop_model(n_ids)

            # Set name of modelled parameter
            name = parameter_names[model_id]
            model.set_bottom_parameter_name(name)

            # Save model in the list
            pop_models[model_id] = model

        self._population_models = pop_models

        # Update parameter names
        self._set_population_model_parameter_names()
