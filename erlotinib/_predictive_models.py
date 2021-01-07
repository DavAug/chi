#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pandas as pd
import pints

import erlotinib as erlo


class DataDrivenPredictiveModel(object):
    """
    A base class for predictive models whose parameters are either drawn from
    distribution, or are set by a :class:`pandas.DataFrame`.
    """

    def __init__(self, predictive_model):
        super(DataDrivenPredictiveModel, self).__init__()

        # Check inputs
        if not isinstance(predictive_model, erlo.PredictiveModel):
            raise ValueError(
                'The provided predictive model has to be an instance of a '
                'erlotinib.PredictiveModel.')

        self._predictive_model = predictive_model

    def get_dosing_regimen(self, final_time=None):
        """
        Returns the dosing regimen of the compound in form of a
        :class:`pandas.DataFrame`.

        The dataframe has a time, a duration, and a dose column, which indicate
        the time point and duration of the dose administration in the time
        units of the mechanistic model, :meth:`MechanisticModel.time_unit`. The
        dose column specifies the amount of the compound that is being
        administered in units of the drug amount variable of the mechanistic
        model.

        If a dosing regimen is set which is administered indefinitely, i.e. a
        finite duration and undefined number of doses, see
        :meth:`set_dosing_regimen`, only the first administration of the
        dose will appear in the dataframe. Alternatively, a final dose time
        ``final_time`` can be provided, up to which the dose events are
        registered.

        If no dosing regimen has been set, ``None`` is returned.

        Parameters
        ----------
        final_time
            Time up to which dose events are registered in the dataframe. If
            ``None``, all dose events are registered, except for indefinite
            dosing regimens. Here, only the first dose event is registered.
        """
        return self._predictive_model.get_dosing_regimen(final_time)

    def get_n_outputs(self):
        """
        Returns the number of outputs.
        """
        return self._predictive_model.get_n_outputs()

    def get_output_names(self):
        """
        Returns the output names.
        """
        return self._predictive_model.get_output_names()

    def get_submodels(self):
        """
        Returns the submodels of the predictive model.
        """
        return self._predictive_model.get_submodels()

    def sample(
            self, times, n_samples=None, seed=None, include_regimen=False):
        """
        Samples "measurements" of the biomarkers from the predictive model and
        returns them in form of a :class:`pandas.DataFrame`.
        """
        raise NotImplementedError

    def set_dosing_regimen(
            self, dose, start, duration=0.01, period=None, num=None):
        """
        Sets the dosing regimen with which the compound is administered.

        By default the dose is administered as a bolus injection (duration on
        a time scale that is 100 fold smaller than the basic time unit). To
        model an infusion of the dose over a longer time period, the
        ``duration`` can be adjusted to the appropriate time scale.

        By default the dose is administered once. To apply multiple doses
        provide a dose administration period.

        .. note::
            This method requires a :class:`MechanisticModel` that supports
            compound administration.

        Parameters
        ----------
        dose
            The amount of the compound that is injected at each administration.
        start
            Start time of the treatment.
        duration
            Duration of dose administration. For a bolus injection, a dose
            duration of 1% of the time unit should suffice. By default the
            duration is set to 0.01 (bolus).
        period
            Periodicity at which doses are administered. If ``None`` the dose
            is administered only once.
        num
            Number of administered doses. If ``None`` and the periodicity of
            the administration is not ``None``, doses are administered
            indefinitely.
        """
        self._predictive_model.set_dosing_regimen(
            dose, start, duration, period, num)


class PosteriorPredictiveModel(DataDrivenPredictiveModel):
    """
    Implements a model that predicts the change of observable biomarkers over
    time based on the inferred posterior distribution of the model parameters.

    A posterior predictive model may be used to check whether the inference
    results agree with observed measurement. A posterior predictive model may
    also be used to predict future measurements of preclinical or clinical
    biomarkers.

    A PosteriorPredictiveModel is instantiated with an instance of a
    :class:`PredictiveModel` and a :class:`pandas.DataFrame` of parameter
    samples generated e.g. with the :class:`SamplingController`. The samples
    approximate the posterior distribution of the model parameters. The
    posterior distribution has to be of the same parametric dimension as the
    predictive model. Future biomarker "measurements" can then be predicted by
    first sampling parameter values from the posterior distribution, and then
    generating "virtual" measurements from the predictive model with those
    parameters.

    Extends :class:`DataDrivenPredictiveModel`.

    Parameters
    ----------
    predictive_model
        An instance of a :class:`PredictiveModel`.
    posterior_samples
        A :class:`pandas.DataFrame` with samples from the posterior
        distribution of the model parameters. The posterior distirbution has
        to be of the same dimension as the number of predictive model
        parameters.
    warm_up_iter
        Number of warm up iterations which are excluded from the approximate
        posterior distribution.
    individual
            The ID of the modelled individual. This argument is used to
            determine the relevant samples in the dataframe. If ``None``, the
            first ID in the ID column is selected.
    param_map
        A dictionary which can be used to map predictive model parameter
        names to the parameter names in the parameter column of the
        posterior dataframe. If ``None``, it is assumed that the naming is
        identical.
    id_key
        Key label of the :class:`DataFrame` which specifies the ID column.
        The ID refers to the identity of an individual. Defaults to
        ``'ID'``.
    param_key
        Key label of the :class:`DataFrame` which specifies the parameter
        name column. Defaults to ``'Parameter'``.
    sample_key
        Key label of the :class:`DataFrame` which specifies the parameter
        sample column. Defaults to ``'Sample'``.
    iter_key
        Key label of the :class:`DataFrame` which specifies the iteration
        column. The iteration refers to the iteration of the sampling
        routine at which the parameter value was sampled. Defaults to
        ``'Iteration'``.
    run_key
        Key label of the :class:`DataFrame` which specifies the run ID
        column. The run ID refers to the run of the sampling
        routine at which the parameter value was sampled. Defaults to
        ``'Run'``.
    """

    def __init__(
            self, predictive_model, posterior_samples, warm_up_iter=None,
            individual=None, param_map=None, id_key='ID',
            param_key='Parameter', sample_key='Sample',
            iter_key='Iteration', run_key='Run'):
        super(PosteriorPredictiveModel, self).__init__(predictive_model)

        # Check input
        if not isinstance(posterior_samples, pd.DataFrame):
            raise TypeError(
                'The posterior samples have to be in a pandas.DataFrame '
                'format.')

        keys = [id_key, sample_key, param_key, iter_key, run_key]
        for key in keys:
            if key not in posterior_samples.keys():
                raise ValueError(
                    'The posterior samples dataframe does not have the key '
                    '<' + str(key) + '>.')

        # Set default parameter map (no mapping)
        if param_map is None:
            param_map = {}

        try:
            param_map = dict(param_map)
        except (TypeError, ValueError):
            raise ValueError(
                'The parameter map has to be convertable to a python '
                'dictionary.')

        # Get default individual
        ids = list(posterior_samples[id_key].dropna().unique())
        if individual is None:
            individual = ids[0]

        if individual not in ids:
            raise ValueError(
                'The individual <' + str(individual) + '> could not be found '
                'in the ID column.')

        # Mask samples for individual
        mask = posterior_samples[id_key] == individual
        posterior_samples = posterior_samples[mask]

        # Get warm-up iterations
        if warm_up_iter is None:
            warm_up_iter = 0
        warm_up_iter = int(warm_up_iter)

        if warm_up_iter < 0:
            raise ValueError(
                'The number of warm-up iterations has to be greater or equal '
                'to zero.')

        if warm_up_iter >= posterior_samples[iter_key].max():
            raise ValueError(
                'The number of warm-up iterations has to be smaller than '
                'the total number of iterations for each run.')

        # Exclude warm up iterations
        mask = posterior_samples[iter_key] > warm_up_iter
        posterior_samples = posterior_samples[mask]

        # Check that posterior has the same dimensionality as the model
        parameter_names = posterior_samples[param_key].unique()
        n_parameters = len(parameter_names)
        if predictive_model.n_parameters() != n_parameters:
            raise ValueError(
                'The dimension of the posterior distribution has to be the '
                'same as the number of parameters of the predictive model.')

        # Create map from posterior parameter names to model parameter names
        parameter_names = self._predictive_model.get_parameter_names()
        for param_id, name in enumerate(parameter_names):
            try:
                parameter_names[param_id] = param_map[name]
            except KeyError:
                # The name is not mapped
                pass

        # Make sure that all parameter names can be found in the dataframe
        df_parameter_names = posterior_samples[param_key].unique()
        for name in parameter_names:
            if name not in df_parameter_names:
                raise ValueError(
                    'The parameter <' + str(name) + '> could not be found in '
                    'the parameter column of the posterior dataframe.')

        # Transform dataframe into more convenient format for sampling
        keys = keys[1:]
        self._format_posterior_samples(
            posterior_samples, parameter_names, keys)

    def _format_posterior_samples(self, posterior, parameter_names, keys):
        """
        Transforms the dataframe of samples into a numpy array of shape
        (n_samples, n_parameters).

        This will increase the efficiency of sampling from the posterior.
        """
        # Unpack keys
        sample_key, param_key, iter_key, run_key = keys

        # Get number of samples and number of parameters
        n_iters = len(posterior[iter_key].unique())
        n_runs = len(posterior[run_key].unique())
        n_samples = n_runs * n_iters
        n_parameters = len(posterior[param_key].unique())

        # Create numpy container for samples
        container = np.empty(shape=(n_samples, n_parameters))

        # Fill container with samples
        for run_id, run in enumerate(posterior[run_key].unique()):
            # Mask samples for run
            mask = posterior[run_key] == run
            temp_df = posterior[mask][[sample_key, param_key, iter_key]]

            # Get container sample range for this run's samples
            start = n_iters * run_id
            end = start + n_iters

            # Fill container with parameter samples
            for param_id, name in enumerate(parameter_names):
                # Get parameter samples
                mask = temp_df[param_key] == name
                samples_df = temp_df[mask]

                # Make sure samples are sorted according to iterations
                samples = samples_df.sort_values(iter_key)[sample_key]

                # Add samples to container
                container[start:end, param_id] = samples.to_numpy()

        # Remember reformated samples
        self._posterior = container

    def sample(
            self, times, n_samples=None, seed=None, include_regimen=False):
        """
        Samples "measurements" of the biomarkers from the posterior predictive
        model and returns them in form of a :class:`pandas.DataFrame`.

        For each of the ``n_samples`` a parameter set is drawn from the
        approximate posterior distribution. These paramaters are then used to
        sample from the predictive model.

        Parameters
        ----------
        times
            An array-like object with times at which the virtual "measurements"
            are performed.
        n_samples
            The number of virtual "measurements" that are performed at each
            time point. If ``None`` the biomarkers are measured only once
            at each time point.
        seed
            A seed for the pseudo-random number generator.
        include_regimen
            A boolean flag which determines whether the information about the
            dosing regimen is included.
        """
        # Make sure n_samples is an integer
        if n_samples is None:
            n_samples = 1
        n_samples = int(n_samples)

        # Sort times
        times = np.sort(times)

        # Create container for samples
        container = pd.DataFrame(
            columns=['ID', 'Biomarker', 'Time', 'Sample'])

        # Get model outputs (biomarkers)
        outputs = self._predictive_model.get_output_names()

        # Instantiate random number generator for sampling from the posterior
        rng = np.random.default_rng(seed=seed)

        # Draw samples
        sample_ids = np.arange(start=1, stop=n_samples+1)
        for sample_id in sample_ids:
            # Sample parameter from posterior
            parameters = rng.choice(self._posterior)

            # Increment seed for each iteration, to avoid repeated patterns
            if seed is not None:
                seed += 1

            # Sample from predictive model
            sample = self._predictive_model.sample(
                parameters, times, n_samples, seed, return_df=False)

            # Append samples to dataframe
            for output_id, name in enumerate(outputs):
                container = container.append(pd.DataFrame({
                    'ID': sample_id,
                    'Biomarker': name,
                    'Time': times,
                    'Sample': sample[output_id, :, 0]}))

        # Add dosing regimen, if set
        final_time = np.max(times)
        regimen = self.get_dosing_regimen(final_time)
        if (regimen is not None) and (include_regimen is True):
            # Append dosing regimen only once for all samples
            container = container.append(regimen)

        return container


class PredictiveModel(object):
    """
    Implements a model that predicts the change of observable biomarkers over
    time.

    This model takes an instance of a :class:`MechanisticModel` and an instance
    of an :class:`ErrorModel` for each mechanistic model output, and predicts
    biomarker values that may be measured in preclinical or clinical
    experiments.

    Parameters
    ----------
    mechanistic_model
        An instance of a :class:`MechanisticModel`.
    error_models
        A list of :class:`ErrorModel` instances, one for each model output of
        the mechanistic model.
    outputs
        A list of the model outputs, which maps the error models to the model
        outputs. If ``None`` the error models are assumed to be listed in the
        same order as the model outputs.
    """

    def __init__(self, mechanistic_model, error_models, outputs=None):
        super(PredictiveModel, self).__init__()

        # Check inputs
        if not isinstance(mechanistic_model, erlo.MechanisticModel):
            raise ValueError(
                'The provided mechanistic model has to be an instance of a '
                'erlotinib.MechanisticModel.')

        for error_model in error_models:
            if not isinstance(error_model, erlo.ErrorModel):
                raise ValueError(
                    'All provided error models have to be instances of a '
                    'erlo.ErrorModel.')

        # Set ouputs
        if outputs is not None:
            mechanistic_model.set_outputs(outputs)

        # Get number of outputs
        n_outputs = mechanistic_model.n_outputs()

        if len(error_models) != n_outputs:
            raise ValueError(
                'Wrong number of error models. One error model has to be '
                'provided for each mechanistic error model.')

        # Rename error model parameters, if more that one output
        if n_outputs > 1:
            # Get output names
            outputs = mechanistic_model.outputs()

            for output_id, error_model in enumerate(error_models):
                # Get original parameter names
                names = error_model.get_parameter_names()

                # Prepend output name
                output = outputs[output_id]
                names = [output + ' ' + name for name in names]

                # Set new parameter names
                error_model.set_parameter_names(names)

        # Remember models
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models

        # Set parameter names and number of parameters
        self._set_number_and_parameter_names()

    def _set_number_and_parameter_names(self):
        """
        Sets the number and names of the free model parameters.
        """
        # Get mechanistic model parameters
        parameter_names = self._mechanistic_model.parameters()

        # Get error model parameters
        for error_model in self._error_models:
            parameter_names += error_model.get_parameter_names()

        # Update number and names
        self._parameter_names = parameter_names
        self._n_parameters = len(self._parameter_names)

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

    def get_dosing_regimen(self, final_time=None):
        """
        Returns the dosing regimen of the compound in form of a
        :class:`pandas.DataFrame`.

        The dataframe has a time, a duration, and a dose column, which indicate
        the time point and duration of the dose administration in the time
        units of the mechanistic model, :meth:`MechanisticModel.time_unit`. The
        dose column specifies the amount of the compound that is being
        administered in units of the drug amount variable of the mechanistic
        model.

        If a dosing regimen is set which is administered indefinitely, i.e. a
        finite duration and undefined number of doses, see
        :meth:`set_dosing_regimen`, only the first administration of the
        dose will appear in the dataframe. Alternatively, a final dose time
        ``final_time`` can be provided, up to which the dose events are
        registered.

        If no dosing regimen has been set, ``None`` is returned.

        Parameters
        ----------
        final_time
            Time up to which dose events are registered in the dataframe. If
            ``None``, all dose events are registered, except for indefinite
            dosing regimens. Here, only the first dose event is registered.
        """
        # Get regimen
        try:
            regimen = self._mechanistic_model.dosing_regimen()
        except AttributeError:
            # The model does not support dosing regimens
            return None

        # Return None if regimen is not set
        if regimen is None:
            return regimen

        # Make sure that final_time is positive
        if final_time is None:
            final_time = np.inf

        # Sort regimen into dataframe
        regimen_df = pd.DataFrame(columns=['Time', 'Duration', 'Dose'])
        for dose_event in regimen.events():
            # Get dose amount
            dose_rate = dose_event.level()
            dose_duration = dose_event.duration()
            dose_amount = dose_rate * dose_duration

            # Get dosing time points
            start_time = dose_event.start()
            period = dose_event.period()
            n_doses = dose_event.multiplier()

            if start_time > final_time:
                # Dose event exceeds final dose time and is therefore
                # not registered
                continue

            if period == 0:
                # Dose is administered only once
                regimen_df = regimen_df.append(pd.DataFrame({
                    'Time': [start_time],
                    'Duration': [dose_duration],
                    'Dose': [dose_amount]}))

                # Continue to next dose event
                continue

            if n_doses == 0:
                # The dose event would be administered indefinitely, so we
                # stop with final_time or 1.
                n_doses = 1

                if np.isfinite(final_time):
                    n_doses = int(abs(final_time) // period)

            # Construct dose times
            dose_times = [start_time + n * period for n in range(n_doses)]

            # Make sure that even for finite periodic dose events the final
            # time is not exceeded
            dose_times = np.array(dose_times)
            mask = dose_times <= final_time
            dose_times = dose_times[mask]

            # Add dose administrations to dataframe
            regimen_df = regimen_df.append(pd.DataFrame({
                'Time': dose_times,
                'Duration': dose_duration,
                'Dose': dose_amount}))

        # If no dose event before final_time exist, return None
        if regimen_df.empty:
            return None

        return regimen_df

    def get_n_outputs(self):
        """
        Returns the number of outputs.
        """
        return self._mechanistic_model.n_outputs()

    def get_output_names(self):
        """
        Returns the output names.
        """
        return self._mechanistic_model.outputs()

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
        Returns the number of parameters of the predictive model.
        """
        return self._n_parameters

    def sample(
            self, parameters, times, n_samples=None, seed=None,
            return_df=True, include_regimen=False):
        """
        Samples "measurements" of the biomarkers from the predictive model and
        returns them in form of a :class:`pandas.DataFrame` or a
        :class:`numpy.ndarray`.

        The mechanistic model is solved for the provided parameters and times,
        and samples around this solution are drawn from the error models for
        each time point.

        The number of samples for each time point can be specified with
        ``n_samples``.

        Parameters
        ----------
        parameters
            An array-like object with the parameter values of the predictive
            model.
        times
            An array-like object with times at which the virtual "measurements"
            are performed.
        n_samples
            The number of virtual "measurements" that are performed at each
            time point. If ``None`` the biomarkers are measured only once
            at each time point.
        seed
            A seed for the pseudo-random number generator.
        return_df
            A boolean flag which determines whether the output is returned as a
            :class:`pandas.DataFrame` or a :class:`numpy.ndarray`. If ``False``
            the samples are returned as a numpy array of shape
            ``(n_outputs, n_times, n_samples)``.
        include_regimen
            A boolean flag which determines whether the dosing regimen
            information is included in the output. If the samples are returned
            as a :class:`numpy.ndarray`, the dosing information is not
            included.
        """
        parameters = np.asarray(parameters)
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The length of parameters does not match n_parameters.')

        # Sort times
        times = np.sort(times)

        # Sort parameters into mechanistic model params and error params
        n_parameters = self._mechanistic_model.n_parameters()
        mechanistic_params = parameters[:n_parameters]
        error_params = parameters[n_parameters:]

        # Solve mechanistic model
        outputs = self._mechanistic_model.simulate(mechanistic_params, times)

        # Create numpy container for samples
        n_outputs = len(outputs)
        n_times = len(times)
        n_samples = n_samples if n_samples is not None else 1
        container = np.empty(shape=(n_outputs, n_times, n_samples))

        # Sample error around mechanistic model outputs
        start_index = 0
        for output_id, error_model in enumerate(self._error_models):
            end_index = start_index + error_model.n_parameters()

            # Sample
            container[output_id, ...] = error_model.sample(
                parameters=error_params[start_index:end_index],
                model_output=outputs[output_id],
                n_samples=n_samples,
                seed=seed)

            # Update start index
            start_index = end_index

        if return_df is False:
            # Return samples in numpy array format
            return container

        # Structure samples in a pandas.DataFrame
        output_names = self._mechanistic_model.outputs()
        sample_ids = np.arange(start=1, stop=n_samples+1)
        samples = pd.DataFrame(
            columns=['ID', 'Biomarker', 'Time', 'Sample'])

        # Fill in all samples at a specific time point at once
        for output_id, name in enumerate(output_names):
            for time_id, time in enumerate(times):
                samples = samples.append(pd.DataFrame({
                    'ID': sample_ids,
                    'Time': time,
                    'Biomarker': name,
                    'Sample': container[output_id, time_id, :]}))

        # Add dosing regimen information, if set
        final_time = np.max(times)
        regimen = self.get_dosing_regimen(final_time)
        if (regimen is not None) and (include_regimen is True):
            # Add dosing regimen for each sample
            for _id in sample_ids:
                regimen['ID'] = _id
                samples = samples.append(regimen)

        return samples

    def set_dosing_regimen(
            self, dose, start, duration=0.01, period=None, num=None):
        """
        Sets the dosing regimen with which the compound is administered.

        By default the dose is administered as a bolus injection (duration on
        a time scale that is 100 fold smaller than the basic time unit). To
        model an infusion of the dose over a longer time period, the
        ``duration`` can be adjusted to the appropriate time scale.

        By default the dose is administered once. To apply multiple doses
        provide a dose administration period.

        .. note::
            This method requires a :class:`MechanisticModel` that supports
            compound administration.

        Parameters
        ----------
        dose
            The amount of the compound that is injected at each administration.
        start
            Start time of the treatment.
        duration
            Duration of dose administration. For a bolus injection, a dose
            duration of 1% of the time unit should suffice. By default the
            duration is set to 0.01 (bolus).
        period
            Periodicity at which doses are administered. If ``None`` the dose
            is administered only once.
        num
            Number of administered doses. If ``None`` and the periodicity of
            the administration is not ``None``, doses are administered
            indefinitely.
        """
        try:
            self._mechanistic_model.set_dosing_regimen(
                dose, start, duration, period, num)
        except AttributeError:
            # This error means that the mechanistic model is a
            # PharmacodynamicModel and therefore no dosing regimen can be set.
            raise AttributeError(
                'The mechanistic model does not support to set dosing '
                'regimens. This may be because the underlying '
                'erlotinib.MechanisticModel is a '
                'erlotinib.PharmacodynamicModel.')


class PriorPredictiveModel(DataDrivenPredictiveModel):
    """
    Implements a model that predicts the change of observable biomarkers over
    time based on the provided distribution of model parameters prior to the
    inference.

    A prior predictive model may be used to check whether the assumptions about
    the parameter distribution ``log_prior`` lead to a predictive distirbution
    that encapsulates the expected measurement values of preclinical and
    clinical biomarkers.

    A PriorPredictiveModel is instantiated with an instance of a
    :class:`PredictiveModel` and a :class:`pints.LogPrior` of the same
    parametric dimension as the predictive model. Future biomarker
    "measurements" can then be predicted by first sampling parameter values
    from the log-prior distribution, and then generating "virtual" measurements
    from the predictive model with those parameters.

    Extends :class:`DataDrivenPredictiveModel`.

    Parameters
    ----------
    predictive_model
        An instance of a :class:`PredictiveModel`.
    log_prior
        An instance of a :class:`pints.LogPrior` of the same dimensionality as
        the number of predictive model parameters.
    """

    def __init__(self, predictive_model, log_prior):
        super(PriorPredictiveModel, self).__init__(predictive_model)

        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError(
                'The provided log-prior has to be an instance of a '
                'pints.LogPrior.')

        if predictive_model.n_parameters() != log_prior.n_parameters():
            raise ValueError(
                'The dimension of the log-prior has to be the same as the '
                'number of parameters of the predictive model.')

        self._log_prior = log_prior

    def sample(self, times, n_samples=None, seed=None, include_regimen=False):
        """
        Samples "measurements" of the biomarkers from the prior predictive
        model and returns them in form of a :class:`pandas.DataFrame`.

        For each of the ``n_samples`` a parameter set is drawn from the
        log-prior. These paramaters are then used to sample from the predictive
        model.

        Parameters
        ----------
        times
            An array-like object with times at which the virtual "measurements"
            are performed.
        n_samples
            The number of virtual "measurements" that are performed at each
            time point. If ``None`` the biomarkers are measured only once
            at each time point.
        seed
            A seed for the pseudo-random number generator.
        include_regimen
            A boolean flag which determines whether the information about the
            dosing regimen is included.
        """
        # Make sure n_samples is an integer
        if n_samples is None:
            n_samples = 1
        n_samples = int(n_samples)

        # Set seed for prior samples
        # (does not affect predictive model)
        if seed is not None:
            # TODO: pints.Priors are not meant to be seeded, so fails when
            # anything else but np.random is used.
            np.random.seed(seed)

            # Set predictive model base seed
            base_seed = seed

        # Sort times
        times = np.sort(times)

        # Create container for samples
        container = pd.DataFrame(
            columns=['ID', 'Biomarker', 'Time', 'Sample'])

        # Get model outputs (biomarkers)
        outputs = self._predictive_model.get_output_names()

        # Draw samples
        sample_ids = np.arange(start=1, stop=n_samples+1)
        for sample_id in sample_ids:
            # Sample parameter
            parameters = self._log_prior.sample().flatten()

            if seed is not None:
                # Set seed for predictive model to base_seed + sample_id
                # (Needs to change every iteration)
                seed = base_seed + sample_id

            # Sample from predictive model
            sample = self._predictive_model.sample(
                parameters, times, n_samples, seed, return_df=False)

            # Append samples to dataframe
            for output_id, name in enumerate(outputs):
                container = container.append(pd.DataFrame({
                    'ID': sample_id,
                    'Biomarker': name,
                    'Time': times,
                    'Sample': sample[output_id, :, 0]}))

        # Add dosing regimen, if set
        final_time = np.max(times)
        regimen = self.get_dosing_regimen(final_time)
        if (regimen is not None) and (include_regimen is True):
            # Append dosing regimen only once for all samples
            container = container.append(regimen)

        return container
