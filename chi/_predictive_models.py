#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy

import numpy as np
import pandas as pd
import pints
import xarray as xr

import chi


class GenerativeModel(object):
    """
    A base class for predictive models whose parameters are drawn from
    a generative distribution.
    """

    def __init__(self, predictive_model):
        super(GenerativeModel, self).__init__()

        # Check inputs
        if not isinstance(predictive_model, chi.PredictiveModel):
            raise ValueError(
                'The provided predictive model has to be an instance of a '
                'chi.PredictiveModel.')

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

        If an indefinitely administered dosing regimen is set, i.e. a
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

    def get_predictive_model(self):
        """
        Returns the predictive model.
        """
        return self._predictive_model

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
            dose administration.

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


class PosteriorPredictiveModel(GenerativeModel):
    r"""
    Implements a model that predicts the change of observable biomarkers over
    time based on the inferred parameter posterior distribution.

    A PosteriorPredictiveModel is instantiated with an instance of a
    :class:`PredictiveModel` and a :class:`xarray.Dataset` of parameter
    posterior samples generated e.g. with the :class:`SamplingController`. The
    samples approximate the posterior distribution of the model parameters. The
    posterior distribution has to be of the same parametric dimension as the
    predictive model. Future biomarker "measurements" can then be predicted by
    first sampling parameter values from the posterior distribution, and then
    generating "virtual" measurements from the predictive model with those
    parameters.

    Formally the posterior predictive model may be defined as

    .. math::
        p(x | t; X^{\text{obs}}) = \int \text{d}\theta \,
        p(x | t; \theta)p(\theta | X^{\text{obs}}),

    where :math:`x` are the measureable biomarker values, :math:`t` is the
    measurement time, :math:`p(x | t; \theta)` is the predictive model
    and :math:`p(\theta | X^{\text{obs}})` is the posterior distribution of
    the model parmeters :math:`\theta` for observations
    :math:`X^{\text{obs}}`.

    Extends :class:`GenerativeModel`.

    :param predictive_model: A predictive model which defines the distribution
        of observable biomarkers over time conditioned on parameter values.
    :type predictive_model: PredictiveModel
    :param posterior_samples: Samples from the posterior distribution of the
        model parameters.
    :type posterior_samples: xarray.Dataset
    :param param_map: A dictionary which can be used to map predictive model
        parameter names to the parameter names in the :class:`xarray.Dataset`.
        If ``None``, it is assumed that the names are identical.
    :type param_map: dict, optional
    """
    def __init__(
            self, predictive_model, posterior_samples, param_map=None):
        super(PosteriorPredictiveModel, self).__init__(predictive_model)

        # Check input
        if not isinstance(posterior_samples, xr.Dataset):
            raise TypeError(
                'The posterior samples have to be a xarray.Dataset.')

        dims = sorted(list(posterior_samples.dims))
        expected_dims = ['chain', 'draw', 'individual']
        if (len(dims) == 2):
            expected_dims = ['chain', 'draw']
        for dim in expected_dims:
            if dim not in dims:
                raise ValueError(
                    'The posterior samples must have the dimensions '
                    '(chain, draw, individual). The current dimensions are <'
                    + str(dims) + '>.')

        # Set default parameter map (no mapping)
        if param_map is None:
            param_map = {}

        try:
            param_map = dict(param_map)
        except (TypeError, ValueError):
            raise ValueError(
                'The parameter map has to be convertable to a python '
                'dictionary.')

        # Check that the parameters of the posterior can be identified in the
        # dataset
        self._parameter_names = self._check_parameters(
            posterior_samples, param_map)

        # Store posterior
        self._posterior = posterior_samples

    def _check_parameters(self, posterior_samples, param_map):
        """
        Checks whether the parameters of the posterior exist in the dataset
        and returns them.
        """
        # Create map from posterior parameter names to model parameter names
        model_names = self._predictive_model.get_parameter_names()
        for param_id, name in enumerate(model_names):
            try:
                model_names[param_id] = param_map[name]
            except KeyError:
                # The name is not mapped
                pass

        # Check that model names are in the dataset
        for parameter in model_names:
            if parameter not in posterior_samples.data_vars:
                raise ValueError(
                    'The parameter <' + str(parameter) + '> cannot be found '
                    'in the posterior.')

        return model_names

    def sample(
            self, times, n_samples=None, individual=None, seed=None,
            include_regimen=False):
        """
        Samples "measurements" of the biomarkers from the posterior predictive
        model and returns them in form of a :class:`pandas.DataFrame`.

        For each of the ``n_samples`` a parameter set is drawn from the
        approximate posterior distribution. These paramaters are then used to
        sample from the predictive model.

        :param times: Times for the virtual "measurements".
        :type times: list, numpy.ndarray of shape (n,)
        :param n_samples: The number of virtual "measurements" that are
            performed at each time point. If ``None`` the biomarkers are
            measured only once at each time point.
        :type n_samples: int, optional
        :param individual: The ID of the modelled individual. If
            ``None``, either the first ID or the population is simulated.
        :type individual: str, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int
        :param include_regimen: A boolean flag which determines whether the
            information about the dosing regimen is included.
        :type include_regimen: bool, optional
        """
        # Make sure n_samples is an integer
        if n_samples is None:
            n_samples = 1
        n_samples = int(n_samples)

        # Check individual for population model
        if isinstance(self._predictive_model, chi.PredictivePopulationModel):
            if individual is not None:
                raise ValueError(
                    "Individual ID's cannot be selected for a "
                    "chi.PredictivePopulationModel. To model an "
                    "individual create a chi.PosteriorPredictiveModel "
                    "with a chi.PredictiveModel.")

        # Check individual for individual predictive model
        else:
            ids = self._posterior.individual
            # Get default individual
            if individual is None:
                individual = str(ids.data[0])

            # Make sure individual exists
            if individual not in ids:
                raise ValueError(
                    'The individual <' + str(individual) + '> could not be '
                    'found in the ID column.')

        # Sort times
        times = np.sort(times)

        # Sort parameters into numpy array for simplified sampling
        n_chains = len(self._posterior.chain)
        n_parameters = self._predictive_model.n_parameters()
        try:
            n_draws = len(self._posterior.sel(
                    individual=individual).dropna(dim='draw').draw)
        # Note: ValueError -> KeyError for xarray>=0.19.0
        except (ValueError, KeyError):
            n_draws = len(self._posterior.dropna(dim='draw').draw)
        posterior = np.empty(shape=(n_chains * n_draws, n_parameters))
        for param_id, parameter in enumerate(self._parameter_names):
            try:
                posterior[:, param_id] = self._posterior[parameter].sel(
                    individual=individual).dropna(dim='draw').values.flatten()
            # Note: ValueError -> KeyError for xarray>=0.19.0
            except (ValueError, KeyError):
                # If individual dimension does not exist, the parameter must
                # be a population parameter.
                posterior[:, param_id] = self._posterior[
                    parameter].dropna(dim='draw').values.flatten()

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
            parameters = rng.choice(posterior)

            # Sample from predictive model
            sample = self._predictive_model.sample(
                parameters, times, n_samples, rng, return_df=False)

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
    r"""
    Implements a model that predicts the change of observable biomarkers over
    time in a patient or a model organism.

    This model takes an instance of a :class:`MechanisticModel` and an instance
    of an :class:`ErrorModel` for each mechanistic model output, and predicts
    biomarker values that may be measured in preclinical or clinical
    experiments.

    Formally, the distribution of measureable observables may be expressed as

    .. math::
        X \sim \mathbb{P}
        \left( f(t, \psi _{\text{m}}), \psi _{\text{e}} \right) ,

    where :math:`X` is a measureable biomarker value at time :math:`t`.
    :math:`f(t, \psi _{\text{m}})` is the output of the mechanistic model for
    model parameters :math:`\psi _{\text{m}}`, and :math:`\mathbb{P}` is the
    distribution around the mechanistic model for error model parameters
    :math:`\psi _{\text{e}}`.

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
        if not isinstance(
                mechanistic_model,
                (chi.MechanisticModel, chi.ReducedMechanisticModel)):
            raise TypeError(
                'The mechanistic model has to be an instance of a '
                'chi.MechanisticModel.')

        for error_model in error_models:
            if not isinstance(
                    error_model, (chi.ErrorModel, chi.ReducedErrorModel)):
                raise TypeError(
                    'All error models have to be instances of a '
                    'chi.ErrorModel.')

        # Copy mechanistic model
        mechanistic_model = mechanistic_model.copy()

        # Set outputs
        if outputs is not None:
            mechanistic_model.set_outputs(outputs)

        # Get number of outputs
        n_outputs = mechanistic_model.n_outputs()

        if len(error_models) != n_outputs:
            raise ValueError(
                'Wrong number of error models. One error model has to be '
                'provided for each mechanistic error model.')

        # Copy error models
        error_models = [
            copy.deepcopy(error_model) for error_model in error_models]

        # Remember models
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models

        # Set parameter names and number of parameters
        self._set_error_model_parameter_names()
        self._set_number_and_parameter_names()

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

        If an indefinitely administered dosing regimen is set, i.e. a
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
        return copy.copy(self._parameter_names)

    def get_submodels(self):
        """
        Returns the submodels of the predictive model in form of a dictionary.
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
            A seed for the pseudo-random number generator or a
            :class:`numpy.random.Generator`.
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
                'chi.MechanisticModel is a '
                'chi.PharmacodynamicModel.')


class PredictivePopulationModel(PredictiveModel):
    r"""
    Implements a model that predicts the change of observable biomarkers over
    time in a population of patients or model organisms.

    This model takes an instance of a :class:`PredictiveModel`, and one
    instance of a :class:`PopulationModel` for each predictive model
    parameter.

    Formally, the distribution of measureable observables may be expressed as

    .. math::
        X \sim
        \mathbb{P}\left( t; \psi _{\text{m}}, \psi _{\text{e}} \right)
        \mathbb{P}\left(\psi _{\text{m}}, \psi _{\text{e}} | \theta \right),

    where :math:`X` is a measureable biomarker value at time :math:`t`.
    :math:`\mathbb{P}\left( t; \psi _{\text{m}}, \psi _{\text{e}} \right)` is
    the predictive model for an individual patient defined by the predictive
    model, and
    :math:`\mathbb{P}\left(\psi _{\text{m}}, \psi _{\text{e}} | \theta \right)`
    is the population distribution with parameters :math:`\theta`.

    Extends :class:`PredictiveModel`.

    Parameters
    ----------
    predictive_model
        An instance of a :class:`PredictiveModel`.
    population_models
        A list of :class:`PopulationModel` instances, one for each predictive
        model parameter.
    params
        A list of the model parameters, which maps the population models to the
        predictive model parameters. If ``None``, the population models are
        assumed to be listed in the same order as the model parameters.
    """

    def __init__(self, predictive_model, population_models, params=None):
        # Check inputs
        if not isinstance(predictive_model, chi.PredictiveModel):
            raise TypeError(
                'The predictive model has to be an instance of a '
                'chi.PredictiveModel.')

        for pop_model in population_models:
            if not isinstance(pop_model, chi.PopulationModel):
                raise TypeError(
                    'All population models have to be instances of a '
                    'chi.PopulationModel.')

        # Get number and names of non-population predictive model
        n_parameters = predictive_model.n_parameters()
        parameter_names = predictive_model.get_parameter_names()

        # Check that there is one population model for each model parameter
        if len(population_models) != n_parameters:
            raise ValueError(
                'One population model has to be provided for each of the '
                '<' + str(n_parameters) + '> mechanistic model-error '
                'model parameters.')

        # Sort population models according to model parameters
        if params is not None:
            # Check that params has the right length, and the correct
            # parameter names
            if len(params) != n_parameters:
                raise ValueError(
                    'Params does not have the correct length. Params is '
                    'expected to contain all model parameter names.')

            if sorted(parameter_names) != sorted(params):
                raise ValueError(
                    'The parameter names in <params> have to coincide with '
                    'the parameter names of the non-populational predictive '
                    'model parameters <' + str(parameter_names) + '>.')

            # Sort population models
            pop_models = ['Place holder'] * n_parameters
            for param_id, name in enumerate(params):
                index = parameter_names.index(name)
                pop_models[index] = population_models[param_id]

            population_models = pop_models

        # Remember predictive model and population models
        self._predictive_model = predictive_model
        self._population_models = [
            copy.copy(pop_model) for pop_model in population_models]

        # Set number and names of model parameters
        self._set_population_parameter_names()
        self._set_number_and_parameter_names()

    def _set_population_parameter_names(self):
        """
        Sets the names of the population model parameters.

        For chi.HeterogeneousModel the bottom-level parameter is used
        as model parameter name.
        """
        # Get predictive model parameters
        bottom_parameter_names = self._predictive_model.get_parameter_names()

        # Construct model parameter names
        for param_id, pop_model in enumerate(self._population_models):
            # Get original population parameters
            pop_model.set_parameter_names(None)
            pop_params = pop_model.get_parameter_names()

            # Construct parameter names
            bottom_name = bottom_parameter_names[param_id]
            if pop_params is not None:
                names = [name + ' ' + bottom_name for name in pop_params]
            else:
                # If the population model has no parameter names,
                # we keep the bottom name
                names = [bottom_name]

            # Update population model parameter names
            pop_model.set_parameter_names(names)

    def _set_number_and_parameter_names(self):
        """
        Updates the number and names of the free model parameters.
        """
        # Construct model parameter names
        parameter_names = []
        for pop_model in self._population_models:
            # Get population parameter
            pop_params = pop_model.get_parameter_names()
            parameter_names += pop_params

        # Update number and names
        self._parameter_names = parameter_names
        self._n_parameters = len(self._parameter_names)

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        .. note:
            Parameters modelled by a :class:`HeterogeneousModel` cannot be
            fixed on the population level. If you would like to fix the
            associated parameter, fix it in the corresponding
            :class:`PredictiveModel`.

        Parameters
        ----------
        name_value_dict
            A dictionary with model parameter names as keys, and parameter
            values as values.
        """
        # Check type of dictionanry
        try:
            name_value_dict = dict(name_value_dict)
        except (TypeError, ValueError):
            raise ValueError(
                'The name-value dictionary has to be convertable to a python '
                'dictionary.')

        # Get population models
        pop_models = self._population_models

        # Convert models to reduced models
        for model_id, pop_model in enumerate(pop_models):
            if not isinstance(pop_model, chi.ReducedPopulationModel):
                pop_models[model_id] = chi.ReducedPopulationModel(pop_model)

        # Fix model parameters
        for pop_model in pop_models:
            pop_model.fix_parameters(name_value_dict)

        # If no parameters are fixed, get original model back
        for model_id, pop_model in enumerate(pop_models):
            if pop_model.n_fixed_parameters() == 0:
                pop_model = pop_model.get_population_model()
                pop_models[model_id] = pop_model

        # Safe reduced models
        self._population_models = pop_models

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

        If an indefinitely administered dosing regimen is set, i.e. a
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

    def get_parameter_names(self):
        """
        Returns the parameter names of the predictive model.
        """
        return copy.copy(self._parameter_names)

    def get_submodels(self):
        """
        Returns the submodels of the predictive model in form of a dictionary.
        """
        # Get submodels from predictive model
        submodels = self._predictive_model.get_submodels()

        # Get original models
        pop_models = []
        for pop_model in self._population_models:
            # Get original population model
            if isinstance(pop_model, chi.ReducedPopulationModel):
                pop_model = pop_model.get_population_model()

            pop_models.append(pop_model)

        submodels['Population models'] = pop_models

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
        Samples "measurements" of the biomarkers from virtual "patients" and
        returns them in form of a :class:`pandas.DataFrame` or a
        :class:`numpy.ndarray`.

        Virtual patients are sampled from the population models in form of
        predictive model parameters. Those parameters are then used to sample
        virtual measurements from the predictive model. For each virtual
        patient one measurement is performed at each of the provided time
        points.

        The number of virtual patients that is being measured can be specified
        with ``n_samples``.

        Parameters
        ----------
        parameters
            An array-like object with the parameter values of the predictive
            model.
        times
            An array-like object with times at which the virtual "measurements"
            are performed.
        n_samples
            The number of virtual "patients" that are measured at each
            time point. If ``None`` the biomarkers are measured only for one
            patient.
        seed
            A seed for the pseudo-random number generator or a
            :class:`numpy.random.Generator`.
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
        # Check inputs
        parameters = np.asarray(parameters)
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The length of parameters does not match n_parameters.')

        if n_samples is None:
            n_samples = 1
        n_samples = int(n_samples)

        # Sort times
        times = np.sort(times)

        # Create container for "virtual patients"
        n_parameters = self._predictive_model.n_parameters()
        patients = np.empty(shape=(n_samples, n_parameters))

        # Sample individuals from population model
        start = 0
        for param_id, pop_model in enumerate(self._population_models):
            # If heterogenous model, use input parameter for all patients
            if isinstance(pop_model, chi.HeterogeneousModel):
                patients[:, param_id] = parameters[start]

                # Increment population parameter counter and continue to next
                # iteration
                start += 1
                continue

            # Get range of relevant population parameters
            end = start + pop_model.n_parameters()

            # Sample patient parameters
            patients[:, param_id] = pop_model.sample(
                parameters=parameters[start:end], n_samples=n_samples,
                seed=seed)

            # Increment population parameter counter
            start = end

        # Create numpy container for samples (measurements of virtual patients)
        n_outputs = self._predictive_model.get_n_outputs()
        n_times = len(times)
        container = np.empty(shape=(n_outputs, n_times, n_samples))

        # Sample measurements for each patient
        for patient_id, patient in enumerate(patients):
            sample = self._predictive_model.sample(
                parameters=patient, times=times, seed=seed, return_df=False)
            container[..., patient_id] = sample[..., 0]

        if return_df is False:
            # Return samples in numpy array format
            return container

        # Structure samples in a pandas.DataFrame
        output_names = self._predictive_model.get_output_names()
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
        self._predictive_model.set_dosing_regimen(
            dose, start, duration, period, num)


class PriorPredictiveModel(GenerativeModel):
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


class StackedPredictiveModel(GenerativeModel):
    r"""
    A generative model that is defined by the weighted average of different
    posterior predictive models.

    A stacked predictive model accepts a list of
    :class:`PosteriorPredictiveModel` instances and a list of relative
    weights of the models. The resulting predictive model is then determined
    by the weighted average of the individual models

    .. math::
        p(x | x^{\mathrm{obs}}) = \sum _m w_m\, p_m(x | x^{\mathrm{obs}}),

    where the sum runs over the individual models and :math:`w_m` is the weight
    of model :math:`m`.

    Extends :class:`GenerativeModel`.
    """
    def __init__(self, predictive_models, weights):
        # Check inputs
        for predictive_model in predictive_models:
            if not isinstance(predictive_model, PosteriorPredictiveModel):
                raise TypeError(
                    'The predictive models must be instances of '
                    'chi.PosteriorPredictiveModel.')

        predictive_model = predictive_models[0].get_predictive_model()
        super(StackedPredictiveModel, self).__init__(predictive_model)

        n_outputs = self._predictive_model.get_n_outputs()
        for predictive_model in predictive_models:
            if n_outputs != predictive_model.get_n_outputs():
                raise ValueError(
                    'All predictive models must have the same number of '
                    'outputs.')

        output_names = self._predictive_model.get_output_names()
        for predictive_model in predictive_models:
            if output_names != predictive_model.get_output_names():
                raise Warning(
                    'The predictive models appear to have different output '
                    'names. Stacking of the predictive distributions might '
                    'therefore not meaningful.')

        if len(predictive_models) != len(weights):
            raise ValueError(
                'The model weights must be of the same length as the number '
                'of predictive models.')

        weights = np.array(weights, dtype=float)

        # Remember models and normalised weights
        self._predictive_models = predictive_models
        self._weights = weights / np.sum(weights)

    def get_predictive_model(self):
        """
        Returns a list of the
        :class:`chi.PosteriorPredictiveModel` instances.
        """
        return self._predictive_models

    def get_weights(self):
        """
        Returns the weights of the individual predictive models.
        """
        return copy.copy(self._weights)

    def sample(
            self, times, n_samples=None, individual=None, seed=None,
            include_regimen=False):
        """
        Samples "measurements" of the biomarkers from the posterior predictive
        model and returns them in form of a :class:`pandas.DataFrame`.

        For each of the ``n_samples`` a parameter set is drawn from the
        approximate posterior distribution. These paramaters are then used to
        sample from the predictive model.

        :param times: Times for the virtual "measurements".
        :type times: list, numpy.ndarray of shape (n,)
        :param n_samples: The number of virtual "measurements" that are
            performed at each time point. If ``None`` the biomarkers are
            measured only once at each time point.
        :type n_samples: int, optional
        :param individual: The ID of the modelled individual. If
            ``None``, either the first ID or the population is simulated.
        :type individual: str, optional
        :param seed: A seed for the pseudo-random number generator.
        :type seed: int
        :param include_regimen: A boolean flag which determines whether the
            information about the dosing regimen is included.
        :type include_regimen: bool, optional
        """
        # Make sure n_samples is an integer
        if n_samples is None:
            n_samples = 1
        n_samples = int(n_samples)

        # Instantiate random number generator
        seed = np.random.default_rng(seed=seed)

        # Sample number of samples from each predictive model
        n_models = len(self._predictive_models)
        model_indices = np.arange(n_models)
        model_draws = np.random.choice(
            model_indices, p=self._weights, size=n_samples)
        samples_per_model = np.zeros(n_models, dtype=int)
        for model_id, model in enumerate(model_indices):
            samples_per_model[model_id] = np.sum(
                model_draws == model)

        # Sample from predictive models
        samples = []
        for model_id, n_samples in enumerate(samples_per_model):
            # Skip model if no samples are drawn from it
            if n_samples == 0:
                continue

            # Sample
            model = self._predictive_models[model_id]
            s = model.sample(times, n_samples, individual, seed=seed)

            # Shift IDs by number of previous draws
            s['ID'] += int(np.sum(samples_per_model[:model_id]))

            # Append samples to list
            samples.append(s)

        # Concatenate all samples to one dataframe
        samples = pd.concat(samples)

        # Add dosing regimen, if set
        final_time = np.max(times)
        regimen = self.get_dosing_regimen(final_time)
        if (regimen is not None) and (include_regimen is True):
            # Append dosing regimen only once for all samples
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
            dose administration.

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
        for predictive_model in self._predictive_models:
            predictive_model.set_dosing_regimen(
                dose, start, duration, period, num)
