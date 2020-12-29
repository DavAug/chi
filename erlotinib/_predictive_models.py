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

        # Remember models
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models

        # Set parameter names and number of parameters
        parameter_names = self._mechanistic_model.parameters()
        for error_model in error_models:
            parameter_names += error_model.get_parameter_names()
        self._parameter_names = parameter_names
        self._n_parameters = len(self._parameter_names)

    def get_n_outputs(self):
        """
        Returns the number of outputs.
        """
        return self._mechanistic_model.n_outputs()

    def get_outputs(self):
        """
        Returns the output names.
        """
        return self._mechanistic_model.outputs()

    def get_parameter_names(self):
        """
        Returns the parameter names of the predictive model.
        """
        return self._parameter_names

    def n_parameters(self):
        """
        Returns the number of parameters of the predictive model.
        """
        return self._n_parameters

    def sample(
            self, parameters, times, n_samples=None, seed=None,
            return_df=True):
        """
        Samples "measurements" of the biomarkers from the predictive model and
        returns them in form of a :class:`pandas.DataFrame`.

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
            columns=['Sample ID', 'Biomarker', 'Time', 'Sample'])

        # Fill in all samples at a specific time point at once
        for output_id, name in enumerate(output_names):
            for time_id, time in enumerate(times):
                samples = samples.append(pd.DataFrame({
                    'Sample ID': sample_ids,
                    'Time': time,
                    'Biomarker': name,
                    'Sample': container[output_id, time_id, :]}))

        return samples


class PriorPredictiveModel(object):
    """
    Implements a model that predicts the change of observable biomarkers over
    time based on assumptions about the model parameters prior to inference.

    A PriorPredictiveModel is instantiated with an instance of a
    :class:`PredictiveModel` and a :class:`pints.LogPrior` of the same
    parametric dimension as the predictive model. Future biomarker
    "measurement" can then be predicted by first sampling parameter values from
    the log-prior distribution, and then sampling from the predictive model
    with those parameters.

    Parameters
    ----------
    predictive_model
        An instance of a :class:`PredictiveModel`.
    log_prior
        An instance of a :class:`pints.LogPrior` of the same dimensionality as
        predictive model parameters.
    """

    def __init__(self, predictive_model, log_prior):
        super(PriorPredictiveModel, self).__init__()

        # Check inputs
        if not isinstance(predictive_model, PredictiveModel):
            raise ValueError(
                'The provided predictive model has to be an instance of a '
                'erlotinib.PredictiveModel.')

        if not isinstance(log_prior, pints.LogPrior):
            raise ValueError(
                'The provided log-prior has to be an instance of a '
                'pints.LogPrior.')

        if predictive_model.n_parameters() != log_prior.n_parameters():
            raise ValueError(
                'The dimension of the log-prior has to be the same as the '
                'number of parameters of the predictive model.')

        self._predictive_model = predictive_model
        self._log_prior = log_prior

    def sample(self, times, n_samples=None, seed=None):
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
            columns=['Sample ID', 'Biomarker', 'Time', 'Sample'])

        # Get model outputs (biomarkers)
        outputs = self._predictive_model.get_outputs()

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
                    'Sample ID': sample_id,
                    'Biomarker': name,
                    'Time': times,
                    'Sample': sample[output_id, :, 0]}))

        return container
