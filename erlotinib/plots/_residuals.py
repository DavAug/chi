#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go

import erlotinib.plots as eplt


class PredictiveResidualPlot(eplt.SingleFigure):
    """
    A figure class that visualises the residual error between the predictions
    of a predictive model and measured observations.

    Expects a :class:`pandas.DataFrame` of measurements with an ID, a time,
    biomarker and a measurement column. This dataset is used as reference to
    compute the residuals.

    Extends :class:`SingleFigure`.

    Parameters
    ----------
    data
        A :class:`pandas.DataFrame` with the time series PD data in form of
        an ID, time, and biomarker column.
    id_key
        Key label of the :class:`DataFrame` which specifies the ID column.
        The ID refers to the identity of an individual. Defaults to
        ``'ID'``.
    time_key
        Key label of the :class:`DataFrame` which specifies the time
        column. Defaults to ``'Time'``.
    biom_key
        Key label of the :class:`DataFrame` which specifies the PD
        biomarker column. Defaults to ``'Biomarker'``.
    meas_key
        Key label of the :class:`DataFrame` which specifies the column of
        the measured PD biomarker. Defaults to ``'Measurement'``.
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(
            self, data, id_key='ID', time_key='Time', biom_key='Biomarker',
            meas_key='Measurement', updatemenu=True):
        super(PredictiveResidualPlot, self).__init__(updatemenu)

        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, biom_key, meas_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Remember data and keys
        self._measurements = data
        self._keys = [id_key, time_key, biom_key, meas_key]

    def _add_mean_predicted_versus_observed_scatter_plot(
            self, meas, pred, biomarker, time_key, sample_key):
        """
        Adds a scatter plot of the mean predictions on the x-axis and
        the measured values on the y-axis. Each individual gets a
        different colour.
        """
        # Get a colour scheme
        colors = plotly.colors.qualitative.Plotly
        n_colors = len(colors)

        # Get measurement keys
        id_key, time_key_m, _, meas_key = self._keys

        # Add scatter plot for each individual
        ids = meas[id_key].unique()
        for index, _id in enumerate(ids):
            # Get relevant measurements
            mask = meas[id_key] == _id
            temp = meas[mask]
            times = temp[time_key_m]
            observations = temp[meas_key]
            mean_predictions = self._get_mean_predictions(
                pred, times, time_key, sample_key)

            # Plot mean predictions versus observations
            color = colors[index % n_colors]
            self._add_residual_trace(
                _id, mean_predictions, observations, color)

    # def _add_prediction_scatter_trace(self, times, samples):
    #     """
    #     Adds scatter plot of samples from the predictive model.
    #     """
    #     # Get colour (light blueish)
    #     color = plotly.colors.qualitative.Pastel2[1]

    #     # Add trace
    #     self._fig.add_trace(
    #         go.Scatter(
    #             x=times,
    #             y=samples,
    #             name="Predicted samples",
    #             showlegend=True,
    #             mode="markers",
    #             marker=dict(
    #                 symbol='circle',
    #                 color=color,
    #                 opacity=0.7,
    #                 line=dict(color='black', width=1))))

    # def _add_prediction_bulk_prob_trace(self, data):
    #     """
    #     Adds the bulk probabilities as two line plots (one for upper and lower
    #     limit) and shaded area to the figure.
    #     """
    #     # Construct times that go from min to max and back to min
    #     # (Important for shading with 'toself')
    #     times = data['Time'].unique()
    #     times = np.hstack([times, times[::-1]])

    #     # Get unique bulk probabilities and sort in descending order
    #     bulk_probs = data['Bulk probability'].unique()
    #     bulk_probs[::-1].sort()

    #     # Get colors (shift start a little bit, because 0th level is too light)
    #     n_traces = len(bulk_probs)
    #     shift = 2
    #     colors = plotly.colors.sequential.Blues[shift:shift+n_traces]

    #     # Add traces
    #     for trace_id, bulk_prob in enumerate(bulk_probs):
    #         # Get relevant upper and lower percentiles
    #         mask = data['Bulk probability'] == bulk_prob
    #         reduced_data = data[mask]

    #         upper = reduced_data['Upper'].to_numpy()
    #         lower = reduced_data['Lower'].to_numpy()
    #         values = np.hstack([upper, lower[::-1]])

    #         # Add trace
    #         self._fig.add_trace(go.Scatter(
    #             x=times,
    #             y=values,
    #             line=dict(width=1, color=colors[trace_id]),
    #             fill='toself',
    #             legendgroup='Model prediction',
    #             name='Predictive model',
    #             text="%s Bulk" % bulk_prob,
    #             hoverinfo='text',
    #             showlegend=True if trace_id == n_traces-1 else False))

    def _add_residual_trace(self, _id, mean_predictions, measurements, color):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(
            go.Scatter(
                x=mean_predictions,
                y=measurements,
                name="ID: %d" % _id,
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))))

    def _get_mean_predictions(
            self, pred, times, time_key, sample_key):
        """
        Returns a list of mean prediction estimates for the provided times.
        """
        means = np.empty(shape=len(times))
        for time_id, time in enumerate(times):
            mask = pred[time_key] == time
            mean = pred[mask][sample_key].mean()
            means[time_id] = mean

        return means

    def _get_relevant_measurements_and_predictions(
            self, data, biomarker, individual, time_key, biom_key):
        """
        Filters the observations for the relevant biomarker and ID. Also makes
        sure that there is a prediction for each measured time point.
        """
        # Get keys of measurement dataframe
        id_key_m, time_key_m, biom_key_m, _ = self._keys

        # Mask measurements for individual (if None keep all individuals)
        measurements = self._measurements
        if individual is not None:
            mask = measurements[id_key_m] == individual
            measurements = measurements[mask]

        # Mask measurements for biomarker
        mask = measurements[biom_key_m] == biomarker
        measurements = measurements[mask]

        # Make sure that there are predictions for each observed time
        measured_times = measurements[time_key_m].dropna().unique()
        mask = data[biom_key] == biomarker
        predictions = data[mask]
        predicted_times = predictions[time_key]
        for time in measured_times:
            if time not in predicted_times:
                raise ValueError(
                    'The prediction dataframe is not compatible with the '
                    'measurement dataframe. The prediction dataframe does not '
                    'provide predictions for the measurement time <%.3f>'
                    % time)

        return measurements, predictions

    # def _compute_bulk_probs(self, data, bulk_probs, time_key, sample_key):
    #     """
    #     Computes the upper and lower percentiles from the predictive model
    #     samples, corresponding to the provided bulk probabilities.
    #     """
    #     # Create container for perecentiles
    #     container = pd.DataFrame(columns=[
    #         'Time', 'Upper', 'Lower', 'Bulk probability'])

    #     # Translate bulk probabilities into percentiles
    #     percentiles = []
    #     for bulk_prob in bulk_probs:
    #         lower = 0.5 - bulk_prob / 2
    #         upper = 0.5 + bulk_prob / 2

    #         percentiles.append([bulk_prob, lower, upper])

    #     # Get unique times
    #     unique_times = data[time_key].unique()

    #     # Fill container with percentiles for each time
    #     for time in unique_times:
    #         # Mask relevant data
    #         mask = data[time_key] == time
    #         reduced_data = data[mask]

    #         # Get percentiles
    #         percentile_df = reduced_data[sample_key].rank(
    #             pct=True)
    #         for item in percentiles:
    #             bulk_prob, lower, upper = item

    #             # Get biomarker value corresponding to percentiles
    #             mask = percentile_df <= lower
    #             biom_lower = reduced_data[mask][sample_key].max()

    #             mask = percentile_df >= upper
    #             biom_upper = reduced_data[mask][sample_key].min()

    #             # Append percentiles to container
    #             container = container.append(pd.DataFrame({
    #                 'Time': [time],
    #                 'Lower': [biom_lower],
    #                 'Upper': [biom_upper],
    #                 'Bulk probability': [str(bulk_prob)]}))

    #     return container

    def add_data(
            self, data, biomarker=None, individual=None, bulk_probs=None,
            time_key='Time', biom_key='Biomarker', sample_key='Sample'):
        r"""
        Adds the residuals of the predicted biomarker values with respect
        to the measured values to the figure.

        Expects a :class:`pandas.DataFrame` with a time, a biomarker and a
        sample column. The time column determines the time of the biomarker
        measurement and the sample column the corresponding biomarker
        measurement. The biomarker column determines the biomarker type.

        The predictions are matched to the observations based on their ID and
        time label.

        A list of bulk probabilities ``bulk_probs`` can be specified, which are
        then added as shaded areas to the figure. The corresponding upper and
        lower percentiles are estimated from the ranks of the provided
        samples.

        .. warning::
            For low sample sizes the illustrated bulk probabilities may deviate
            significantly from the theoretical bulk probabilities. The upper
            and lower limit are determined from the rank of the samples for
            each time point.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        biomarker
            The predicted bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first biomarker
            type in the biomarker column is selected.
        individual
            The ID of the individual whose measurements are used as reference
            for the predictive residuals. Defaults to ``None`` which compares
            the predictions to all individuals.
        bulk_probs
            A list of bulk probabilities that are illustrated in the
            figure. If ``None`` the samples are illustrated as a scatter plot.
        time_key
            Key label of the :class:`pandas.DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`pandas.DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        sample_key
            Key label of the :class:`pandas.DataFrame` which specifies the
            sample column. Defaults to ``'Sample'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [time_key, biom_key, sample_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Check that selected individual exists in measurement dataframe
        if individual is not None:
            id_key_m = self._keys[0]
            ids = self._measurements[id_key_m].unique()
            if individual not in ids:
                raise ValueError(
                    'The ID <' + str(individual) + '> does not exist in the '
                    'measurement dataframe.')

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].dropna().unique()
        if biomarker is None:
            biomarker = biom_types[0]

        if biomarker not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Check that selected biomarker exists in the measurement dataframe
        biom_key_m = self._keys[2]
        biomarkers = self._measurements[biom_key_m].unique()
        if biomarker not in biomarkers:
            raise ValueError(
                'The biomarker <' + str(biomarker) + '> does not exist in the '
                'measurement dataframe.')

        # Get the relevant observations
        meas, pred = self._get_relevant_measurements_and_predictions(
            data, biomarker, individual, time_key, biom_key)

        # # Add bulk probabilities to plot

        # # Not more than 7 bulk probabilities are allowed (Purely aesthetic
        # # criterion)
        # if len(bulk_probs) > 7:
        #     raise ValueError(
        #         'At most 7 different bulk probabilities can be illustrated at '
        #         'the same time.')

        # # Make sure that bulk probabilities are between 0 and 1
        # bulk_probs = [float(probability) for probability in bulk_probs]
        # for probability in bulk_probs:
        #     if (probability < 0) or (probability > 1):
        #         raise ValueError(
        #             'The provided bulk probabilities have to between 0 and 1.')

        # # Add bulk probabilities to figure
        # percentile_df = self._compute_bulk_probs(
        #     data, bulk_probs, time_key, sample_key)
        # self._add_prediction_bulk_prob_trace(percentile_df)

        # Add mean predictions versus observations as scatter points
        self._add_mean_predicted_versus_observed_scatter_plot(
            meas, pred, biomarker, time_key, sample_key)
