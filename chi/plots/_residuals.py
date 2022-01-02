#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go

from chi import plots


class ResidualPlot(plots.SingleFigure):
    """
    A figure class that visualises the residual error between the predictions
    of a predictive model and measured observations.

    Expects a :class:`pandas.DataFrame` of measurements with an ID, a time,
    observable and a value column. This dataset is used as reference to
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
    obs_key
        Key label of the :class:`DataFrame` which specifies the observable
        column. Defaults to ``'Observable'``.
    value_key
        Key label of the :class:`DataFrame` which specifies the column of
        the measured values. Defaults to ``'Value'``.
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(
            self, measurements, id_key='ID', time_key='Time',
            obs_key='Observable', value_key='Value', updatemenu=True):
        super(ResidualPlot, self).__init__(updatemenu)

        # Check input format
        if not isinstance(measurements, pd.DataFrame):
            raise TypeError(
                'Measurements has to be pandas.DataFrame.')

        for key in [id_key, time_key, obs_key, value_key]:
            if key not in measurements.keys():
                raise ValueError(
                    'Measurements does not have the key <' + str(key) + '>.')

        # Remember data and keys
        self._measurements = measurements
        self._keys = [id_key, time_key, obs_key, value_key]

    def _add_predicted_versus_observed_scatter_plot(
            self, meas, pred, show_residuals, show_relative, time_key,
            sample_key):
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
            observations = temp[meas_key].to_numpy()
            mean_predictions = self._get_mean_predictions(
                pred, times, time_key, sample_key)

            if show_residuals is True:
                # Compute residuals of observations from mean predictions
                observations -= mean_predictions

            if show_relative is True:
                # Normalise observations by mean predictions
                observations /= mean_predictions

            # Plot mean predictions versus observations
            color = colors[index % n_colors]
            self._add_residual_trace(
                _id, mean_predictions, observations, color)

        # Add default axes labels
        xlabel = 'Prediction'
        ylabel = 'Residual' if show_residuals is True else 'Observable'
        if show_relative is True:
            ylabel += ' in rel. units'
        self._fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel)

    def _add_residual_trace(
            self, _id, mean_predictions, measurements, color):
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
            # Compute mean
            mask = pred[time_key] == time
            means[time_id] = pred[mask][sample_key].mean()

        return means

    def _get_relevant_measurements(
            self, data, biomarker, individual, time_key):
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
        predicted_times = data[time_key].to_numpy()
        for time in measured_times:
            if time not in predicted_times:
                raise ValueError(
                    'The prediction dataframe is not compatible with the '
                    'measurement dataframe. The prediction dataframe does not '
                    'provide predictions for the measurement time <%.3f>'
                    % time)

        return measurements

    def add_data(
            self, data, observable=None, individual=None, show_residuals=True,
            show_relative=False, time_key='Time', obs_key='Observable',
            value_key='Value'):
        r"""
        Adds the residuals of the predicted values with respect
        to the measured values to the figure.

        Expects a :class:`pandas.DataFrame` with a time, an observable and a
        value column. The time column determines the times of the
        measurements and the value column the measured
        values. The observable column determines the measured observable.

        The predictions are matched to the observations based on their ID and
        time label. If multiple predictions are provided for one measured time
        point, the mean prediction is computed as reference.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and observable column.
        observable
            The predicted bimoarker. This argument is used to determine the
            relevant rows in the dataframe. If ``None``, the first observable
            type in the observable column is selected.
        individual
            The ID of the individual whose measurements are used as reference
            for the predictive residuals. Defaults to ``None`` which compares
            the predictions to all individuals.
        show_residuals
            A boolean flag which indicates whether the residuals are plotted
            on the y axis, or the measurements themselves. Defaults to
            ``True``.
        show_relative
            A boolean flag which indicates whether the observations/residuals
            are normalised by the mean predictions. Defaults to ``False``.
        time_key
            Key label of the :class:`pandas.DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        obs_key
            Key label of the :class:`pandas.DataFrame` which specifies the
            observable column. Defaults to ``'Observable'``.
        value_key
            Key label of the :class:`pandas.DataFrame` which specifies the
            value column. Defaults to ``'Value'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [time_key, obs_key, value_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Check that selected individual exists in measurement dataframe
        if individual is not None:
            id_key_m = self._keys[0]
            ids = list(self._measurements[id_key_m].unique())
            if individual not in ids:
                raise ValueError(
                    'The ID <' + str(individual) + '> does not exist in the '
                    'measurement dataframe.')

        # Default to first bimoarker, if observable is not specified
        biom_types = data[obs_key].dropna().unique()
        if observable is None:
            observable = biom_types[0]

        if observable not in biom_types:
            raise ValueError(
                'The observable could not be found in the observable column.')

        # Check that selected observable exists in the measurement dataframe
        obs_key_m = self._keys[2]
        observables = self._measurements[obs_key_m].unique()
        if observable not in observables:
            raise ValueError(
                'The observable <' + str(observable) + '> does not exist in '
                'the measurement dataframe.')

        # Mask predictions for observable
        mask = data[obs_key] == observable
        data = data[mask]

        # Get the relevant observations
        meas = self._get_relevant_measurements(
            data, observable, individual, time_key)

        # Add mean predictions versus observations as scatter points
        self._add_predicted_versus_observed_scatter_plot(
            meas, data, show_residuals, show_relative, time_key,
            value_key)
