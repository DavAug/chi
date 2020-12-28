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


class PDPredictivePlot(eplt.SingleFigure):
    """
    A figure class that visualises the predictions of a predictive
    pharmacodynamic model.

    Extends :class:`SingleFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(PDPredictivePlot, self).__init__(updatemenu)

    def _add_data_trace(self, label, times, biomarker, color):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=biomarker,
                name="ID: %d" % label,
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))))

    def _add_prediction_scatter_trace(self, times, samples):
        """
        Adds scatter plot of samples from the predictive model.
        """
        # Get colour (light blueish)
        color = plotly.colors.qualitative.Pastel2[1]

        # Add trace
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=samples,
                name="Predicted samples",
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))))

    def _add_prediction_bulk_prob_trace(self, data):
        """
        Adds the bulk probabilities as two line plots (one for upper and lower
        limit) and shaded area to the figure.
        """
        # Construct times that go from min to max and back to min
        # (Important for shading with 'toself')
        times = data['Time'].to_numpy()
        times = np.hstack([times, times[::-1]])

        # Get unique bulk probabilities
        bulk_probs = data['Bulk probabilities'].unique()

        # Get colors (shift start a little bit, because 0th level is too light)
        n_traces = len(bulk_probs)
        shift = 2
        colors = plotly.colors.sequential.Blues[shift:shift+n_traces:-1]

        # Add traces
        for trace_id, bulk_prob in bulk_probs:
            # Get relevant upper and lower percentiles
            mask = data['Bulk probability'] == bulk_prob
            reduced_data = data[mask]

            upper = reduced_data['Upper'].to_numpy()
            lower = reduced_data['Lower'].to_numpy()
            values = np.hstack([upper, lower[::-1]])

            # Add trace
            self._fig.add_trace(go.Scatter(
                x=times,
                y=values,
                line=dict(width=1, color=colors[trace_id]),
                fill='toself',
                legendgroup='Model prediction',
                name='Prediction',
                text="%s Bulk" % bulk_prob,
                hoverinfo='text',
                showlegend=True if trace_id == 0 else False))

    def _compute_bulk_probs(self, data, bulk_probs, time_key, sample_key):
        """
        Computes the upper and lower percentiles from the predictive model
        samples, corresponding to the provided bulk probabilities.
        """
        # Create container for perecentiles
        container = pd.DataFrame(columns=[
            'Time', 'Upper', 'Lower', 'Bulk probability'])

        # Translate bulk probabilities into percentiles
        percentiles = []
        for bulk_prob in bulk_probs:
            lower = 0.5 - bulk_prob / 2
            upper = 0.5 + bulk_prob / 2

            percentiles.append([bulk_prob, lower, upper])

        # Get unique times
        unique_times = data[time_key].unique()

        # Fill container with percentiles for each time
        for time in unique_times:
            # Mask relevant data
            mask = data[time_key] == time
            reduced_data = data[mask]

            # Get percentiles
            reduced_data['Percentile'] = reduced_data[sample_key].rank(
                pct=True)
            for item in percentiles:
                bulk_prob, lower, upper = item

                # Get biomarker value corresponding to percentiles
                mask = reduced_data['Percentile'] <= lower
                biom_lower = reduced_data[mask]['Sample'].max()

                mask = reduced_data['Percentile'] >= upper
                biom_upper = reduced_data[mask]['Sample'].min()

                # Append percentiles to container
                container = container.append(pd.DataFrame({
                    'Time': [time],
                    'Lower': [biom_lower],
                    'Upper': [biom_upper],
                    'Bulk probability': [str(bulk_prob)]}))

        return container

    def add_data(
            self, data, id_key='ID', time_key='Time', biom_key='Biomarker'):
        """
        Adds pharmacodynamic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time and a PD
        biomarker column, and adds a scatter plot of the biomarker time series
        to the figure. Each individual receives a unique colour.

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
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, biom_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Get a unique colour for each individual
        ids = data[id_key].unique()
        n_ids = len(ids)
        colors = plotly.colors.qualitative.Plotly[:n_ids]

        # Fill figure with scatter plots of individual data
        for index, label in enumerate(ids):
            # Get individual data
            mask = data[id_key] == label
            times = data[time_key][mask]
            biomarker = data[biom_key][mask]
            color = colors[index]

            # Create Scatter plot
            self._add_data_trace(label, times, biomarker, color)

    def add_prediction(
            self, data, biom=None, bulk_probs=[0.3, 0.6, 0.9], time_key='Time',
            biom_key='Biomarker', sample_key='Sample'):
        r"""
        Adds the prediction for the observable pharmacodynamic biomarker values
        to the figure.

        Expects a :class:`pandas.DataFrame` with a time, a PD biomarker and a
        sample column. The time column determines the time of the biomarker
        measurement and the sample column the corresponding biomarker
        measurement. The biomarker column determines the biomarker type.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        biom
            The predicted bimoarker. This argument is used to determin the
            relevant rows in dataframe. If ``None`` the first biomarker type
            in the biomarker column is selected.
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

        # Default to first bimoarker, if biomarker is not specified
        biom_types = data[biom_key].unique()
        if biom is None:
            biom = biom_types[0]

        if biom not in biom_types:
            raise ValueError(
                'The biomarker could not be found in the biomarker column.')

        # Mask data for biomarker
        mask = data[biom_key] == biom
        data = data[mask]

        # Add samples as scatter plot if no bulk probabilites are provided, and
        # terminate method
        if bulk_probs is None:
            times = data[time_key]
            samples = data[sample_key]
            self._add_prediction_scatter_trace(times, samples)

            return None

        # Make sure that bulk probabilities are between 0 and 1
        bulk_probs = [float(probability) for probability in bulk_probs]
        for probability in bulk_probs:
            if (probability < 0) or (probability > 1):
                raise ValueError(
                    'The provided bulk probabilities have to between 0 and 1.')

        # Add bulk probabilities to figure
        percentile_df = self._compute_bulk_probs(
            data, bulk_probs, time_key, sample_key)
        self._add_prediction_bulk_prob_trace(percentile_df)


class PDTimeSeriesPlot(eplt.SingleFigure):
    """
    A figure class that visualises measurements of a pharmacodynamic biomarker
    across multiple individuals.

    Measurements of a pharmacodynamic biomarker over time are visualised as a
    scatter plot.

    Extends :class:`SingleFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(PDTimeSeriesPlot, self).__init__(updatemenu)

    def _add_data_trace(self, label, times, biomarker, color):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=biomarker,
                name="ID: %d" % label,
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))))

    def _add_simulation_trace(self, times, biomarker):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=biomarker,
                name="Model",
                showlegend=True,
                mode="lines",
                line=dict(color='black')))

    def add_data(
            self, data, id_key='ID', time_key='Time', biom_key='Biomarker'):
        """
        Adds pharmacodynamic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time and a PD
        biomarker column, and adds a scatter plot of the biomarker time series
        to the figure. Each individual receives a unique colour.

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
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, biom_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Get a unique colour for each individual
        ids = data[id_key].unique()
        n_ids = len(ids)
        colors = plotly.colors.qualitative.Plotly[:n_ids]

        # Fill figure with scatter plots of individual data
        for index, label in enumerate(ids):
            # Get individual data
            mask = data[id_key] == label
            times = data[time_key][mask]
            biomarker = data[biom_key][mask]
            color = colors[index]

            # Create Scatter plot
            self._add_data_trace(label, times, biomarker, color)

    def add_simulation(self, data, time_key='Time', biom_key='Biomarker'):
        """
        Adds a pharmacodynamic time series simulation to the figure.

        Expects a :class:`pandas.DataFrame` with a time and a PD biomarker
        column, and adds a line plot of the biomarker time series to the
        figure.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        time_key
            Key label of the :class:`DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [time_key, biom_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        times = data[time_key]
        biomarker = data[biom_key]

        self._add_simulation_trace(times, biomarker)


class PKTimeSeriesPlot(eplt.SingleSubplotFigure):
    """
    A figure class that visualises measurements of a pharmacokinetic biomarker
    across multiple individuals.

    Measurements of a pharmacokinetic biomarker over time are visualised as a
    scatter plot.

    Extends :class:`SingleSubplotFigure`.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(PKTimeSeriesPlot, self).__init__()

        self._create_template_figure(
            rows=2, cols=1, shared_x=True, row_heights=[0.2, 0.8])

        if updatemenu:
            self._add_updatemenu()

    def _add_dose_trace(self, label, times, dose, color):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=dose,
                name="ID: %d" % label,
                legendgroup="ID: %d" % label,
                showlegend=False,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))),
            row=1,
            col=1)

    def _add_biom_trace(self, label, times, biomarker, color):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(
            go.Scatter(
                x=times,
                y=biomarker,
                name="ID: %d" % label,
                legendgroup="ID: %d" % label,
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))),
            row=2,
            col=1)

    def _add_updatemenu(self):
        """
        Adds a button to the figure that switches the biomarker scale from
        linear to logarithmic.
        """
        self._fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"yaxis2.type": "linear"}],
                            label="Linear y-scale",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis2.type": "log"}],
                            label="Log y-scale",
                            method="relayout"
                        )
                    ]),
                    pad={"r": 0, "t": -10},
                    showactive=True,
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
                )
            ]
        )

    def add_data(
            self, data, id_key='ID', time_key='Time', biom_key='Biomarker',
            dose_key='Dose'):
        """
        Adds pharmacokinetic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an ID, a time and a PD
        biomarker column, and adds a scatter plot of the biomarker time series
        to the figure. Each individual receives a unique colour.

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
        dose_key
            Key label of the :class:`DataFrame` which specifies the dose
            column. Defaults to ``'Dose'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, biom_key, dose_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Set axis labels to dataframe keys
        self.set_axis_labels(time_key, biom_key, dose_key)

        # Get a unique colour for each individual
        ids = data[id_key].unique()
        n_ids = len(ids)
        colors = plotly.colors.qualitative.Plotly[:n_ids]

        # Fill figure with scatter plots of individual data
        for index, label in enumerate(ids):
            # Get individual data
            mask = data[id_key] == label
            times = data[time_key][mask]
            biomarker = data[biom_key][mask]
            dose = data[dose_key][mask]
            color = colors[index]

            # Create scatter plot of dose events
            self._add_dose_trace(label, times, dose, color)

            # Create Scatter plot
            self._add_biom_trace(label, times, biomarker, color)

    def add_simulation(
            self, data, time_key='Time', biom_key='Biomarker',
            dose_key='Dose'):
        """
        Adds a pharmacokinetic time series simulation to the figure.

        Expects a :class:`pandas.DataFrame` with a time, a PK biomarker,
        and a dose column. A line plot of the biomarker time series, as well
        as the dosing regimen is added to the figure.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the time series PD simulation in
            form of a time and biomarker column.
        time_key
            Key label of the :class:`DataFrame` which specifies the time
            column. Defaults to ``'Time'``.
        biom_key
            Key label of the :class:`DataFrame` which specifies the PD
            biomarker column. Defaults to ``'Biomarker'``.
        """
        raise NotImplementedError

    def set_axis_labels(self, time_label, biom_label, dose_label):
        """
        Sets the label of the time axis, the biomarker axis, and the dose axis.
        """
        self._fig.update_xaxes(title=time_label, row=2)
        self._fig.update_yaxes(title=dose_label, row=1)
        self._fig.update_yaxes(title=biom_label, row=2)
