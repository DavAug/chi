#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import pandas as pd
import plotly.colors
import plotly.graph_objects as go

import erlotinib.plots as eplt


class PDTimeSeriesPlot(eplt.SingleFigure):
    """
    A figure class that visualises pharmacodynamic data and PD simulation
    results.

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
    A figure class that visualises pharmacodynamic data and PD simulation
    results.

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
        raise NotImplementedError

    def set_axis_labels(self, time_label, biom_label, dose_label):
        """
        Sets the label of the time axis, the biomarker axis, and the dose axis.
        """
        self._fig.update_xaxes(title=time_label, row=2)
        self._fig.update_yaxes(title=dose_label, row=1)
        self._fig.update_yaxes(title=biom_label, row=2)
