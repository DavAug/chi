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


class PDTimeSeriesPlot(eplt.Figure):
    """
    A figure class that visualises pharmacodynamic data and PD simulation
    results.
    """

    def __init__(self):
        super(PDTimeSeriesPlot, self).__init__()

    def _add_trace(self, label, times, pd_data, color):
        """
        Adds scatter plot of an indiviudals pharamcodynamics to figure.
        """
        self._fig.add_trace(go.Scatter(
                x=times,
                y=pd_data,
                name="ID: %d" % label,
                showlegend=True,
                mode="markers",
                marker=dict(
                    symbol='circle',
                    color=color,
                    opacity=0.7,
                    line=dict(color='black', width=1))))

    def add_data(self, data, id_key='ID', time_key='TIME', data_key='DATA'):
        """
        Adds pharmacodynamic time series data of (multiple) individuals to
        the figure.

        Expects a :class:`pandas.DataFrame` with an `ID`, `time` and `data`
        column, and adds a scatter plot of the data values versus time points
        to the figure. Each individual receives a unique colour.

        By default the column keys for the ID, time and data column are
        ``'ID'``, ``'TIME'``, and ``'DATA'``, respectively.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, time_key, data_key]:
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
            pd_data = data[pd_key][mask]
            color = colors[index]

            # Create Scatter plot
            self._add_trace(label, times, pd_data, color)
