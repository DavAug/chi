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


class PDDataPlot(eplt.Figure):
    """
    Figure class that visualises pharmacodynamic data.
    """

    def __init__(self):
        super(PDDataPlot, self).__init__()

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

    def add_data(self, data, id_key=None, time_key=None, pd_key=None):
        """
        Adds pharmacodynamic data of multiple individuals to figure.

        Expects a pandas.DataFrame with an `ID`, `TIME` and `PD` column. If the
        column names differ, column keys can be identifies using the optional
        key arguments `id_key`, `time_key` and `pd_key`.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                'Data has to be pandas.DataFrame.')

        if id_key is None:
            id_key = 'ID'
        if time_key is None:
            time_key = 'TIME'
        if pd_key is None:
            pd_key = 'PD'

        for key in [id_key, time_key, pd_key]:
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
