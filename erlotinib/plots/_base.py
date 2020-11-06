#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import plotly.graph_objects as go


class Figure(object):
    """
    Base class for figures.
    """

    def __init__(self):
        super(Figure, self).__init__()

        self._fig = go.Figure()
        self._set_layout()

    def _set_layout(self):
        """
        Configures the basic layout of the figure.

        - Size
        - Template
        - Log and linear y scale switches
        """
        self._fig.update_layout(
            autosize=True,
            template="plotly_white",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"yaxis.type": "linear"}],
                            label="Linear y-scale",
                            method="relayout"
                        ),
                        dict(
                            args=[{"yaxis.type": "log"}],
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

    def add_data(self, data):
        """
        Adds data to the figure.
        """
        raise NotImplementedError

    def set_axis_labels(self, xlabel, ylabel):
        """
        Sets the x axis and y axis label.
        """
        self._fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel)

    def show(self):
        """
        Displays figure.
        """
        self._fig.show()
