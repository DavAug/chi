#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import plotly.graph_objects as go


class SingleFigure(object):
    """
    Base class for plot classes that generate a single figure.

    Parameter
    ---------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(SingleFigure, self).__init__()

        self._fig = go.Figure()
        self._set_layout(updatemenu)

    def _set_layout(self, updatemenu):
        """
        Configures the basic layout of the figure.

        - Size
        - Template
        - Log and linear y scale switches
        """
        self._fig.update_layout(
            autosize=True,
            template="plotly_white")

        if updatemenu:
            self._fig.update_layout(
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
        Sets the x axis, and y axis label of the figure.
        """
        self._fig.update_layout(
            xaxis_title=xlabel,
            yaxis_title=ylabel)

    def show(self):
        """
        Displays the figure.
        """
        self._fig.show()


class MultiFigure(object):
    """
    Base class for plot classes that generate multiple figures.
    """

    def __init__(self):
        super(MultiFigure, self).__init__()

        # Create a template figure
        self._fig = go.Figure()
        self._set_layout()
        self._figs = [self._fig]

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

    def show(self):
        """
        Displays the figures.
        """
        for fig in self._figs:
            fig.show()
