#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SingleFigure(object):
    """
    Base class for plot classes that generate a single figure.

    Parameters
    ----------
    updatemenu
        Boolean flag that enables or disables interactive buttons, such as a
        logarithmic scale switch for the y-axis.
    """

    def __init__(self, updatemenu=True):
        super(SingleFigure, self).__init__()

        self._fig = go.Figure()
        self._set_layout(updatemenu)
        self._configure_modebar()

    def _configure_modebar(self):
        #TODO: CHecl whether this works!
        # Change this for all base figures and make sure that it works for all figures
        # Check whether in inkscape updatemenu is easily removed
        """
        Configures the modebar of the figures as follows:

        - Makes images downlodable as .svg
        - By default upscales image axes labels by 1.5.
        - Hides plotly logo
        - Allows to draw into figures.
        - Removes Pan, Box select, Zoom in, Zoom out, Reset axis
        """
        self._config = dict({
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': 'erlotinib_image',
                'height': 600,
                'width': 800},
            'displaylogo': False,
            'modeBarButtonsToAdd': [
                'drawopenpath', 'eraseshape'],
            'modeBarButtonsToRemove': [
                 'pan2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d']
        })

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
        self._fig.show(config=self._config)


class SingleSubplotFigure(object):
    """
    Base class for plot classes that a single figure with subplots.
    """

    def __init__(self):
        super(SingleSubplotFigure, self).__init__()

        self._fig = go.Figure()
        self._set_layout()

    def _create_template_figure(
            self, rows, cols, x_title=None, y_title=None, shared_x=False,
            shared_y=False, spacing=0.05, column_widths=None,
            row_heights=None):
        """
        Creates a template figure using :meth:`plotly.make_subplots`.
        """
        self._fig = make_subplots(
            rows=rows, cols=cols, shared_xaxes=shared_x, shared_yaxes=shared_y,
            x_title=x_title, y_title=y_title, horizontal_spacing=spacing,
            vertical_spacing=spacing, column_widths=column_widths,
            row_heights=row_heights)

        # Set layout
        self._set_layout()

    def _set_layout(self):
        """
        Configures the basic layout of the figure.

        - Size
        - Template
        """
        self._fig.update_layout(
            autosize=True,
            template="plotly_white")

    def show(self):
        """
        Displays the figure.
        """
        self._fig.show()


class MultiFigure(object):
    """
    Base class for plot classes that generate multiple figures.
    """

    def __init__(self, updatemenu=True):
        super(MultiFigure, self).__init__()

        # Create a template figure
        self._fig = go.Figure()
        self._set_layout(updatemenu)
        self._figs = [self._fig]

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

    def show(self):
        """
        Displays the figures.
        """
        for fig in self._figs:
            fig.show()


class MultiSubplotFigure(MultiFigure):
    """
    Base class for plot classes that generate multiple figures with subplots.

    Extends :class:`MultiFigure`.
    """

    def __init__(self):
        super(MultiSubplotFigure, self).__init__(updatemenu=False)

    def _create_template_figure(
            self, rows, cols, x_title=None, y_title=None, shared_x=False,
            shared_y=True, spacing=0.05):
        """
        Creates a template figure using :meth:`plotly.make_subplots`.
        """
        self._fig = make_subplots(
            rows=rows, cols=cols, shared_xaxes=shared_x, shared_yaxes=shared_y,
            x_title=x_title, y_title=y_title, horizontal_spacing=spacing,
            vertical_spacing=spacing)

        # Set layout
        self._set_layout(updatemenu=False)
