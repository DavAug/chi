#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import dash_bootstrap_components as dbc
import dash_core_components as dcc

import erlotinib as erlo


class PDSimulationController(erlo.apps.BaseApp):
    """
    Creates an app which simulates a :class:`PharmacodynamicModel`.

    Parameter sliders can be used to adjust parameter values during
    the simulation.

    Extends :class:`BaseApp`.
    """

    def __init__(self):
        super(PDSimulationController, self).__init__(
            name='PDSimulationController')

        # Instantiate figure
        self._fig = erlo.plots.PDTimeSeriesPlot()

        # Create default layout
        self._sliders = dbc.Alert(
            "No model has been chosen.", color="primary")
        self._set_layout()

    def _create_figure_component(self):
        """
        Returns a figure component.
        """
        figure = dbc.Col(
            children=[dcc.Graph(
                figure=self._fig._fig,
                id='fig')],
            md=9
        )

        return figure

    def _create_sliders_component(self):
        """
        Returns a slider component.
        """
        sliders = dbc.Col(
            children=[self._sliders],
            md=3,
            style={'marginTop': '5em'}
        )

        return sliders

    def _set_layout(self):
        """
        Sets the layout of the app.

        - Plot of simulation/data on the left.
        - Parameter sliders on the right.
        """
        self._app.layout = dbc.Container(
            children=[dbc.Row([
                self._create_figure_component(),
                self._create_sliders_component()])],
            style={'marginTop': '5em'})

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
        self._fig.add_data(data, id_key, time_key, biom_key)

    def add_model(self, model):
        """
        Adds a :class:`PharmacodynamicModel` to the application.

        One parameter slider is generated for each model parameter, and
        the solution for a default set of parameters is added to the figure.
        """



# For simple debugging the app can be launched by executing the python file.
if __name__ == "__main__":
    app = PDSimulationController()
    app.add_data(erlo.DataLibrary().lung_cancer_control_group(True))
    app.start_application(debug=True)
