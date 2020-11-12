#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np

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
        self._sliders = [dbc.Alert(
            "No model has been chosen.", color="primary")]
        self._set_layout()

        # Create default simulation and slider settings
        self._times = np.linspace(start=0, stop=30)
        self._slider_range = (0, 10)
        self._slider_value = 1

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

    def _create_sliders(self, parameters, pk_input):
        """
        Creates one slider for each parameter.

        The pk_input (typically drug concentration) is visualised in a
        slightly separate block than the remaining parameters.
        """
        sliders = []
        lower, upper = self._slider_range

        # Add pharamcokinetic input slider
        if pk_input is not None:
            slider = [
                html.Label('%s' % pk_input),
                dcc.Slider(
                    id='%s' % pk_input,
                    value=self._slider_value,
                    min=lower,
                    max=upper,
                    step=0.1,
                    marks={str(lower): str(lower), str(upper): str(upper)})
            ]
            sliders += slider

        # Add sliders for remaining parameter
        parameters.remove(pk_input)
        for parameter in parameters:
            slider = [
                html.Label('%s' % parameter),
                dcc.Slider(
                    id='%s' % parameter,
                    value=self._slider_value,
                    min=lower,
                    max=upper,
                    step=0.1,
                    marks={str(lower): str(lower), str(upper): str(upper)})
            ]
            sliders += slider

        self._sliders = sliders

    def _create_sliders_component(self):
        """
        Returns a slider component.
        """
        sliders = dbc.Col(
            children=self._sliders,
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
        if not isinstance(model, erlo.PharmacodynamicModel):
            raise TypeError(
                'Model has to be an instance of '
                'erlotinib.PharamcodynamicModel.')

        # Add one slider for each parameter to the app
        parameters = model.parameters()
        pk_input = model.pk_input()
        self._create_sliders(parameters, pk_input)
        self._set_layout()

        # # Add simulation of model to the figure
        # self._add_simulation()

    def set_axis_labels(self, xlabel, ylabel):
        """
        Sets the x axis, and y axis label of the figure.
        """
        self._fig.set_axis_labels(xlabel, ylabel)


# For simple debugging the app can be launched by executing the python file.
if __name__ == "__main__":
    # Get data and model
    data = erlo.DataLibrary().lung_cancer_control_group(True)
    path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
    model = erlo.PharmacodynamicModel(path)

    # Set up demo app
    app = PDSimulationController()
    app.add_data(data)
    app.add_model(model)

    app.start_application(debug=True)
