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

        # Instantiate figure and sliders
        self._fig = erlo.plots.PDTimeSeriesPlot(updatemenu=False)
        self._sliders = _SlidersComponent()

        # Create default layout
        self._set_layout()

        # Create default simulation and slider settings
        self._times = np.linspace(start=0, stop=30)

    def _add_simulation(self):
        """
        Adds trace of simulation results to the figure.
        """
        # Get parameter values
        parameters = []
        print(self._sliders)
        for slider_component in self._sliders:
            _, slider = slider_component
            value = slider.value
            parameters += value

        print(parameters)

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

    def _create_sliders(self, pk_input, initial_values, parameters):
        """
        Creates one slider for each parameter, and groups the slider by
        1. Pharmacokinetic input
        2. Initial values (of states)
        3. Parameters
        """
        # Add one slider for each parameter
        for parameter in parameters:
            self._sliders.add_slider(slider_id=parameter)

        # Create PK input slider group
        if pk_input is not None:
            self._sliders.group_sliders(
                slider_ids=[pk_input], group_id='Pharmacokinetic input')
            parameters.remove(pk_input)

        # Create initial values slider group
        self._sliders.group_sliders(
            slider_ids=initial_values, group_id='Initial values')
        for state in initial_values:
            parameters.remove(state)

        # Create parameters slider group
        self._sliders.group_sliders(
            slider_ids=parameters, group_id='Parameters')

    def _create_sliders_component(self):
        """
        Returns a slider component.
        """
        sliders = dbc.Col(
            children=self._sliders(),
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

    def _simulate(parameters, times):
        """
        Returns simulation of pharmacodynamic model in standard format, i.e.
        pandas.DataFrame with 'Time' and 'Biomarker' column.
        """
        #TODO:

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
        pk_input = model.pk_input()
        state_names = model._state_names
        parameters = model.parameters()
        self._create_sliders(pk_input, state_names, parameters)
        self._set_layout()

        # # Add simulation of model to the figure
        # self._add_simulation()

    def set_axis_labels(self, xlabel, ylabel):
        """
        Sets the x axis, and y axis label of the figure.
        """
        self._fig.set_axis_labels(xlabel, ylabel)


class _SlidersComponent(object):
    """
    A helper class that helps to organise the sliders of the
    :class:`SimulationController`.

    The sliders are arranged horizontally. Sliders may be grouped by meaning.
    """

    def __init__(self):
        # Set defaults
        self._range = (0, 10)
        self._value = 1
        self._sliders = {}
        self._slider_groups = {}

    def __call__(self):
        # Returns the contents in form of a list of dash components.

        # If no sliders have been added, print a default message.
        if not self._sliders:
            default = [dbc.Alert(
                "No model has been chosen.", color="primary")]
            return default

        # If sliders have not been grouped, print a default message.
        if not self._sliders:
            default = [dbc.Alert(
                "Sliders have not been grouped.", color="primary")]
            return default

        # Group and label sliders
        contents = self._compose_contents()
        return contents

    def _compose_contents(self):
        """
        Returns the grouped sliders with labels as a list of dash components.
        """
        contents = []
        for group_id in self._slider_groups.keys():
            # Create label for group
            group_label = html.Label(group_id)

            # Group sliders
            group = self._slider_groups[group_id]
            container = []
            for slider_id in group:
                # Create label for slider
                label = html.Label(slider_id, style={'fontSize': '0.8rem'})
                slider = self._sliders[slider_id]

                # Add label and slider to group container
                container += [
                    dbc.Col(children=[label], width=12),
                    dbc.Col(children=[slider], width=12)]

            # Convert slider group to dash component
            group = dbc.Row(
                children=container, style={'marginBottom': '1em'})

            # Add label and group to contents
            contents += [group_label, group]

        return contents

    def add_slider(
            self, slider_id, value=1, min_value=0, max_value=10,
            step_size=0.1):
        """
        Adds a slider.

        Parameters
        ----------
        slider_id
            ID of the slider.
        value
            Default value of the slider.
        min_value
            Minimal value of slider.
        max_value
            Maximal value of slider.
        step_size
            Elementary step size of slider.
        """
        self._sliders[slider_id] = dcc.Slider(
            id=slider_id,
            value=value,
            min=min_value,
            max=max_value,
            step=step_size,
            marks={
                str(min_value): str(min_value),
                str(max_value): str(max_value)})

    def group_sliders(self, slider_ids, group_id):
        """
        Visually groups sliders. Group ID will be used as label.

        Each slider can only be in one group.
        """
        # Check that incoming sliders do not belong to any group already
        for index, existing_group in enumerate(self._slider_groups.values()):
            for slider in slider_ids:
                if slider in existing_group:
                    raise ValueError(
                        'Slider <' + str(slider) + '> exists already in group '
                        '<' + str(self._slider_groups.keys()[index]) + '>.')

        self._slider_groups[group_id] = slider_ids


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
