#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy

import pints
import plotly.colors
import plotly.graph_objects as go
import xarray as xr

import erlotinib.plots as eplt


class MarginalPosteriorPlot(eplt.MultiSubplotFigure):
    """
    A figure class that visualises the marginal posterior probability for each
    parameter across individuals.

    One figure is generated for each parameter, which contains a marginal
    histogram of the sampled parameter values for each individual. The
    estimates for each indiviudal are plotted next to each other.

    This figure can be used to assess the convergence of the sampling method,
    as well as the variation of parameter estimates across individuals.

    Extends :class:`MultiFigure`.
    """
    def __init__(self):
        super(MarginalPosteriorPlot, self).__init__()

    def _add_histogram_plots(self, fig_id, data, colors):
        """
        Adds histogram plots of the parameter samples for each individual to
        the figure.

        One figure contains only the histograms of one parameter.
        """
        # Get number of colours
        n_colors = len(colors)

        # Check whether data are samples of a population parameter or
        # contains samples for many individuals
        is_population = False
        try:
            # Get ids of individuals
            ids = data.individual.values
        except AttributeError:
            is_population = True

        if is_population is True:
            # Compute diagnostics
            samples = data.values
            diagnostics = self._compute_diagnostics(samples)

            # Add trace
            color = colors[0]
            index = 0
            individual = 'Population'
            samples = samples.flatten()
            self._add_trace(
                fig_id, index, individual, samples, diagnostics, color)

            return None

        # Add trace for each individual
        for index, individual in enumerate(ids):
            # Get data for individual
            samples = data.sel(individual=individual).dropna(dim='draw')

            # Compute diagnostics
            samples = samples.values
            diagnostics = self._compute_diagnostics(samples)

            # Add trace
            samples = samples.flatten()
            color = colors[index % n_colors]
            self._add_trace(
                fig_id, index, individual, samples, diagnostics, color)

    def _add_trace(
            self, fig_id, index, individual, samples, diagnostics, color):
        """
        Adds a histogram of an indiviudals samples to a figure.
        """
        # Get figure
        fig = self._figs[fig_id]

        # Add trace
        rhat, = diagnostics
        fig.add_trace(
            go.Histogram(
                y=samples,
                name='%s' % str(individual),
                hovertemplate=(
                    'Sample: %{y:.2f}<br>' +
                    'Rhat: %.02f<br>' % rhat),
                visible=True,
                marker=dict(color=color),
                opacity=0.8),
            row=1,
            col=index+1)

        # Turn off xaxis ticks
        fig.update_xaxes(
            tickvals=[],
            row=1,
            col=index+1)

        # Set x axis title to individual
        fig.update_xaxes(
            title_text=str(individual), row=1, col=index+1)

    def _compute_diagnostics(self, data):
        """
        Computes and returns convergence metrics.

        - Rhat
        """
        # Compute rhat
        rhat = pints.rhat(chains=data)

        # Collect diagnostics
        diagnostics = [rhat]

        return diagnostics

    def add_data(self, data):
        """
        Adds marginal histograms of the samples across runs to the figure.

        The histograms of population parameters are visualised in separate
        figures,  while the individual parameters for one parameter type
        are grouped together.

        :param data: A :class:`xarray.Dataset` with the posterior samples.
        :type data: xarray.Dataset
        """
        # Check input format
        if not isinstance(data, xr.Dataset):
            raise TypeError(
                'The data has to be a xarray.Dataset.')

        dims = sorted(list(data.dims))
        expected_dims = ['chain', 'draw', 'individual']
        if (len(dims) == 2):
            expected_dims = ['chain', 'draw']
        for dim in expected_dims:
            if dim not in dims:
                raise ValueError(
                    'The data must have the dimensions '
                    '(chain, draw, individual). The current dimensions are <'
                    + str(dims) + '>.')

        # Get a colours
        colors = plotly.colors.qualitative.Plotly

        # Create a template figure (assigns it to self._fig)
        try:
            n_ids = len(data.individual)
        except AttributeError:
            # If data does not have individual attribute, all parameters
            # are population parameters
            n_ids = 1
        self._create_template_figure(
            rows=1, cols=n_ids, x_title='Normalised counts', spacing=0.01)

        # Create one figure for each parameter
        figs = []
        parameters = list(data.data_vars.keys())
        for parameter in parameters:
            # Check that parameter has as many ids as columns in template
            # figure
            try:
                number_ids = len(data[parameter].individual)
            except AttributeError:
                # If parameter does not have individual attribute, it's a
                # population parameter
                number_ids = 1

            if number_ids != n_ids:
                # Overwrite old n_ids
                n_ids = number_ids

                # Create a new template
                self._create_template_figure(
                    rows=1, cols=n_ids, x_title='Normalised counts',
                    spacing=0.01)

            # Append a copy of the template figure to all figures
            figs.append(copy.copy(self._fig))

        self._figs = figs

        # Add samples to parameter figures
        for index, parameter in enumerate(parameters):
            # Set y label of plot to parameter name
            self._figs[index].update_yaxes(
                title_text=parameter, row=1, col=1)

            # Get estimates for this parameter
            samples = data[parameter]

            # Add marginal histograms for all individuals
            self._add_histogram_plots(index, samples, colors)
