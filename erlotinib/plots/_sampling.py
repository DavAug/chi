#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy

import numpy as np
import pandas as pd
import pints
import plotly.colors
import plotly.graph_objects as go

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

        # Add trace for each individual
        ids = data[self._id_key].unique()
        for index, individual in enumerate(ids):
            # Get individual data
            mask = data[self._id_key] == individual
            samples = data[
                [self._sample_key, self._iter_key, self._run_key]][mask]

            # Compute diagnostics
            diagnostics = self._compute_diagnostics(samples)

            # Add trace
            color = colors[index % n_colors]
            samples = samples[self._sample_key]
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

    def _compute_diagnostics(self, data):
        """
        Computes and returns convergence metrics.

        - Rhat
        """
        # Compute Rhat
        # Reshape data into shape needed for pints.rhat
        n_iterations = len(data[self._iter_key].unique())
        runs = data[self._run_key].unique()
        n_runs = len(runs)

        container = np.empty(shape=(n_runs, n_iterations))
        for index, run in enumerate(runs):
            mask = data[self._run_key] == run
            container[index, :] = data[self._sample_key][mask].to_numpy()

        # Compute rhat
        rhat = pints.rhat(chains=container)

        # Collect diagnostics
        diagnostics = [rhat]

        return diagnostics

    def add_data(
            self, data, warm_up_iter=0, id_key='ID', param_key='Parameter',
            sample_key='Sample', iter_key='Iteration', run_key='Run'):
        """
        Adds marginal histograms of the samples across runs to the figure. The
        estimates are grouped by the individual ID.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the parameter samples in form of
            an ID, parameter, sample, iteration, and run column.
        warm_up_iter
            Number of warm up iterations which are excluded for the histogram
            construction.
        id_key
            Key label of the :class:`DataFrame` which specifies the ID column.
            The ID refers to the identity of an individual. Defaults to
            ``'ID'``.
        param_key
            Key label of the :class:`DataFrame` which specifies the parameter
            name column. Defaults to ``'Parameter'``.
        sample_key
            Key label of the :class:`DataFrame` which specifies the parameter
            sample column. Defaults to ``'Sample'``.
        iter_key
            Key label of the :class:`DataFrame` which specifies the iteration
            column. The iteration refers to the iteration of the sampling
            routine at which the parameter value was sampled. Defaults to
            ``'Iteration'``.
        run_key
            Key label of the :class:`DataFrame` which specifies the
            sample run (or chain) column. Defaults to ``'Run'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        keys = [param_key, id_key, sample_key, iter_key, run_key]
        for key in keys:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')
        self._id_key, self._sample_key, self._iter_key, self._run_key = keys[
            1:]

        if warm_up_iter >= data[self._iter_key].max():
            raise ValueError(
                'The number of warm up iterations has to be smaller than the '
                'total number of iterations for each run.')

        # Get a colours
        colors = plotly.colors.qualitative.Plotly

        # Create a template figure (assigns it to self._fig)
        n_ids = len(data[id_key].unique())
        self._create_template_figure(
            rows=1, cols=n_ids, x_title='Normalised counts', spacing=0.01)

        # Create one figure for each parameter
        figs = []
        parameters = data[param_key].unique()
        for parameter in parameters:
            # Check that parameter has as many ids as columns in template
            # figure
            mask = data[param_key] == parameter
            number_ids = len(data[mask][id_key].unique())
            if number_ids != n_ids:
                # Create a new template
                self._create_template_figure(
                    rows=1, cols=number_ids, x_title='Normalised counts',
                    spacing=0.01)

            # Append a copy of the template figure to all figures
            figs.append(copy.copy(self._fig))

        self._figs = figs

        # Exclude warm up iterations
        mask = data[self._iter_key] > warm_up_iter
        data = data[mask]

        # Add estimates to parameter figures
        for index, parameter in enumerate(parameters):
            # Set y label of plot to parameter name
            self._figs[index].update_yaxes(
                title_text=parameter, row=1, col=1)

            # Get estimates for this parameter
            mask = data[param_key] == parameter
            samples = data[mask][[id_key, sample_key, iter_key, run_key]]

            # Add marginal histograms for all individuals
            self._add_histogram_plots(index, samples, colors)
