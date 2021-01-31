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


class TracePlot(eplt.MultiSubplotFigure):
    """
    #TODO:

    .. note::
        Needs samples to be ordered according to their sample iteration,
        e.g as returned by the :meth:`SamplingController.run`.
    """
    def __init__(self):
        super(TracePlot, self).__init__()

    def _add_parameter_trace_plots(self, fig_id, data, colors):
        """
        Adds trace plots of the parameter samples for each individual to
        the figure.

        One figure contains only the traces of one parameter
        (across all individuals).
        """
        # Add figure trace for each individual
        ids = data[self._id_key].unique()
        for index, individual in enumerate(ids):
            # Get individual data
            mask = data[self._id_key] == individual
            samples = data[
                [self._sample_key, self._iter_key, self._run_key]][mask]

            # Thin samples
            if self._thinning_ratio > 1:
                mask = samples[self._iter_key] % self._thinning_ratio == 0
                samples = samples[mask]

            # Add parameter traces for invdividual
            self._add_traces_for_individual(
                fig_id, index, individual, samples, colors)

        # Shade warm up iterations grey
        if self._warm_up_iter > 0:
            fig = self._figs[fig_id]
            fig.add_hrect(
                y0=0, y1=self._warm_up_iter, line_width=0, fillcolor='grey',
                opacity=0.5)

    def _add_traces_for_individual(
            self, fig_id, index, individual, samples, colors):
        """
        Adds all traces for an individual for the given parameter.

        Essentially fills up the column in a figure dedicated to the
        individual identified by the fig_id.
        """
        # Compute diagnostics
        diagnostics = self._compute_diagnostics(samples)
        rhat = diagnostics['Rhat']

        # Get number of colours (one colour for each trace)
        runs = samples[self._run_key].unique()
        n_colors = len(runs)

        # Thin traces for visualisation
        mask = samples[self._run_key] == runs[0]
        n_iterations = len(samples[mask])
        thinning_ratio = n_iterations // self._vis_max

        # Add each trace plot to figure
        for run_id, run in enumerate(runs):
            # Get chain samples
            mask = samples[self._run_key] == run
            trace = samples[mask][[self._sample_key, self._iter_key]]

            # Add trace
            color = colors[run_id % n_colors]
            ess = diagnostics['ESS for each chain'][run_id]
            self._add_trace(
                fig_id, index, individual, run, trace, thinning_ratio, rhat,
                ess, color)

    def _add_trace(
            self, fig_id, index, individual, run, trace, thinning_ratio,
            rhat, ess, color):
        """
        Adds a line plot of an indiviudals parameter trace to the figure.
        """
        # Get figure
        fig = self._figs[fig_id]

        # Apply thinning
        samples = trace[self._sample_key].to_numpy()
        iterations = trace[self._iter_key].to_numpy()
        if thinning_ratio > 1:
            samples = trace[self._sample_key].to_numpy()[::thinning_ratio]
            iterations = trace[self._iter_key].to_numpy()[::thinning_ratio]

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=samples,
                y=iterations,
                name='Run %s' % str(run),
                legendgroup='Run %s' % str(run),
                hovertemplate=(
                    'Sample: %{x:.2f}<br>' +
                    'Iteration: %{y:d}<br>' +
                    'Rhat: %.02f<br>' % rhat +
                    'Effective sample size: %.02f<br>' % ess),
                visible=True,
                showlegend=True if index == 0 else False,
                marker=dict(color=color),
                opacity=0.8),
            row=1,
            col=index+1)

        # Turn off xaxis ticks
        fig.update_xaxes(
            tickvals=[],
            row=1,
            col=index+1)

        # Set x axis title
        fig.update_xaxes(title='%s' % str(individual), col=index+1)

    def _compute_diagnostics(self, data):
        """
        Computes and returns convergence metrics.

        - Rhat
        - ESS
        """
        # Exclude warm up iterations
        mask = data[self._iter_key] > self._warm_up_iter
        data = data[mask]

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

        # Compute effective sample size of each chain
        ess = []
        for chain in container:
            # TODO: Temporary because pints wants 2 dim array
            chain = np.expand_dims(chain, 1)
            chain_ess = pints.effective_sample_size(chain)[0]
            ess.append(chain_ess)

        # Collect diagnostics
        diagnostics = {
            'Rhat': rhat,
            'ESS for each chain': ess}

        return diagnostics

    def add_data(
            self, data, warm_up_iter=0, thinning_ratio=1, id_key='ID',
            param_key='Parameter', sample_key='Sample', iter_key='Iteration',
            run_key='Run', vis_max=500):
        """
        #TODO:
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
        self._warm_up_iter = warm_up_iter
        self._thinning_ratio = thinning_ratio
        self._vis_max = vis_max

        # Get a colours
        colors = plotly.colors.qualitative.Plotly

        # Create one figure for each parameter
        figs = []
        parameters = data[param_key].unique()
        for parameter in parameters:
            # Check that parameter has as many ids as columns in template
            # figure
            mask = data[param_key] == parameter
            number_ids = len(data[mask][id_key].unique())

            # Create a new template
            self._create_template_figure(
                rows=1, cols=number_ids, y_title='Sample iteration',
                x_title=parameter, spacing=0.01)

            # Append a copy of the template figure to all figures
            figs.append(copy.copy(self._fig))

        self._figs = figs

        # Add traces to parameter figures
        for index, parameter in enumerate(parameters):
            # Get estimates for this parameter
            mask = data[param_key] == parameter
            samples = data[mask][[id_key, sample_key, iter_key, run_key]]

            # Add marginal traces for all individuals
            self._add_parameter_trace_plots(index, samples, colors)
