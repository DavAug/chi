#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import warnings

import numpy as np
import pandas as pd
import pints

import erlotinib as erlo


class _PartiallyFixedLogPosterior(pints.LogPDF):
    """
    Wrapper class for a `pints.LogPosterior` to fix the values of some
    parameters.

    This allows to reduce the parameter dimensionality of the log-posterior
    at the cost of fixing some parameters at a constant value.
    """

    def __init__(self, log_posterior, mask, values):
        super(_PartiallyFixedLogPosterior, self).__init__()

        self._log_posterior = log_posterior

        if len(mask) != self._log_posterior.n_parameters():
            raise ValueError(
                'Length of mask has to match number of log-posterior '
                'parameters.')

        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise ValueError(
                'Mask has to be a boolean array.')

        n_fixed = int(np.sum(mask))
        if n_fixed != len(values):
            raise ValueError(
                'Values has to have the same length as the number of '
                'fixed parameters.')

        # Create a parameter array for later calls of the log-posterior
        self._parameters = np.empty(shape=len(mask))
        self._parameters[mask] = np.asarray(values)

        # Update the 'free' number of parameters
        self._mask = ~mask
        self._n_parameters = int(np.sum(self._mask))

    def __call__(self, parameters):
        # Fill in 'free' parameters
        self._parameters[self._mask] = np.asarray(parameters)

        return self._log_posterior(self._parameters)

    def log_likelihood(self):
        """
        Returns log-likelihood.
        """
        return self._log_posterior.log_likelihood()

    def log_prior(self):
        """
        Returns log-prior.
        """
        return self._log_posterior.log_prior()

    def n_parameters(self):
        """
        Returns the number of 'free' parameters of the log-posterior.
        """
        return self._n_parameters


class InferenceController(object):
    """
    A base class for inference controllers.

    Parameters
    ----------

    log_posterior
        An instance of a :class:`LogPosterior` or a list of
        :class:`LogPosterior` instances. If multiple log-posteriors are
        provided, they have to be defined on the same parameters space.
    """

    def __init__(self, log_posterior):
        super(InferenceController, self).__init__()

        # Convert log-posterior to a list of log-posteriors
        try:
            log_posteriors = list(log_posterior)
        except TypeError:
            # If log-posterior cannot be converted to a list, it likely means
            # that there is only one log-posterior
            log_posteriors = [log_posterior]

        for log_posterior in log_posteriors:
            if not isinstance(log_posterior, erlo.LogPosterior):
                raise ValueError(
                    'Log-posterior has to be an instance of a '
                    '`erlotinib.LogPosterior`.')

        # Check that the log-posteriors have the same number of parameters
        n_parameters = log_posteriors[0].n_parameters()
        for log_posterior in log_posteriors:
            if log_posterior.n_parameters() != n_parameters:
                raise ValueError(
                    'All log-posteriors have to be defined on the same '
                    'parameter space.')

        self._log_posteriors = log_posteriors
        self._log_prior = self._log_posteriors[0].log_prior()

        # Set defaults
        self._n_runs = 10
        self._parallel_evaluation = True
        self._transform = None

        # Sample initial parameters from log-prior
        self._initial_params = self._log_prior.sample(self._n_runs)

        # Get parameter names
        self._all_params = np.array(
            self._log_posteriors[0].get_parameter_names())
        self._parameters = list(self._all_params)

        # Get number of parameters
        self._n_parameters = self._log_posteriors[0].n_parameters()

    def fix_parameters(self, mask, values):
        """
        Fixes the value of a subset of log-posterior parameters for the
        inference.

        The fixed parameters are fixed during the inference which will
        reduce the dimensionality of the search space at the cost of excluding
        those parameters from the inference.
        """
        # If some parameters have been fixed already, retrieve original
        # log-posterior.
        log_posteriors = self._log_posteriors
        for index, log_posterior in enumerate(log_posteriors):
            if isinstance(log_posterior, _PartiallyFixedLogPosterior):
                log_posteriors[index] = log_posterior._log_posterior

        # Fix parameters
        for index, log_posterior in enumerate(log_posteriors):
            self._log_posteriors[index] = _PartiallyFixedLogPosterior(
                log_posterior, mask, values)

        # Set transform to None and raise a warning that fixing parameters
        # reverts any previously applied transformations.
        if self._transform:
            warnings.warn(
                'Fixing parameters resets any previously applied parameter '
                'transformations.')
        self._transform = None

        # Sample new initial points
        initial_params = self._log_prior.sample(self._n_runs)

        # Keep initial points for 'free' parameters
        # Have to transpose so 1d mask can be applied to 2d initial parameters
        # of shape (n_runs, n_params)
        mask = ~np.array(mask)
        self._initial_params = initial_params.transpose()[mask].transpose()

        # Keep parameter names for 'free' parameters
        self._parameters = list(self._all_params[mask])

    def set_n_runs(self, n_runs):
        """
        Sets the number of times the inference routine is run.

        Each run starts from a random sample of the log-prior.
        """
        self._n_runs = int(n_runs)

        # Sample new initial points
        initial_params = self._log_prior.sample(self._n_runs)

        # Mask samples if some parameters have been fixed
        # I.e. they are not used for the optimisation, and therefore
        # no intitial parameters are required.

        # Get one log-posterior, because all log-posterior are treated the same
        log_posterior = self._log_posteriors[0]
        if isinstance(log_posterior, _PartiallyFixedLogPosterior):
            mask = log_posterior._mask
            # Have to transpose so 1d mask can be applied to 2d initial
            # parameters of shape (n_runs, n_params)
            initial_params = initial_params.transpose()[mask].transpose()

        self._initial_params = initial_params

    def set_parallel_evaluation(self, run_in_parallel):
        """
        Enables or disables parallel evaluation using either a
        :class:`pints.ParallelEvaluator` or a
        :class:`pints.SequentialEvaluator`.

        If ``run_in_parallel=True``, the method will run using a number of
        worker processes equal to the detected CPU core count. The number of
        workers can be set explicitly by setting ``run_in_parallel`` to an
        integer greater than ``0``. Parallelisation can be disabled by setting
        ``run_in_parallel`` to ``0`` or ``False``.
        """
        if not isinstance(run_in_parallel, (bool, int)):
            raise ValueError(
                '`run_in_parallel` has to a boolean or an integer.')
        if run_in_parallel < 0:
            raise ValueError(
                '`run_in_parallel` cannot be negative.')

        self._parallel_evaluation = run_in_parallel

    def set_transform(self, transform):
        """
        Sets the transformation that transforms the parameter space into the
        search space.

        Transformations of the search space can significantly improve the
        performance of the inference routine.

        ``transform`` has to be an instance of `pints.Transformation` and must
        have the same dimension as the parameter space.
        """
        if not isinstance(transform, pints.Transformation):
            raise ValueError(
                'Transform has to be an instance of `pints.Transformation`.')
        if transform.n_parameters() != self._n_parameters:
            raise ValueError(
                'The dimensionality of the transform does not match the '
                'dimensionality of the log-posterior.')
        self._transform = transform


class OptimisationController(InferenceController):
    """
    Sets up an optimisation routine that attempts to find the parameter values
    that maximise a :class:`pints.LogPosterior`.

    By default the optimisation is run 10 times from different initial
    starting points. Starting points are randomly sampled from the
    specified :class:`pints.LogPrior`. The optimisation is run by default in
    parallel using :class:`pints.ParallelEvaluator`.

    Extends :class:`InferenceController`.
    """

    def __init__(self, log_posterior):
        super(OptimisationController, self).__init__(log_posterior)

        # Set default optimiser
        self._optimiser = pints.CMAES

    def run(
            self, n_max_iterations=None, log_to_screen=False):
        """
        Runs the optimisation and returns the maximum a posteriori probability
        parameter estimates in from of a :class:`pandas.DataFrame` with the
        columns 'Parameter', 'Estimate', 'Score' and 'Run'.

        The number of maximal iterations of the optimisation routine can be
        limited by setting ``n_max_iterations`` to a finite, non-negative
        integer value.
        """

        # Initialise result dataframe
        result = pd.DataFrame(
            columns=['ID', 'Parameter', 'Estimate', 'Score', 'Run'])

        # Initialise intermediate container for individual runs
        run_result = pd.DataFrame(
            columns=['Parameter', 'Estimate', 'Score', 'Run'])
        run_result['Parameter'] = self._parameters

        # Get posterior
        for log_posterior in self._log_posteriors:
            individual_result = pd.DataFrame(
                columns=['ID', 'Parameter', 'Estimate', 'Score', 'Run'])

            # Run optimisation multiple times
            for run_id, init_p in enumerate(self._initial_params):
                opt = pints.OptimisationController(
                    function=log_posterior,
                    x0=init_p,
                    method=self._optimiser,
                    transform=self._transform)

                # Configure optimisation routine
                opt.set_log_to_screen(log_to_screen)
                opt.set_max_iterations(iterations=n_max_iterations)
                opt.set_parallel(self._parallel_evaluation)

                # Find optimal parameters
                try:
                    estimates, score = opt.run()
                except Exception:
                    # If inference breaks fill estimates with nan
                    estimates = [np.nan] * self._n_parameters
                    score = np.nan

                # Save estimates and score of runs
                run_result['Estimate'] = estimates
                run_result['Score'] = score
                run_result['Run'] = run_id + 1
                individual_result = individual_result.append(run_result)

            # Save runs for individual
            individual_result['ID'] = log_posterior.get_id()
            result = result.append(individual_result)

        return result

    def set_optimiser(self, optimiser):
        """
        Sets method that is used to find the maximum a posteiori probability
        estimates.
        """
        if not issubclass(optimiser, pints.Optimiser):
            raise ValueError(
                'Optimiser has to be a `pints.Optimiser`.'
            )
        self._optimiser = optimiser


class SamplingController(InferenceController):
    """
    Sets up a sampling routine that attempts to find the posterior
    distribution of parameters defined by a :class:`pints.LogPosterior`.

    By default the sampling is run 10 times from different initial
    starting points. Starting points are randomly sampled from the
    specified :class:`pints.LogPrior`. The optimisation is run by default in
    parallel using :class:`pints.ParallelEvaluator`.

    Extends :class:`InferenceController`.
    """

    def __init__(self, log_posterior):
        super(SamplingController, self).__init__(log_posterior)

        # Set default sampler
        self._sampler = pints.HaarioACMC

    def run(self, n_iterations=10000):
        """
        Runs the sampling routine and returns the sampled parameter values in
        form of a :class:`pandas.DataFrame` with columns 'Parameter', 'Sample',
        'Iteration' and 'Run'.

        The number of iterations of the sampling routine can be set by setting
        ``n_iterations`` to a finite, non-negative integer value. By default
        the routines run for 10000 iterations.

        Parameters
        ----------
        n_iterations
            A non-negative integer number which sets the number of iterations
            of each MCMC run.
        """
        # Set up sampler
        sampler = pints.MCMCController(
            log_pdf=self._log_posterior,
            chains=self._n_runs,
            x0=self._initial_params,
            method=self._sampler,
            transform=self._transform)

        # Configure sampling routine
        sampler.set_log_to_screen(True)
        sampler.set_log_interval(iters=500, warm_up=3)
        sampler.set_max_iterations(iterations=n_iterations)
        sampler.set_parallel(self._parallel_evaluation)

        # Run sampling routine
        output = sampler.run()

        # Initialise result dataframe
        result = pd.DataFrame(
            columns=['Parameter', 'Sample', 'Iteration', 'Run'])

        # Initialise intermediate container for individual runs
        container = pd.DataFrame(
            columns=['Parameter', 'Sample', 'Iteration', 'Run'])

        # Safe sample results to dataframe
        for run_id, samples in enumerate(output):
            container['Iteration'] = np.arange(
                start=1, stop=len(samples)+1)
            container['Run'] = run_id + 1

            for param_id, name in enumerate(self._parameters):
                container['Parameter'] = name
                container['Sample'] = samples[:, param_id]

                result = result.append(container)

        return result

    def set_initial_parameters(
            self, data, param_key='Parameter', est_key='Estimate',
            score_key='Score', run_key='Run'):
        """
        Sets the initial parameter values of the MCMC runs to the parameter set
        with the maximal a posteriori probability across a number of parameter
        sets.

        This method is intended to use in conjunction with the results of the
        :class:`OptimisationController`.

        It expects a :class:`pandas.DataFrame` with the columns 'Parameter',
        'Estimate', 'Score' and 'Run'. The maximum a posteriori probability
        values across all estimates is determined and used as initial point
        for the MCMC runs.

        If multiple parameter sets assume the maximal a posteriori probability
        value, a parameter set is drawn randomly from them.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the parameter estimates in form of
            a parameter, estimate and score column.
        param_key
            Key label of the :class:`DataFrame` which specifies the parameter
            name column. Defaults to ``'Parameter'``.
        est_key
            Key label of the :class:`DataFrame` which specifies the parameter
            estimate column. Defaults to ``'Estimate'``.
        score_key
            Key label of the :class:`DataFrame` which specifies the score
            estimate column. The score refers to the maximum a posteriori
            probability associated with the estimate. Defaults to ``'Score'``.
        run_key
            Key label of the :class:`DataFrame` which specifies the
            optimisation run column. Defaults to ``'Run'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [param_key, est_key, score_key, run_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        parameters = data[param_key].unique()
        for param in parameters:
            if param not in self._parameters:
                warnings.warn(
                    'The parameter <' + str(param) + '> could not be '
                    'associated with a non-fixed model parameter, and was '
                    'therefore not set.')

        # Get estimates with maximum a posteriori probability
        max_prob = data[score_key].max()
        mask = data[score_key] == max_prob
        data = data[mask]

        # Find a unique set of parameter values
        runs = data[run_key].unique()
        selected_param_set = np.random.choice(runs)
        mask = data[run_key] == selected_param_set
        data = data[mask]

        # Set initial parameters to map estimates
        for param in parameters:
            # Get estimate
            mask = data[param_key] == param
            map_estimate = data[est_key][mask].to_numpy()

            # Set initial value to map estimate for all runs
            mask = self._parameters == param
            self._initial_params[:, mask] = map_estimate

    def set_sampler(self, sampler):
        """
        Sets method that is used to sample from the log-posterior.
        """
        if not issubclass(sampler, pints.MCMCSampler):
            raise ValueError(
                'Sampler has to be a `pints.MCMCSampler`.'
            )
        self._sampler = sampler
