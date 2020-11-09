#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pandas as pd
import pints


class OptimisationController(object):
    """
    Attempts to find the parameter values that maximise a `pints.LogPosterior`.

    By default the optimisation is run 10 times from random points drawn from
    the `pints.LogPrior`.
    """

    def __init__(self, log_posterior, optimiser):
        super(OptimisationController, self).__init__()

        if not isinstance(log_posterior, pints.LogPosterior):
            raise ValueError(
                'Log-posterior has to be an instance of `pints.LogPosterior`.'
            )
        self._log_posterior = log_posterior

        if not isinstance(optimiser, pints.Optimiser):
            raise ValueError(
                'Optimiser has to be an instance of `pints.Optimiser`.'
            )
        self._optimiser = optimiser

        # Set defaults
        self._n_runs = 10
        self._transform = None

        # Sample initial parameters from log-prior
        log_prior = log_posterior.log_prior()
        self._initial_params = log_prior.sample(self._n_runs)

        # Get parameter names
        self._parameters = self._log_posterior._model.parameters()

    def fix_parameters(self, mask, values):
        """
        Fixes the value of a subset of log-posterior parameters for the
        optimisation.

        The fixed parameters are fixed during the optimisation which will
        reduce the dimensionality of the search at the cost of excluding
        the fixed parameters from the optimisation.
        """
        # If some parameters have been fixed already, retrieve original
        # log-posterior.
        log_posterior = self._log_posterior
        if isinstance(log_posterior, _PartiallyFixedLogPosterior):
            log_posterior = log_posterior.log_posterior()

        # Fix parameters
        self._log_posterior = _PartiallyFixedLogPosterior(
            log_posterior, mask, values)

        # Set transform to None and raise a warning that fixing parameters
        # reverts any previously applied transformations.
        if self._transform:
            raise Warning(
                'Fixing parameters resets any previously applied parameter '
                'transformations.')
        self._transform = None

        # Mask initial points
        # Have to transpose so 1d mask can be applied to 2d initial parameters
        # of shape (n_runs, n_params)
        mask = self._log_posterior._mask
        transposed = self._initial_params.transpose()
        self._initial_params = transposed[mask].transpose()

        # Mask parameter names
        parameters = log_posterior._model.parameters()
        self._parameters = np.array(parameters)[mask]

    def set_transform(self, transform):
        """
        Sets the transformation from the parameter space to the search space
        that is used for the optimisation.

        Transform has to be an instance of `pints.Transformation` and must have
        the same dimension as the search space.
        """
        if not isinstance(transform, pints.Transformation):
            raise ValueError(
                'Transform has to be an instance of `pints.Transformation`.')
        if transform.n_parameters() != self._log_posterior.n_parameters():
            raise ValueError(
                'The dimensionality of the transform does not match the '
                'dimensionality of the log-posterior.')
        self._transform = transform

    def set_n_runs(self, n_runs):
        """
        Sets how often the optimisation routine is run. Each run starts from a
        random sample of the log-prior.
        """
        self._n_runs = int(n_runs)

        # Sample new initial points
        log_prior = self._log_posterior.log_prior()
        initial_params = log_prior.sample(self._n_runs)

        # Mask samples if some parameters have been fixed
        # I.e. they are not used for the optimisation, and therefore
        # no intitial parameters are required.
        if isinstance(self._log_posterior, _PartiallyFixedLogPosterior):
            mask = self._log_posterior._mask
            # Have to transpose so 1d mask can be applied to 2d initial
            # parameters of shape (n_runs, n_params)
            initial_params = initial_params.transpose()[mask].transpose()

        self._initial_params = initial_params

    def run(self):
        """
        Runs the optimisation and returns the maximum a posteriori probability
        parameter estimates.
        """
        # Initialise result dataframe
        result = pd.DataFrame(
            columns=['Parameter', 'Estimate', 'Score', 'Run'])

        # Initialise intermediate container for individual runs
        container = pd.DataFrame(
            columns=['Parameter', 'Estimate', 'Score', 'Run'])
        container['Parameter'] = self._parameters
        n_parameters = self._log_posterior.n_parameters()

        # Run optimisation multiple times
        for run_id, init_p in enumerate(self._initial_params):
            opt = pints.OptimisationController(
                function=self._log_posterior,
                x0=init_p,
                method=self._optimiser,
                transform=self._transform)

            # Configure optimisation routine
            opt.set_log_to_screen(False)
            opt.set_parallel(True)

            # Find optimal parameters
            try:
                estimates, score = opt.run()
            except Exception:
                # If inference breaks fill estimates with nan
                estimates = [np.nan] * n_parameters
                score = np.nan

            # Save estimates and score
            container['Estimate'] = estimates
            container['Score'] = [score] * n_parameters
            container['Run'] = [run_id] * n_parameters
            result = result.append(container)

        return result


class _PartiallyFixedLogPosterior(pints.LogPDF):
    """
    Wrapper class for a `pints.LogPosterior` to fix the values of some
    parameters.

    This allows to reduce the parameter dimensionality of the log-posterior
    at the cost of fixing some parameters at a constant value.
    """

    def __init__(self, log_posterior, mask, values):
        super(_PartiallyFixedLogPosterior, self).__init__()

        if not isinstance(log_posterior, pints.LogPosterior):
            raise ValueError(
                'Log-posterior has to be an instance of `pints.LogPosterior`.'
            )
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

    def evaluateS1(self, parameters):
        raise NotImplementedError

    def log_posterior(self):
        """
        Returns log-posterior.
        """
        return self._log_posterior

    def n_parameters(self):
        """
        Returns the number of 'free' parameters of the log-posterior.
        """
        return self._n_parameters
