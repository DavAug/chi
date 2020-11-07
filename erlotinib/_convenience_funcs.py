#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pints


def optimise(
        objective_function, optimiser, initial_params, n_runs=1,
        boundaries=None, max_iterations=None):
    """
    A wrapper around a pints.OptimisationController that returns optimal
    parameters.
    """
    if not isinstance(objective_function, (pints.ErrorMeasure, pints.LogPDF)):
        raise ValueError(
            'Objective function has to be an instance of `pints.ErrorMeasure` '
            'or `pints.LogPDF`.')
    if not issubclass(optimiser, pints.Optimiser):
        raise ValueError(
            'Optimiser has to be an instance of `pints.Optimiser`.')

    n_parameters = objective_function.n_parameters()
    initial_params = np.asarray(initial_params)
    if initial_params.shape != (n_runs, n_parameters):
        raise ValueError(
            'Initial parameters has the wrong shape! Expected shape = '
            '(%d, %d).' % (n_runs, n_parameters))
    if boundaries is not None:
        if not isinstance(boundaries, pints.Boundaries):
            raise ValueError(
                'Boundaries have to be an instance of `pints.Bouandries`.')

    # Define container for estimates and scores
    parameters = np.empty(shape=(n_runs, n_parameters))
    scores = np.empty(shape=n_runs)

    # Run optimisation multiple times
    for run_id, init_p in enumerate(initial_params):
        opt = pints.OptimisationController(
            function=objective_function,
            x0=init_p,
            method=optimiser,
            boundaries=boundaries)

        # Configure optimisation routine
        opt.set_log_to_screen(False)
        opt.set_parallel(True)
        if max_iterations:
            opt.set_max_iterations(max_iterations)

        # Find optimal parameters
        try:
            estimates, score = opt.run()
        except Exception:
            # If inference breaks fill estimates with nan
            estimates = np.array([np.nan] * n_parameters)
            score = np.nan

        # Save estimates and score
        parameters[run_id, :] = estimates
        scores[run_id] = score

    return parameters, scores
