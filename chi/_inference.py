#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import arviz as az
import numpy as np
import pandas as pd
import pints
from tqdm.notebook import tqdm
import xarray as xr

import chi


# TODO: Revisit pointwise log-likelihood evaluation for population models
def _compute_hierarchical_pointwise_log_likelihood(
        log_likelihood, posterior, chain_coords, draw_coords,
        per_individual, show_chain_progress_bar):  # pragma: no cover
    """
    Computes the pointwise log-likelihoods for each sample in the
    posterior for a hierarchical model.

    If per individual is ``True`` the pointwise log-likelihoods are
    aggregated per individual. Otherwise the pointwise log-likelihoods
    are computed for each observation.
    """
    # Create container for pointwise log-likelihoods
    # (either per individual or per observation)
    n_chains, n_draws, _ = posterior.shape
    if per_individual is True:
        n_ids = log_likelihood.n_log_likelihoods()
        pointwise_ll = np.empty(shape=(n_chains, n_draws, n_ids))
    else:
        n_obs = np.sum(log_likelihood.n_observations())
        pointwise_ll = np.empty(shape=(n_chains, n_draws, n_obs))

    for chain_id, chain in enumerate(tqdm(
            posterior, disable=not show_chain_progress_bar)):
        for draw_id, draw in enumerate(chain):
            # Compute pointwise log-likelihood for the
            # given iteration and chain
            pointwise_ll[chain_id, draw_id, :] = \
                log_likelihood.compute_pointwise_ll(
                    draw, per_individual)

    # Annotate as xarray
    individual = log_likelihood.get_id(individual_ids=True)
    if per_individual:
        pointwise_ll = xr.DataArray(
            data=pointwise_ll,
            coords={
                'chain': chain_coords,
                'draw': draw_coords,
                'individual': individual},
            dims=('chain', 'draw', 'individual'))
    else:
        obs_coords = []
        for index, n_obs in enumerate(log_likelihood.n_observations()):
            _id = individual[index]
            obs_coords += [
                '%s Observation %d' % (_id, obs) for obs in range(1, n_obs+1)]
        pointwise_ll = xr.DataArray(
            data=pointwise_ll,
            coords={
                'chain': chain_coords,
                'draw': draw_coords,
                'observation': obs_coords},
            dims=('chain', 'draw', 'observation'))

    return pointwise_ll


def _compute_individual_pointwise_log_likelihood(
        log_likelihood, posterior, chain_coords, draw_coords,
        show_chain_progress_bar):  # pragma: no cover
    """
    Computes the pointwise log-likelihoods for each sample in the
    posterior for a non-hierarchical model.
    """
    # Create container for pointwise log-likelihoods
    # (either per individual or per observation)
    n_obs = np.sum(log_likelihood.n_observations())
    n_chains, n_draws, _ = posterior.shape
    pointwise_ll = np.empty(shape=(n_chains, n_draws, n_obs))

    for chain_id, chain in enumerate(tqdm(
            posterior, disable=not show_chain_progress_bar)):
        for draw_id, draw in enumerate(chain):
            # Compute pointwise log-likelihood for the
            # given iteration and chain
            pointwise_ll[chain_id, draw_id, :] = \
                log_likelihood.compute_pointwise_ll(draw)

    # Annotate as xarray
    obs_coords = []
    for out_id, n_obs in enumerate(log_likelihood.n_observations()):
        output_id = out_id + 1
        obs_coords += [
            'Output %d Observation %d' % (output_id, obs)
            for obs in range(1, n_obs+1)]
    pointwise_ll = xr.DataArray(
        data=pointwise_ll,
        coords={
            'chain': chain_coords,
            'draw': draw_coords,
            'observation': obs_coords},
        dims=('chain', 'draw', 'observation'))

    return pointwise_ll


def _check_parameters(
        log_likelihood, posterior_samples, param_map,
        is_hierarchical):  # pragma: no cover
    """
    Checks that all log-likelihood parameters can be found in the
    posterior.

    For individual log-likelihoods the parameter names must coincide
    with variables of the posterior xr.Dataset, or the names can be
    mapped witg the param_map.

    For hierarchical log-likelihoods the top-level parameters can
    be equally identified. However the bottom level parameters are
    more complicated:

    The posterior xr.Dataset will have a variable <parameter name>
    which has the individuals' IDs as a dimension. So to check the
    existence of the bottom parameters we first check for the
    parameter name and then the ID.
    """
    # Create map from posterior parameter names to model parameter names
    # (This map applies equally to top and bottom parameters - IDs excluded)
    model_names = log_likelihood.get_parameter_names()
    for param_id, name in enumerate(model_names):
        try:
            model_names[param_id] = param_map[name]
        except KeyError:
            # The name is not mapped
            pass

    # Check that model names are in the dataset
    for parameter in model_names:
        if parameter not in posterior_samples.data_vars:
            raise ValueError(
                'The parameter <' + str(parameter) + '> cannot be found '
                'in the posterior.')

    # Also get the mapped top-parameters
    # (For non-hierarchical models top_parameters are the same as
    # model_names)
    top_parameters = model_names
    if is_hierarchical is True:
        top_parameters = log_likelihood.get_parameter_names(
            exclude_bottom_level=True)

    # Filter heterogenous parameters from top-level parameters
    # (In posterior Dataset they are treated as bottom level parameters)
    top_parameters = _filter_top_parameters(top_parameters)

    # For a hierarchical log-likelihood make sure that bottom parameters
    # also exist for all IDs
    if is_hierarchical is True:
        ids = log_likelihood.get_id(individual_ids=True)
        for parameter in model_names:
            # Skip if parameter is top level
            if parameter in top_parameters:
                continue

            # Check that IDs exist
            param_ids = posterior_samples[parameter].individual
            for _id in ids:
                if _id not in param_ids:
                    raise ValueError(
                        'The ID <' + str(_id) + '> does not exist in the '
                        'posterior for parameter <' + str(parameter) + '>.')

    return model_names, top_parameters


def _filter_top_parameters(top_parameters):  # pragma: no cover
    """
    Filter heterogenously modelled top parameters from the list.

    For the formatting of the hierarchical posterior samples we need to
    distinguish between top and bottom level parameters.

    Top parameters are parameters of the population model, bottom parameters
    are parameters of the individuals. However, top and bottom level
    parameters are not mutually exclusive, since for a HeterogeneousModel
    the individual parameters are also top level parameters.

    In the posterior Dataset parameters that are associated to individuals
    have an individual coordinate, while parameters that are associated to
    the population do not have an individual coordinate.

    Almost all top-level parameters will fall into the class of parameters
    without individual coordinates, except HeterogenousModel parameters.
    Hence, we filter them from the top-level parameter list, such that they
    are treated as individual parameters for the formatting.
    """
    top_params = []
    top_parameters = top_parameters.copy()
    while top_parameters:
        # Keep parameter if it only occurs once
        # (else it must be a heterogeneously modelled parameter)
        parameter = top_parameters.pop(0)
        if parameter not in top_parameters:
            top_params.append(parameter)

        # Remove all remaining occurences
        top_parameters = [p for p in top_parameters if p != parameter]

    return top_params


def _format_posterior(
        parameter_names, top_parameters, posterior_samples,
        ids):  # pragma: no cover
    """
    Formats the dataset into a numpy array of shape
    (n_iterations, n_parameters), such that looping through the iterations
    can be done more efficiently.
    """
    n_chains = len(posterior_samples.chain)
    n_parameters = len(parameter_names)
    n_draws = len(posterior_samples.dropna(dim='draw').draw)

    # Collect samples
    id_index = 0
    n_ids = len(ids) if ids is not None else 1
    posterior = np.empty(shape=(n_chains, n_draws, n_parameters))
    for param_id, parameter in enumerate(parameter_names):
        if parameter in top_parameters:
            try:
                # Assumes values have shape (n_chains, n_draws)
                posterior[:, :, param_id] = \
                    posterior_samples[parameter].values
            except ValueError:
                # If broadcasting fails, values must be transposed
                posterior[:, :, param_id] = \
                    posterior_samples[parameter].values.T

            # Continue to next parameter
            continue  # pragma: no cover

        # Get parameters of individual
        _id = ids[id_index % n_ids]
        try:
            # Assumes values have shape (n_chains, n_draws)
            posterior[:, :, param_id] = \
                posterior_samples[parameter].sel(individual=_id).values
        except ValueError:
            # If broadcasting fails, values must be transposed
            posterior[:, :, param_id] = \
                posterior_samples[parameter].sel(individual=_id).values.T

        # Shift id index to next individual
        # (This crucially relies on the fact that individual parameters of
        # one type are grouped together, i.e. we iterate first through all
        # individuals before moving on to the next parameter.)
        id_index += 1

    return posterior


def compute_pointwise_loglikelihood(
        log_likelihood, posterior_samples, individual=None, param_map=None,
        per_individual=True, return_inference_data=False,
        show_chain_progress_bar=False):  # pragma: no cover
    """
    Computes the pointwise log-likelihood for each observation and each
    parameter sample from the posterior distribution.

    For a :class:`HierarchicalLogLikelihood` pointwise log-likelihoods are
    by default computed and aggregated per individual. If the pointwise
    log-likelihoods are supposed to be computed per observation,
    ``per_individual`` can be set to ``False``. For more info see
    :meth:`HierarchicalLogLikelihood.compute_pointwise_ll`.

    :param log_likelihood: The log-likelihood of the model parameters.
    :type log_likelihood: LogLikelihood, HierarchicalLogLikelihood
    :param posterior_samples: Samples from the posterior distribution of the
        model parameters.
    :type posterior_samples: xarray.Dataset
    :param individual: The individual for which the log-likelihoods are
        evaluated. If ``None`` the first individual is chosen.
    :type individual: str, optional
    :param param_map: A dictionary which can be used to map log-likelihood
        parameter names to the parameter names in the :class:`xarray.Dataset`.
        If ``None``, it is assumed that the names are identical. For
        hierarchical models top and bottom names can be mapped, IDs excluded.
    :type param_map: dict, optional
    :param per_individual: A boolean flag that determines whether the
            scores are computed per individual or per observation.
    :type per_individual: bool, optional
    :param return_inference_data: A boolean flag which determines whether the
        log-likelihoods and the posterior are returned as
        :class:`arviz.InferenceData`.
    :type return_inference_data: bool, optional
    :param show_chain_progress_bar: A boolean flag which determines whether the
        progress for each chain is visualised as a progress bar.
    :type show_chain_progress_bar: bool, optional
    """
    # Check inputs
    if not isinstance(
            log_likelihood,
            (chi.LogLikelihood, chi.HierarchicalLogLikelihood)):
        raise TypeError(
            'The log-likelihood must be either an instance of a '
            'chi.LogLikelihood or a '
            'chi.HierarchicalLogLikelihood.')
    if not isinstance(posterior_samples, xr.Dataset):
        raise TypeError(
            'The posterior samples must be an instance of a '
            'xarray.Dataset.')

    dims = sorted(list(posterior_samples.dims))
    expected_dims = ['chain', 'draw', 'individual']
    if (len(dims) == 2):
        expected_dims = ['chain', 'draw']
    for dim in expected_dims:
        if dim not in dims:
            raise ValueError(
                'The posterior samples must have the dimensions '
                '(chain, draw, individual). The current dimensions are <'
                + str(dims) + '>.')

    # Set default parameter map (no mapping)
    if param_map is None:
        param_map = {}

    try:
        param_map = dict(param_map)
    except (TypeError, ValueError):
        raise TypeError(
            'The parameter map has to be convertable to a python '
            'dictionary.')

    # Check whether model is hierarchical
    is_hierarchical = False
    if isinstance(log_likelihood, chi.HierarchicalLogLikelihood):
        is_hierarchical = True

    # Check individual for population model
    if is_hierarchical is True:
        if individual is not None:
            raise ValueError(
                "Individual IDs cannot be selected for a "
                "chi.HierarchicalLogLikelihood. To evaluate compute "
                "the pointwise for an individual separately, please use "
                "construct a log-likelihood for the individual alone.")

    # Check individual exists
    else:
        ids = posterior_samples.individual
        if individual is None:
            # Get default
            individual = str(ids.data[0])
        if individual not in ids:
            raise ValueError(
                'The individual <' + str(individual) + '> could not be '
                'found in the "individual" dimension.')

        # Get relevant samples and forget about individual
        posterior_samples = posterior_samples.sel(individual=individual)

    # Check parameters
    # (For individual log-likelihoods, top_parameters are identical to
    # parameter_names)
    parameter_names, top_parameters = _check_parameters(
        log_likelihood, posterior_samples, param_map, is_hierarchical)

    # Sort parameters into numpy array for simplified iteration
    ids = log_likelihood.get_id(individual_ids=True) if isinstance(
        log_likelihood, chi.HierarchicalLogLikelihood) else None
    posterior = _format_posterior(
        parameter_names, top_parameters, posterior_samples, ids)

    # Compute pointwise log-likelihoods
    chain_coords = posterior_samples.chain
    draw_coords = posterior_samples.draw
    if is_hierarchical is True:
        pointwise_ll = _compute_hierarchical_pointwise_log_likelihood(
            log_likelihood, posterior, chain_coords, draw_coords,
            per_individual, show_chain_progress_bar)
    else:
        pointwise_ll = _compute_individual_pointwise_log_likelihood(
            log_likelihood, posterior, chain_coords, draw_coords,
            show_chain_progress_bar)

    if return_inference_data is False:
        return pointwise_ll

    # Compose posterior samples and pointwise log-likelihood to
    # arviz.InferenceData
    samples_stats = xr.Dataset({'log_likelihood': pointwise_ll})
    inference_data = az.InferenceData(
        posterior=posterior_samples,
        sample_stats=samples_stats)

    return inference_data


class InferenceController(object):
    """
    A base class for inference controllers.

    :param log_posterior: The log-posterior from which is sampled.
    :type log_posterior: chi.LogPosterior, chi.HierarchicalLogPosterior
    :param seed: Seed for the random initialisation. Initial points are sampled
        from the log-prior.
    :type seed: int, optional
    """

    def __init__(self, log_posterior, seed=None):
        super(InferenceController, self).__init__()

        # Convert log-posterior to a list of log-posteriors
        if not isinstance(
                log_posterior,
                (chi.LogPosterior, chi.HierarchicalLogPosterior)):
            raise ValueError(
                'The log-posterior has to be an instance of a '
                'chi.LogPosterior or a '
                'chi.HierarchicalLogPosterior')

        # Check that the log-posteriors have the same number of parameters
        self._log_posterior = log_posterior

        # Set defaults
        self._n_runs = 5
        self._parallel_evaluation = True
        self._transform = None
        self._seed = seed

        # Sample initial parameters from log-prior
        self._initial_params = self._log_posterior.sample_initial_parameters(
            n_samples=self._n_runs, seed=self._seed)

    def set_n_runs(self, n_runs):
        """
        Sets the number of times the inference routine is run.

        Each run starts from a random sample of the log-prior.
        """
        self._n_runs = int(n_runs)

        # Sample initial parameters from log-prior
        self._initial_params = self._log_posterior.sample_initial_parameters(
            n_samples=self._n_runs, seed=self._seed)

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
        if transform.n_parameters() != self._log_posterior.n_parameters():
            raise ValueError(
                'The dimensionality of the transform does not match the '
                'dimensionality of the log-posterior.')
        self._transform = transform


class OptimisationController(InferenceController):
    """
    Sets up an optimisation routine that attempts to find the parameter values
    that maximise a :class:`pints.LogPosterior`. If multiple log-posteriors are
    provided, the posteriors are assumed to be structurally identical and only
    differ due to different data sources.

    By default the optimisation is run 5 times from different initial
    starting points. Starting points are randomly sampled from the
    specified :class:`pints.LogPrior`. The optimisation is run by default in
    parallel using :class:`pints.ParallelEvaluator`.

    Extends :class:`InferenceController`.

    :param log_posterior: The log-posterior from which is sampled.
    :type log_posterior: chi.LogPosterior, chi.HierarchicalLogPosterior
    :param seed: Seed for the random initialisation. Initial points are sampled
        from the log-prior.
    :type seed: int
    """

    def __init__(self, log_posterior, seed=None):
        super(OptimisationController, self).__init__(log_posterior, seed)

        # Set default optimiser
        self._optimiser = pints.CMAES

    def run(
            self, n_max_iterations=10000, show_run_progress_bar=False,
            log_to_screen=False):
        """
        Runs the optimisation and returns the maximum a posteriori probability
        parameter estimates in from of a :class:`pandas.DataFrame` with the
        columns 'ID', 'Parameter', 'Estimate', 'Score' and 'Run'.

        The number of maximal iterations of the optimisation routine can be
        limited by setting ``n_max_iterations`` to a finite, non-negative
        integer value.

        Parameters
        ----------

        n_max_iterations
            The maximal number of optimisation iterations to find the MAP
            estimates for each log-posterior. By default the maximal number
            of iterations is set to 10000.
        show_run_progress_bar
            A boolean flag which indicates whether a progress bar for looping
            through the optimisation runs is displayed.
        log_to_screen
            A boolean flag which indicates whether the optimiser logging output
            is displayed.
        """

        # Initialise result dataframe
        result = pd.DataFrame(
            columns=['ID', 'Parameter', 'Estimate', 'Score', 'Run'])

        # Initialise intermediate container for individual runs
        run_result = pd.DataFrame(
            columns=['ID', 'Parameter', 'Estimate', 'Score', 'Run'])
        run_result['Parameter'] = self._log_posterior.get_parameter_names()

        # Set ID of individual (or IDs of parameters, if hierarchical)
        run_result['ID'] = self._log_posterior.get_id()

        # Run optimisation multiple times
        for run_id in tqdm(
                range(self._n_runs), disable=not show_run_progress_bar):
            opt = pints.OptimisationController(
                function=self._log_posterior,
                x0=self._initial_params[run_id, :],
                method=self._optimiser,
                transformation=self._transform)

            # Configure optimisation routine
            opt.set_log_to_screen(log_to_screen)
            opt.set_max_iterations(iterations=n_max_iterations)
            opt.set_parallel(self._parallel_evaluation)

            # Find optimal parameters
            try:
                estimates, score = opt.run()
            except Exception:
                # If inference breaks fill estimates with nan
                estimates = [np.nan] * self._log_posterior.n_parameters()
                score = np.nan

            # Save estimates and score of runs
            run_result['Estimate'] = estimates
            run_result['Score'] = score
            run_result['Run'] = run_id + 1
            result = pd.concat([result, run_result])

        return result

    def set_optimiser(self, optimiser):
        """
        Sets method that is used to find the maximum a posteiori probability
        estimates.
        """
        if not issubclass(optimiser, pints.Optimiser):
            raise ValueError(
                'Optimiser has to be a `pints.Optimiser`.')
        self._optimiser = optimiser


class SamplingController(InferenceController):
    """
    Sets up a sampling routine that attempts to find the posterior
    distribution of parameters defined by a :class:`pints.LogPosterior`. If
    multiple log-posteriors are provided, the posteriors are assumed to be
    structurally identical and only differ due to different data sources.

    By default the sampling is run 5 times from different initial
    starting points. Starting points are randomly sampled from the
    specified :class:`pints.LogPrior`. The optimisation is run by default in
    parallel using :class:`pints.ParallelEvaluator`.

    Extends :class:`InferenceController`.

    :param log_posterior: The log-posterior from which is sampled.
    :type log_posterior: chi.LogPosterior, chi.HierarchicalLogPosterior
    :param seed: Seed for the random initialisation. Initial points are sampled
        from the log-prior.
    :type seed: int
    """

    def __init__(self, log_posterior, seed=None):
        super(SamplingController, self).__init__(log_posterior, seed)

        # Set default sampler
        self._sampler = pints.HaarioBardenetACMC

    def _format_chains(self, chains, divergent_iters):
        """
        Formats the chains generated by pints in shape of
        (n_chains, n_iterations, n_parameters) to a xarray.Dataset, where
        each parameter name gets an entry yielding a xarray.DataArray of
        shape (n_chains, n_iterations, n_ids) or if ID is None
        (n_chains, n_iterations).

        Note that the naming of the dimensions matter for ArviZ so we call
        the dimensions (chain, draw, individual) (the last dimension
        is not set by ArviZ).
        """
        # Get all parameter names
        names = self._log_posterior.get_parameter_names()

        # Get top-level parameter names
        top_parameters = self._log_posterior.get_parameter_names(
            exclude_bottom_level=True)

        # Get IDs of bottom-level parameters
        ids = self._log_posterior.get_id(unique=True)

        # Convert names and ids to numpy arrays
        ids = np.array(ids) if isinstance(ids, list) else ids
        names = np.array(names)

        # Get the coordinates for the chains and draws
        n_chains, n_draws, _ = chains.shape
        chain_coords = list(range(n_chains))
        draw_coords = list(range(n_draws))

        # Sort samples of parameters into xarrays
        # Start with top-level parameters (have no ID)
        container = {}
        bottom_parameters = []
        for idp, parameter in enumerate(names):
            # Check if parameter is a bottom-level parameter
            if parameter not in top_parameters:
                # Add to bottom parameter list if parameter is not already in
                # it
                if parameter not in bottom_parameters:
                    bottom_parameters.append(parameter)
                continue

            # Append top parameter samples to container
            container[parameter] = xr.DataArray(
                data=chains[:, :, idp],
                dims=['chain', 'draw'],
                coords={'chain': chain_coords, 'draw': draw_coords})

        # Add samples of bottom parameters (include IDs now)
        for idp, parameter in enumerate(bottom_parameters):
            mask = names == parameter
            container[parameter] = xr.DataArray(
                data=chains[:, :, mask],
                dims=['chain', 'draw', 'individual'],
                coords={
                    'chain': chain_coords,
                    'draw': draw_coords,
                    'individual': ids})

        # Add information about divergent iterations
        attrs = {'divergent iterations': 'false'}
        if divergent_iters:
            attrs['divergent iterations'] = 'true'
            for idx, iters in enumerate(divergent_iters):
                attrs['divergent iterations chain %d' % idx] = iters

        return xr.Dataset(container, attrs=attrs)

    def run(
            self, n_iterations=10000, hyperparameters=None,
            log_to_screen=False):
        """
        Runs the sampling routine and returns the sampled parameter values in
        form of a :class:`xarray.Dataset` with :class:`xarray.DataArray`
        instances for each parameter.

        If multiple posteriors are inferred a list of :class:`xarray.Dataset`
        instances is returned.

        The number of iterations of the sampling routine can be set by setting
        ``n_iterations`` to a finite, non-negative integer value. By default
        the routines run for 10000 iterations.

        :param n_iterations: A non-negative integer number which sets the
            number of iterations of the MCMC runs.
        :type n_iterations: int, optional
        :param hyperparameters: A list of hyperparameters for the sampling
            method. If ``None`` the default hyperparameters are set.
        :type hyperparameters: list[float], optional
        :param log_to_screen: A boolean flag which can be used to print
            the progress of the runs to the screen. The progress is printed
            every 500 iterations.
        :type log_to_screen: bool, optional
        """
        # Set up sampler
        sampler = pints.MCMCController(
            log_pdf=self._log_posterior,
            chains=self._n_runs,
            x0=self._initial_params,
            method=self._sampler,
            transformation=self._transform)

        # Configure sampling routine
        sampler.set_log_to_screen(log_to_screen)
        sampler.set_log_interval(iters=20, warm_up=3)
        sampler.set_max_iterations(iterations=n_iterations)
        sampler.set_parallel(self._parallel_evaluation)

        if hyperparameters:
            for s in sampler.samplers():
                s.set_hyper_parameters(hyperparameters)

        # Run sampling routine
        chains = sampler.run()

        # If Hamiltonian Monte Carlo, get number of divergent
        # iterations
        divergent_iters = None
        if issubclass(
                self._sampler, (pints.HamiltonianMCMC, pints.NoUTurnMCMC)):
            divergent_iters = [
                s.divergent_iterations() for s in sampler.samplers()]

        # Format chains
        chains = self._format_chains(chains, divergent_iters)

        return chains

    def set_sampler(self, sampler):
        """
        Sets method that is used to sample from the log-posterior.
        """
        if not issubclass(sampler, pints.MCMCSampler):
            raise ValueError(
                'Sampler has to be a `pints.MCMCSampler`.'
            )
        self._sampler = sampler
