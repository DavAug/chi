#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy
import unittest

import numpy as np
import pandas as pd
import pints

import erlotinib as erlo


class TestInferenceController(unittest.TestCase):
    """
    Test the erlotinib.InferenceController base class.
    """

    @classmethod
    def setUpClass(cls):
        # Get test data and model
        data = erlo.DataLibrary().lung_cancer_control_group()
        individual = 40
        mask = data['ID'] == individual  # Arbitrary test id
        data = data[mask]
        mask = data['Biomarker'] == 'Tumour volume'  # Arbitrary biomarker
        times = data[mask]['Time'].to_numpy()
        observed_volumes = data[mask]['Measurement'].to_numpy()

        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = erlo.PharmacodynamicModel(path)
        error_model = erlo.ConstantAndMultiplicativeGaussianErrorModel()
        cls.log_likelihood = erlo.LogLikelihood(
            mechanistic_model, error_model, observed_volumes, times)
        cls.log_likelihood.set_id(individual)

        # Create posterior
        log_prior_tumour_volume = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_drug_conc = pints.UniformLogPrior(-1E-3, 1E-3)
        log_prior_kappa = pints.UniformLogPrior(-1E-3, 1E-3)
        log_prior_lambda_0 = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_lambda_1 = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_sigma_base = pints.HalfCauchyLogPrior(location=0, scale=3)
        log_prior_sigma_rel = pints.HalfCauchyLogPrior(location=0, scale=3)
        cls.log_prior = pints.ComposedLogPrior(
            log_prior_tumour_volume,
            log_prior_drug_conc,
            log_prior_kappa,
            log_prior_lambda_0,
            log_prior_lambda_1,
            log_prior_sigma_base,
            log_prior_sigma_rel)
        log_posterior = erlo.LogPosterior(cls.log_likelihood, cls.log_prior)

        # Set up optmisation controller
        cls.controller = erlo.InferenceController(log_posterior)

        cls.n_ids = 1
        cls.n_params = 7

    def test_call_bad_input(self):
        # Wrong type of log-posterior
        log_posterior = 'bad log-posterior'

        with self.assertRaisesRegex(ValueError, 'Log-posterior has to be'):
            erlo.InferenceController(log_posterior)

        # Log-posteriors don't have the same number of parameters
        log_posterior_1 = erlo.LogPosterior(
            self.log_likelihood, self.log_prior)
        log_posterior_2 = erlo.LogPosterior(
            self.log_likelihood, self.log_prior)
        log_posterior_2._n_parameters = 20

        with self.assertRaisesRegex(ValueError, 'All log-posteriors have to'):
            erlo.InferenceController([log_posterior_1, log_posterior_2])

    def test_set_n_runs(self):
        n_runs = 5
        self.controller.set_n_runs(n_runs)

        self.assertEqual(self.controller._n_runs, n_runs)
        self.assertEqual(
            self.controller._initial_params.shape,
            (self.n_ids, n_runs, self.n_params))

    def test_parallel_evaluation(self):
        # Set to sequential
        self.controller.set_parallel_evaluation(False)
        self.assertFalse(self.controller._parallel_evaluation)

        # Set to parallel
        self.controller.set_parallel_evaluation(True)
        self.assertTrue(self.controller._parallel_evaluation)

    def test_parallel_evaluation_bad_input(self):
        # Non-boolean and non-integer
        with self.assertRaisesRegex(ValueError, '`run_in_parallel` has'):
            self.controller.set_parallel_evaluation(2.2)

        # Negative input
        with self.assertRaisesRegex(ValueError, '`run_in_parallel` cannot'):
            self.controller.set_parallel_evaluation(-2)

    def test_set_transform(self):
        # Apply transform
        transform = pints.LogTransformation(n_parameters=7)
        self.controller.set_transform(transform)

        self.assertEqual(self.controller._transform, transform)

    def test_set_transform_bad_transform(self):
        # Try to set transformation that is not a `pints.Transformation`
        transform = 'bad transform'

        with self.assertRaisesRegex(ValueError, 'Transform has to be an'):
            self.controller.set_transform(transform)

        # Try to set transformation with the wrong dimension
        transform = pints.LogTransformation(n_parameters=10)

        with self.assertRaisesRegex(ValueError, 'The dimensionality of the'):
            self.controller.set_transform(transform)


class TestOptimisationController(unittest.TestCase):
    """
    Tests the erlotinib.OptimisationController class.
    """

    @classmethod
    def setUpClass(cls):
        # Set up test problems
        # Model I: Individual with ID 40
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        cls.problem = erlo.ProblemModellingController(model, error_models)

        data = erlo.DataLibrary().lung_cancer_control_group()
        cls.problem.set_data(data, {'myokit.tumour_volume': 'Tumour volume'})

        n_parameters = 7
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        cls.problem.set_log_prior(log_priors)
        cls.log_posterior_id_40 = cls.problem.get_log_posterior(
            individual='40')

        # Model II: Hierarchical model across all individuals
        pop_models = [
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.HeterogeneousModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.LogNormalModel()]
        problem = copy.deepcopy(cls.problem)
        problem.set_population_model(pop_models)

        n_parameters = 5 + 8 * 2 + 2
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        problem.set_log_prior(log_priors)
        cls.hierarchical_posterior = problem.get_log_posterior()

        # Get IDs for testing
        cls.ids = data['ID'].unique()

    def test_run(self):
        # Case I: Individual with ID 40
        optimiser = erlo.OptimisationController(self.log_posterior_id_40)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        optimiser.set_parallel_evaluation(False)

        optimiser.set_n_runs(3)
        result = optimiser.run(n_max_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Estimate')
        self.assertEqual(keys[3], 'Score')
        self.assertEqual(keys[4], 'Run')

        ids = result['ID'].unique()
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 'ID 40')

        n_parameters = 7
        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), n_parameters)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')
        self.assertEqual(parameters[5], 'Sigma base')
        self.assertEqual(parameters[6], 'Sigma rel.')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

        # Case II: Hierarchical model
        optimiser = erlo.OptimisationController(self.hierarchical_posterior)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        optimiser.set_parallel_evaluation(False)

        optimiser.set_n_runs(3)
        result = optimiser.run(n_max_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Estimate')
        self.assertEqual(keys[3], 'Score')
        self.assertEqual(keys[4], 'Run')

        # One ID for each prefix
        ids = result['ID'].unique()
        self.assertEqual(len(ids), 11)  # nids + 'Pooled'
        self.assertEqual(ids[0], 'Pooled')
        self.assertEqual(ids[1], 'ID ' + str(self.ids[0]))
        self.assertEqual(ids[2], 'ID ' + str(self.ids[1]))
        self.assertEqual(ids[3], 'ID ' + str(self.ids[2]))
        self.assertEqual(ids[4], 'ID ' + str(self.ids[3]))
        self.assertEqual(ids[5], 'ID ' + str(self.ids[4]))
        self.assertEqual(ids[6], 'ID ' + str(self.ids[5]))
        self.assertEqual(ids[7], 'ID ' + str(self.ids[6]))
        self.assertEqual(ids[8], 'ID ' + str(self.ids[7]))
        self.assertEqual(ids[9], 'Mean')
        self.assertEqual(ids[10], 'Std.')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 7)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')
        self.assertEqual(parameters[5], 'Sigma base')
        self.assertEqual(parameters[6], 'Sigma rel.')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_run_catch_exception(self):
        # Check failure of optimisation doesn't interrupt all runs
        # (CMAES returns NAN for 1-dim problems)

        # Get test data and model
        problem = copy.deepcopy(self.problem)
        problem.fix_parameters({
            'myokit.drug_concentration': 1,
            'myokit.kappa': 1,
            'myokit.lambda_0': 1,
            'myokit.lambda_1': 1,
            'Sigma base': 1,
            'Sigma rel.': 1})
        problem.set_log_prior([pints.UniformLogPrior(1E-3, 1E1)])
        log_posteriors = problem.get_log_posterior()

        # Set up optmisation controller
        optimiser = erlo.OptimisationController(log_posteriors[0])
        optimiser.set_n_runs(3)
        result = optimiser.run(n_max_iterations=10)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Estimate')
        self.assertEqual(keys[3], 'Score')
        self.assertEqual(keys[4], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_set_optmiser(self):
        optimiser = erlo.OptimisationController(self.log_posterior_id_40)
        optimiser.set_optimiser(pints.PSO)
        self.assertEqual(optimiser._optimiser, pints.PSO)

        optimiser.set_optimiser(pints.CMAES)
        self.assertEqual(optimiser._optimiser, pints.CMAES)

    def test_set_optimiser_bad_input(self):
        optimiser = erlo.OptimisationController(self.log_posterior_id_40)
        with self.assertRaisesRegex(ValueError, 'Optimiser has to be'):
            optimiser.set_optimiser(str)


class TestSamplingController(unittest.TestCase):
    """
    Tests the erlotinib.SamplingController class.
    """

    @classmethod
    def setUpClass(cls):
        # Set up test problems
        # Model I: Individual with ID 40
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        problem = erlo.ProblemModellingController(model, error_models)

        data = erlo.DataLibrary().lung_cancer_control_group()
        problem.set_data(data, {'myokit.tumour_volume': 'Tumour volume'})

        n_parameters = 7
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        problem.set_log_prior(log_priors)
        cls.log_posterior_id_40 = problem.get_log_posterior(individual='40')

        # Model II: Hierarchical model across all individuals
        pop_models = [
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.HeterogeneousModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.LogNormalModel()]
        problem.set_population_model(pop_models)

        n_parameters = 5 + 8 * 2 + 2
        log_priors = [
            pints.HalfCauchyLogPrior(location=0, scale=3)] * n_parameters
        problem.set_log_prior(log_priors)
        cls.hierarchical_posterior = problem.get_log_posterior()

        # Get IDs for testing
        cls.ids = data['ID'].unique()

    def test_run(self):
        # Case I: Individual with ID 40
        sampler = erlo.SamplingController(self.log_posterior_id_40)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        sampler.set_parallel_evaluation(False)

        sampler.set_n_runs(3)
        result = sampler.run(n_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Sample')
        self.assertEqual(keys[3], 'Iteration')
        self.assertEqual(keys[4], 'Run')

        ids = result['ID'].unique()
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 'ID 40')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 7)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')
        self.assertEqual(parameters[5], 'Sigma base')
        self.assertEqual(parameters[6], 'Sigma rel.')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

        # Case II: Hierarchical model
        sampler = erlo.SamplingController(self.hierarchical_posterior)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        sampler.set_parallel_evaluation(False)

        sampler.set_n_runs(3)
        result = sampler.run(n_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Sample')
        self.assertEqual(keys[3], 'Iteration')
        self.assertEqual(keys[4], 'Run')

        # One ID for each prefix
        ids = result['ID'].unique()
        self.assertEqual(len(ids), 11)  # nids + 'Pooled' + 'Mean' + 'Std'
        self.assertEqual(ids[0], 'Pooled')
        self.assertEqual(ids[1], 'ID ' + str(self.ids[0]))
        self.assertEqual(ids[2], 'ID ' + str(self.ids[1]))
        self.assertEqual(ids[3], 'ID ' + str(self.ids[2]))
        self.assertEqual(ids[4], 'ID ' + str(self.ids[3]))
        self.assertEqual(ids[5], 'ID ' + str(self.ids[4]))
        self.assertEqual(ids[6], 'ID ' + str(self.ids[5]))
        self.assertEqual(ids[7], 'ID ' + str(self.ids[6]))
        self.assertEqual(ids[8], 'ID ' + str(self.ids[7]))
        self.assertEqual(ids[9], 'Mean')
        self.assertEqual(ids[10], 'Std.')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 7)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')
        self.assertEqual(parameters[5], 'Sigma base')
        self.assertEqual(parameters[6], 'Sigma rel.')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_set_initial_parameters(self):
        # Test case I: Individual data
        n_runs = 10
        sampler = erlo.SamplingController(self.log_posterior_id_40)
        sampler.set_n_runs(n_runs)

        # Create test data
        # First run estimates both params as 1 and second run as 2
        params = ['myokit.kappa', 'myokit.lambda_1'] * 2
        estimates = [1, 1, 2, 2]
        scores = [0.3, 0.3, 5, 5]
        runs = [1, 1, 2, 2]

        data = pd.DataFrame({
            'ID': 'ID 40',
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

        # Get initial values before setting them
        default_params = sampler._initial_params.copy()

        # Set initial values and test behaviour
        sampler.set_initial_parameters(data)
        new_params = sampler._initial_params

        n_ids = 1
        n_parameters = 7
        self.assertEqual(
            default_params.shape, (n_ids, n_runs, n_parameters))
        self.assertEqual(new_params.shape, (n_ids, n_runs, n_parameters))

        # Compare values. All but 3rd and 5th parameter should coincide.
        # 3rd and 5th should correspong map estimates
        self.assertTrue(np.array_equal(
            new_params[0, :, 0], default_params[0, :, 0]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 1], default_params[0, :, 1]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 2], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 3], default_params[0, :, 3]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 4], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 5], default_params[0, :, 5]))

        # Check that it works fine even if ID cannot be found
        data['ID'] = 'Some ID'
        sampler.set_initial_parameters(data)
        new_params = sampler._initial_params

        self.assertEqual(
            default_params.shape, (n_ids, n_runs, n_parameters))
        self.assertEqual(new_params.shape, (n_ids, n_runs, n_parameters))

        # Compare values. All but 3rd and 5th index should coincide.
        # 3rd and 5th should correspong map estimates
        self.assertTrue(np.array_equal(
            new_params[0, :, 0], default_params[0, :, 0]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 1], default_params[0, :, 1]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 2], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 3], default_params[0, :, 3]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 4], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 5], default_params[0, :, 5]))

        # Check that it works fine even if parameter cannot be found
        data['ID'] = 'ID 40'
        data['Parameter'] = ['SOME', 'PARAMETERS'] * 2
        sampler.set_initial_parameters(data)
        new_params = sampler._initial_params

        self.assertEqual(
            default_params.shape, (n_ids, n_runs, n_parameters))
        self.assertEqual(new_params.shape, (n_ids, n_runs, n_parameters))

        # Compare values. All but 3rd and 5th index should coincide.
        # 3rd and 5th should correspong map estimates
        self.assertTrue(np.array_equal(
            new_params[0, :, 0], default_params[0, :, 0]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 1], default_params[0, :, 1]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 2], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 3], default_params[0, :, 3]))
        self.assertTrue(np.array_equal(
            new_params[0, :, 4], np.array([2] * 10)))
        self.assertTrue(np.array_equal(
            new_params[0, :, 5], default_params[0, :, 5]))

    def test_set_initial_parameters_bad_input(self):
        sampler = erlo.SamplingController(self.log_posterior_id_40)

        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            sampler.set_initial_parameters, data)

        # Create test data
        # First run estimates both params as 1 and second run as 2
        params = ['myokit.lambda_0', 'noise param 1'] * 2
        estimates = [1, 1, 2, 2]
        scores = [0.3, 0.3, 5, 5]
        runs = [1, 1, 2, 2]

        test_data = pd.DataFrame({
            'ID': 'ID 40',
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

        # Rename id key
        data = test_data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <ID>.',
            sampler.set_initial_parameters, data)

        # Rename parameter key
        data = test_data.rename(columns={'Parameter': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Parameter>.',
            sampler.set_initial_parameters, data)

        # Rename estimate key
        data = test_data.rename(columns={'Estimate': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Estimate>.',
            sampler.set_initial_parameters, data)

        # Rename score key
        data = test_data.rename(columns={'Score': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Score>.',
            sampler.set_initial_parameters, data)

        # Rename run key
        data = test_data.rename(columns={'Run': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Run>.',
            sampler.set_initial_parameters, data)

    def test_set_sampler(self):
        sampler = erlo.SamplingController(self.log_posterior_id_40)
        sampler.set_sampler(pints.HamiltonianMCMC)
        self.assertEqual(sampler._sampler, pints.HamiltonianMCMC)

        sampler.set_sampler(pints.HaarioACMC)
        self.assertEqual(sampler._sampler, pints.HaarioACMC)

    def test_set_sampler_bad_input(self):
        sampler = erlo.SamplingController(self.log_posterior_id_40)
        with self.assertRaisesRegex(ValueError, 'Sampler has to be'):
            sampler.set_sampler(str)


if __name__ == '__main__':
    unittest.main()
