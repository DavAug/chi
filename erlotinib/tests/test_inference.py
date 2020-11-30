#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pandas as pd
import pints

import erlotinib as erlo


class _NonStandardLikelihood(pints.LogPDF):
    """
    The erlotinib.InferenceController class assumes that the likelihoods have
    a private attribute `_problem` from which the parameter names defined in
    the erlotinib models can be recovered.

    I this attribute does not exist, we generate generic names and to test this
    we create a class here.
    """
    def __init__(self):
        super(_NonStandardLikelihood, self).__init__()

    def n_parameters(self):
        # Supposed to return the number of parameters
        # Here fixed to match the toy PKPD + error model used below
        return 6


class TestOptimisationController(unittest.TestCase):
    """
    Tests the erlotinib.OptimisationController class.
    """

    @classmethod
    def setUpClass(cls):
        # Get test data and model
        data = erlo.DataLibrary().lung_cancer_control_group()
        individual = 40
        mask = data['#ID'] == individual  # Arbitrary test id
        times = data[mask]['TIME in day'].to_numpy()
        observed_volumes = data[mask]['TUMOUR VOLUME in cm^3'].to_numpy()

        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)

        # Create inverse problem
        problem = erlo.InverseProblem(model, times, observed_volumes)
        cls.log_likelihood = pints.GaussianLogLikelihood(problem)
        log_prior_tumour_volume = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_drug_conc = pints.UniformLogPrior(-1E-3, 1E-3)
        log_prior_kappa = pints.UniformLogPrior(-1E-3, 1E-3)
        log_prior_lambda_0 = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_lambda_1 = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_sigma = pints.HalfCauchyLogPrior(location=0, scale=3)
        cls.log_prior = pints.ComposedLogPrior(
            log_prior_tumour_volume,
            log_prior_drug_conc,
            log_prior_kappa,
            log_prior_lambda_0,
            log_prior_lambda_1,
            log_prior_sigma)
        log_posterior = erlo.LogPosterior(cls.log_likelihood, cls.log_prior)
        log_posterior.set_id(individual)

        # Set up optmisation controller
        cls.optimiser = erlo.OptimisationController(log_posterior)

        cls.n_ids = 1
        cls.n_params = 6

    def test_call_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'Log-posterior has to be'):
            erlo.OptimisationController('bad log-posterior')

    def test_call_pooled_log_pdf(self):
        # Create a pints.PooledLogPDF by dublicating problem
        log_likelihood = pints.PooledLogPDF(
            log_pdfs=[self.log_likelihood, self.log_likelihood],
            pooled=[True]*self.n_params)

        # Test that OptimisationController can be instantiate without error
        log_posterior = erlo.LogPosterior(log_likelihood, self.log_prior)
        erlo.OptimisationController(log_posterior)

    def test_call_nonstandard_log_pdf(self):
        # Create a pints.PooledLogPDF by dublicating problem
        log_likelihood = _NonStandardLikelihood()

        # Test that OptimisationController can be instantiate without error
        log_posterior = erlo.LogPosterior(log_likelihood, self.log_prior)
        erlo.OptimisationController(log_posterior)

    def test_run(self):
        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        self.optimiser.set_parallel_evaluation(False)

        self.optimiser.set_n_runs(3)
        result = self.optimiser.run(n_max_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Estimate')
        self.assertEqual(keys[3], 'Score')
        self.assertEqual(keys[4], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), self.n_params)
        self.assertEqual(parameters[0], 'Param 1')
        self.assertEqual(parameters[1], 'Param 2')
        self.assertEqual(parameters[2], 'Param 3')
        self.assertEqual(parameters[3], 'Param 4')
        self.assertEqual(parameters[4], 'Param 5')
        self.assertEqual(parameters[5], 'Param 6')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

        # Check failure of optimisation doesn't interrupt all runs
        # (CMAES returns NAN for 1-dim problems)
        self.optimiser.set_n_runs(3)
        self.optimiser._initial_params[0, 0, 0] = -1
        result = self.optimiser.run(n_max_iterations=10)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Estimate')
        self.assertEqual(keys[3], 'Score')
        self.assertEqual(keys[4], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 6)
        self.assertEqual(parameters[0], 'Param 1')
        self.assertEqual(parameters[1], 'Param 2')
        self.assertEqual(parameters[2], 'Param 3')
        self.assertEqual(parameters[3], 'Param 4')
        self.assertEqual(parameters[4], 'Param 5')
        self.assertEqual(parameters[5], 'Param 6')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_set_n_runs(self):
        n_runs = 5
        self.optimiser.set_n_runs(n_runs)

        self.assertEqual(self.optimiser._n_runs, n_runs)
        self.assertEqual(
            self.optimiser._initial_params.shape,
            (self.n_ids, n_runs, self.n_params))

    def test_set_optmiser(self):
        self.optimiser.set_optimiser(pints.PSO)
        self.assertEqual(self.optimiser._optimiser, pints.PSO)

        self.optimiser.set_optimiser(pints.CMAES)
        self.assertEqual(self.optimiser._optimiser, pints.CMAES)

    def test_set_optimiser_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'Optimiser has to be'):
            self.optimiser.set_optimiser(str)

    def test_parallel_evaluation(self):
        # Set to sequential
        self.optimiser.set_parallel_evaluation(False)
        self.assertFalse(self.optimiser._parallel_evaluation)

        # Set to parallel
        self.optimiser.set_parallel_evaluation(True)
        self.assertTrue(self.optimiser._parallel_evaluation)

    def test_parallel_evaluation_bad_input(self):
        # Non-boolean and non-integer
        with self.assertRaisesRegex(ValueError, '`run_in_parallel` has'):
            self.optimiser.set_parallel_evaluation(2.2)

        # Negative input
        with self.assertRaisesRegex(ValueError, '`run_in_parallel` cannot'):
            self.optimiser.set_parallel_evaluation(-2)

    def test_set_transform_bad_transform(self):
        # Try to set transformation that is not a `pints.Transformation`
        transform = 'bad transform'

        with self.assertRaisesRegex(ValueError, 'Transform has to be an'):
            self.optimiser.set_transform(transform)

        # Try to set transformation with the wrong dimension
        transform = pints.LogTransformation(n_parameters=10)

        with self.assertRaisesRegex(ValueError, 'The dimensionality of the'):
            self.optimiser.set_transform(transform)

    def test_set_transform(self):
        # Apply transform
        transform = pints.LogTransformation(n_parameters=6)
        self.optimiser.set_transform(transform)

        self.assertEqual(self.optimiser._transform, transform)


class TestSamplingController(unittest.TestCase):
    """
    Tests the erlotinib.SamplingController class.
    """

    @classmethod
    def setUpClass(cls):
        # Get test data and model
        data = erlo.DataLibrary().lung_cancer_control_group()
        cls.individual = 40
        mask = data['#ID'] == cls.individual  # Arbitrary test id
        times = data[mask]['TIME in day'].to_numpy()
        observed_volumes = data[mask]['TUMOUR VOLUME in cm^3'].to_numpy()

        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)

        # Create inverse problem
        problem = erlo.InverseProblem(model, times, observed_volumes)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_prior_tumour_volume = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_drug_conc = pints.UniformLogPrior(-1E-3, 1E-3)
        log_prior_kappa = pints.UniformLogPrior(-1E-3, 1E-3)
        log_prior_lambda_0 = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_lambda_1 = pints.UniformLogPrior(1E-3, 1E1)
        log_prior_sigma = pints.HalfCauchyLogPrior(location=0, scale=3)
        log_prior = pints.ComposedLogPrior(
            log_prior_tumour_volume,
            log_prior_drug_conc,
            log_prior_kappa,
            log_prior_lambda_0,
            log_prior_lambda_1,
            log_prior_sigma)
        log_posterior = erlo.LogPosterior(log_likelihood, log_prior)
        log_posterior.set_id(cls.individual)

        # Set up sampling controller
        cls.sampler = erlo.SamplingController(log_posterior)

        cls.n_ids = 1
        cls.n_params = 6

    def test_call_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'Log-posterior has to be'):
            erlo.SamplingController('bad log-posterior')

    def test_run(self):
        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        self.sampler.set_parallel_evaluation(False)

        self.sampler.set_n_runs(3)
        result = self.sampler.run(n_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Parameter')
        self.assertEqual(keys[2], 'Sample')
        self.assertEqual(keys[3], 'Iteration')
        self.assertEqual(keys[4], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), self.n_params)
        self.assertEqual(parameters[0], 'Param 1')
        self.assertEqual(parameters[1], 'Param 2')
        self.assertEqual(parameters[2], 'Param 3')
        self.assertEqual(parameters[3], 'Param 4')
        self.assertEqual(parameters[4], 'Param 5')
        self.assertEqual(parameters[5], 'Param 6')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_set_initial_parameters(self):
        n_runs = 10
        self.sampler.set_n_runs(n_runs)

        # Create test data
        # First run estimates both params as 1 and second run as 2
        params = ['Param 3', 'Param 5'] * 2
        estimates = [1, 1, 2, 2]
        scores = [0.3, 0.3, 5, 5]
        runs = [1, 1, 2, 2]

        data = pd.DataFrame({
            'ID': self.individual,
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

        # Get initial values before setting them
        default_params = self.sampler._initial_params.copy()

        # Set initial values and test behaviour
        self.sampler.set_initial_parameters(data)
        new_params = self.sampler._initial_params

        self.assertEqual(
            default_params.shape, (self.n_ids, n_runs, self.n_params))
        self.assertEqual(new_params.shape, (self.n_ids, n_runs, self.n_params))

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
        self.sampler.set_initial_parameters(data)
        new_params = self.sampler._initial_params

        self.assertEqual(
            default_params.shape, (self.n_ids, n_runs, self.n_params))
        self.assertEqual(new_params.shape, (self.n_ids, n_runs, self.n_params))

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
        data['ID'] = self.individual
        data['Parameter'] = ['SOME', 'PARAMETERS'] * 2
        self.sampler.set_initial_parameters(data)
        new_params = self.sampler._initial_params

        self.assertEqual(
            default_params.shape, (self.n_ids, n_runs, self.n_params))
        self.assertEqual(new_params.shape, (self.n_ids, n_runs, self.n_params))

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
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            self.sampler.set_initial_parameters, data)

        # Create test data
        # First run estimates both params as 1 and second run as 2
        params = ['myokit.lambda_0', 'noise param 1'] * 2
        estimates = [1, 1, 2, 2]
        scores = [0.3, 0.3, 5, 5]
        runs = [1, 1, 2, 2]

        test_data = pd.DataFrame({
            'ID': self.individual,
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

        # Rename id key
        data = test_data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <ID>.',
            self.sampler.set_initial_parameters, data)

        # Rename parameter key
        data = test_data.rename(columns={'Parameter': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Parameter>.',
            self.sampler.set_initial_parameters, data)

        # Rename estimate key
        data = test_data.rename(columns={'Estimate': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Estimate>.',
            self.sampler.set_initial_parameters, data)

        # Rename score key
        data = test_data.rename(columns={'Score': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Score>.',
            self.sampler.set_initial_parameters, data)

        # Rename run key
        data = test_data.rename(columns={'Run': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Run>.',
            self.sampler.set_initial_parameters, data)

    def test_set_n_runs(self):
        n_runs = 5
        self.sampler.set_n_runs(n_runs)

        self.assertEqual(self.sampler._n_runs, n_runs)
        self.assertEqual(
            self.sampler._initial_params.shape,
            (self.n_ids, n_runs, self.n_params))

    def test_set_sampler(self):
        self.sampler.set_sampler(pints.HamiltonianMCMC)
        self.assertEqual(self.sampler._sampler, pints.HamiltonianMCMC)

        self.sampler.set_sampler(pints.HaarioACMC)
        self.assertEqual(self.sampler._sampler, pints.HaarioACMC)

    def test_set_sampler_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'Sampler has to be'):
            self.sampler.set_sampler(str)

    def test_parallel_evaluation(self):
        # Set to sequential
        self.sampler.set_parallel_evaluation(False)
        self.assertFalse(self.sampler._parallel_evaluation)

        # Set to parallel
        self.sampler.set_parallel_evaluation(True)
        self.assertTrue(self.sampler._parallel_evaluation)

    def test_parallel_evaluation_bad_input(self):
        # Non-boolean and non-integer
        with self.assertRaisesRegex(ValueError, '`run_in_parallel` has'):
            self.sampler.set_parallel_evaluation(2.2)

        # Negative input
        with self.assertRaisesRegex(ValueError, '`run_in_parallel` cannot'):
            self.sampler.set_parallel_evaluation(-2)

    def test_set_transform_bad_transform(self):
        # Try to set transformation that is not a `pints.Transformation`
        transform = 'bad transform'

        with self.assertRaisesRegex(ValueError, 'Transform has to be an'):
            self.sampler.set_transform(transform)

        # Try to set transformation with the wrong dimension
        transform = pints.LogTransformation(n_parameters=10)

        with self.assertRaisesRegex(ValueError, 'The dimensionality of the'):
            self.sampler.set_transform(transform)

    def test_set_transform(self):
        # Apply transform
        transform = pints.LogTransformation(n_parameters=self.n_params)
        self.sampler.set_transform(transform)

        self.assertEqual(self.sampler._transform, transform)


if __name__ == '__main__':
    unittest.main()
