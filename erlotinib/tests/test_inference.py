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
        mask = data['#ID'] == 40  # Arbitrary test id
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
        log_posterior = pints.LogPosterior(cls.log_likelihood, cls.log_prior)

        # Set up optmisation controller
        cls.optimiser = erlo.OptimisationController(log_posterior)

    def test_call_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'Log-posterior has to be'):
            erlo.OptimisationController('bad log-posterior')

    def test_call_pooled_log_pdf(self):
        # Create a pints.PooledLogPDF by dublicating problem
        log_likelihood = pints.PooledLogPDF(
            log_pdfs=[self.log_likelihood, self.log_likelihood],
            pooled=[True]*6)

        # Test that OptimisationController can be instantiate without error
        log_posterior = pints.LogPosterior(log_likelihood, self.log_prior)
        erlo.OptimisationController(log_posterior)

    def test_call_nonstandard_log_pdf(self):
        # Create a pints.PooledLogPDF by dublicating problem
        log_likelihood = _NonStandardLikelihood()

        # Test that OptimisationController can be instantiate without error
        log_posterior = pints.LogPosterior(log_likelihood, self.log_prior)
        erlo.OptimisationController(log_posterior)

    def test_fix_parameters_bad_mask(self):
        # Mask length doesn't match number of parameters
        mask = [False, True, True]
        value = [1, 1]

        with self.assertRaisesRegex(ValueError, 'Length of mask'):
            self.optimiser.fix_parameters(mask, value)

        # Mask is not boolean
        mask = ['False', 'True', 'True', 'False', 'True', 'True']
        value = [1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'Mask has to be'):
            self.optimiser.fix_parameters(mask, value)

    def test_fix_parameters_bad_values(self):
        # Number of values doesn't match the number of parameters to fix
        mask = [False, True, True, False, True, True]
        value = [1, 1, 1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'Values has to have the same'):
            self.optimiser.fix_parameters(mask, value)

    def test_fix_parameters(self):
        # Fix all but parameter 1 and 4
        mask = [False, True, True, False, True, True]
        value = [1, 1, 1, 1]

        self.optimiser.fix_parameters(mask, value)

        parameters = self.optimiser._parameters
        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.lambda_0')

        # Fix a different set of parameters
        mask = [False, True, False, True, True, False]
        value = [1, 1, 1]

        self.optimiser.fix_parameters(mask, value)

        parameters = self.optimiser._parameters
        self.assertEqual(len(parameters), 3)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.kappa')
        self.assertEqual(parameters[2], 'noise param 1')

    def test_run(self):
        # Fix
        mask = [False, True, True, False, False, False]
        value = [0, 0]
        self.optimiser.fix_parameters(mask, value)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        self.optimiser.set_parallel_evaluation(False)

        self.optimiser.set_n_runs(3)
        result = self.optimiser.run(n_max_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'Parameter')
        self.assertEqual(keys[1], 'Estimate')
        self.assertEqual(keys[2], 'Score')
        self.assertEqual(keys[3], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 4)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.lambda_0')
        self.assertEqual(parameters[2], 'myokit.lambda_1')
        self.assertEqual(parameters[3], 'noise param 1')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

        # Check failure of optimisation doesn't interrupt all runs
        # (CMAES returns NAN for 1-dim problems)
        mask = [True, True, True, True, True, False]
        value = [1, 0, 0, 1, 1]
        self.optimiser.fix_parameters(mask, value)

        self.optimiser.set_n_runs(3)
        result = self.optimiser.run(n_max_iterations=10)

        keys = result.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'Parameter')
        self.assertEqual(keys[1], 'Estimate')
        self.assertEqual(keys[2], 'Score')
        self.assertEqual(keys[3], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'noise param 1')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_set_n_runs(self):
        # Unfix all parameters (just to reset possibly fixed parameters)
        mask = [False, False, False, False, False, False]
        value = []
        self.optimiser.fix_parameters(mask, value)

        self.optimiser.set_n_runs(5)

        self.assertEqual(self.optimiser._n_runs, 5)
        self.assertEqual(self.optimiser._initial_params.shape, (5, 6))

        # Fix parameters
        mask = [True, True, True, False, False, False]
        value = [1, 1, 1]
        self.optimiser.fix_parameters(mask, value)

        self.optimiser.set_n_runs(20)

        self.assertEqual(self.optimiser._n_runs, 20)
        self.assertEqual(self.optimiser._initial_params.shape, (20, 3))

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
        # Unfix all parameters (just to reset possibly fixed parameters)
        mask = [False, False, False, False, False, False]
        value = []
        self.optimiser.fix_parameters(mask, value)

        # Apply transform
        transform = pints.LogTransformation(n_parameters=6)
        self.optimiser.set_transform(transform)

        self.assertEqual(self.optimiser._transform, transform)

        # Fix parameters and apply transform again
        mask = [False, True, True, True, True, True]
        value = [1, 1, 1, 1, 1]
        self.optimiser.fix_parameters(mask, value)

        self.assertIsNone(self.optimiser._transform)

        transform = pints.LogTransformation(n_parameters=1)
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
        mask = data['#ID'] == 40  # Arbitrary test id
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
        log_posterior = pints.LogPosterior(log_likelihood, log_prior)

        # Set up sampling controller
        cls.sampler = erlo.SamplingController(log_posterior)

    def test_call_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'Log-posterior has to be'):
            erlo.SamplingController('bad log-posterior')

    def test_fix_parameters_bad_mask(self):
        # Mask length doesn't match number of parameters
        mask = [False, True, True]
        value = [1, 1]

        with self.assertRaisesRegex(ValueError, 'Length of mask'):
            self.sampler.fix_parameters(mask, value)

        # Mask is not boolean
        mask = ['False', 'True', 'True', 'False', 'True', 'True']
        value = [1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'Mask has to be'):
            self.sampler.fix_parameters(mask, value)

    def test_fix_parameters_bad_values(self):
        # Number of values doesn't match the number of parameters to fix
        mask = [False, True, True, False, True, True]
        value = [1, 1, 1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'Values has to have the same'):
            self.sampler.fix_parameters(mask, value)

    def test_fix_parameters(self):
        # Fix all but parameter 1 and 4
        mask = [False, True, True, False, True, True]
        value = [1, 1, 1, 1]

        self.sampler.fix_parameters(mask, value)

        parameters = self.sampler._parameters
        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.lambda_0')

        # Fix a different set of parameters
        mask = [False, True, False, True, True, False]
        value = [1, 1, 1]

        self.sampler.fix_parameters(mask, value)

        parameters = self.sampler._parameters
        self.assertEqual(len(parameters), 3)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.kappa')
        self.assertEqual(parameters[2], 'noise param 1')

    def test_run(self):
        # Fix
        mask = [False, True, True, False, False, False]
        value = [0, 0]
        self.sampler.fix_parameters(mask, value)

        # Set evaluator to sequential, because otherwise codecov
        # complains that posterior was never evaluated.
        # (Potentially codecov cannot keep track of multiple CPUs)
        self.sampler.set_parallel_evaluation(False)

        self.sampler.set_n_runs(3)
        result = self.sampler.run(n_iterations=20)

        keys = result.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'Parameter')
        self.assertEqual(keys[1], 'Sample')
        self.assertEqual(keys[2], 'Iteration')
        self.assertEqual(keys[3], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 4)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.lambda_0')
        self.assertEqual(parameters[2], 'myokit.lambda_1')
        self.assertEqual(parameters[3], 'noise param 1')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

        # Check failure of optimisation doesn't interrupt all runs
        # (CMAES returns NAN for 1-dim problems)
        mask = [True, True, True, True, True, False]
        value = [1, 0, 0, 1, 1]
        self.sampler.fix_parameters(mask, value)

        self.sampler.set_n_runs(3)
        result = self.sampler.run(n_iterations=10)

        keys = result.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'Parameter')
        self.assertEqual(keys[1], 'Sample')
        self.assertEqual(keys[2], 'Iteration')
        self.assertEqual(keys[3], 'Run')

        parameters = result['Parameter'].unique()
        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'noise param 1')

        runs = result['Run'].unique()
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0], 1)
        self.assertEqual(runs[1], 2)
        self.assertEqual(runs[2], 3)

    def test_set_initial_parameters(self):
        # Unfix all parameters and set to 10 runs
        mask = [False, False, False, False, False, False]
        value = []
        self.sampler.fix_parameters(mask, value)
        self.sampler.set_n_runs(10)

        # Create test data
        # First run estimates both params as 1 and second run as 2
        params = ['myokit.lambda_0', 'noise param 1'] * 2
        estimates = [1, 1, 2, 2]
        scores = [0.3, 0.3, 5, 5]
        runs = [1, 1, 2, 2]

        data = pd.DataFrame({
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

        # Get initial values before setting them
        default_params = self.sampler._initial_params

        # Set initial values and test behaviour
        self.sampler.set_initial_parameters(data)
        new_params = self.sampler._initial_params

        self.assertEqual(default_params.shape, (10, 6))
        self.assertEqual(new_params.shape, (10, 6))

        # Compare values. All but 3rd and 5th index should coincide.
        # 3rd and 5th should correspong map estimates
        self.assertTrue(np.array_equal(new_params[:, 0], default_params[:, 0]))
        self.assertTrue(np.array_equal(new_params[:, 1], default_params[:, 1]))
        self.assertTrue(np.array_equal(new_params[:, 2], default_params[:, 2]))
        self.assertTrue(np.array_equal(new_params[:, 3], np.array([2] * 10)))
        self.assertTrue(np.array_equal(new_params[:, 4], default_params[:, 4]))
        self.assertTrue(np.array_equal(new_params[:, 5], np.array([2] * 10)))

        # Check that it works fine even if parameter cannot be found
        data['Parameter'] = ['SOME', 'PARAMETERS'] * 2
        self.sampler.set_initial_parameters(data)
        new_params = self.sampler._initial_params

        self.assertEqual(default_params.shape, (10, 6))
        self.assertEqual(new_params.shape, (10, 6))

        # Compare values. All but 3rd and 5th index should coincide.
        # 3rd and 5th should correspong map estimates
        self.assertTrue(np.array_equal(new_params[:, 0], default_params[:, 0]))
        self.assertTrue(np.array_equal(new_params[:, 1], default_params[:, 1]))
        self.assertTrue(np.array_equal(new_params[:, 2], default_params[:, 2]))
        self.assertTrue(np.array_equal(new_params[:, 3], np.array([2] * 10)))
        self.assertTrue(np.array_equal(new_params[:, 4], default_params[:, 4]))
        self.assertTrue(np.array_equal(new_params[:, 5], np.array([2] * 10)))

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
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

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
        # Unfix all parameters (just to reset possibly fixed parameters)
        mask = [False, False, False, False, False, False]
        value = []
        self.sampler.fix_parameters(mask, value)

        self.sampler.set_n_runs(5)

        self.assertEqual(self.sampler._n_runs, 5)
        self.assertEqual(self.sampler._initial_params.shape, (5, 6))

        # Fix parameters
        mask = [True, True, True, False, False, False]
        value = [1, 1, 1]
        self.sampler.fix_parameters(mask, value)

        self.sampler.set_n_runs(20)

        self.assertEqual(self.sampler._n_runs, 20)
        self.assertEqual(self.sampler._initial_params.shape, (20, 3))

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
        # Unfix all parameters (just to reset possibly fixed parameters)
        mask = [False, False, False, False, False, False]
        value = []
        self.sampler.fix_parameters(mask, value)

        # Apply transform
        transform = pints.LogTransformation(n_parameters=6)
        self.sampler.set_transform(transform)

        self.assertEqual(self.sampler._transform, transform)

        # Fix parameters and apply transform again
        mask = [False, True, True, True, True, True]
        value = [1, 1, 1, 1, 1]
        self.sampler.fix_parameters(mask, value)

        self.assertIsNone(self.sampler._transform)

        transform = pints.LogTransformation(n_parameters=1)
        self.sampler.set_transform(transform)

        self.assertEqual(self.sampler._transform, transform)


if __name__ == '__main__':
    unittest.main()
