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


class TestProblemModellingController(unittest.TestCase):
    """
    Tests the erlotinib.ProblemModellingController class.
    """
    @classmethod
    def setUpClass(cls):
        # Create test dataset
        ids = [0, 0, 0, 1, 1, 1, 2, 2]
        times = [0, 1, 2, 2, np.nan, 4, 1, 3]
        volumes = [np.nan, 0.3, 0.2, 0.5, 0.1, 0.2, 0.234, np.nan]
        cytokines = [3.4, 0.3, 0.2, 0.5, np.nan, 234, 0.234, 0]
        cls.data = pd.DataFrame({
            'ID': ids,
            'Time': times,
            'Biomarker': volumes,
            'Biomarker 1': volumes,
            'Biomarker 2': cytokines})

        # Create test problem
        cls.problem = erlo.ProblemModellingController(
            cls.data, biom_keys=['Biomarker'])

        # Create test model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.model = erlo.PharmacodynamicModel(path)
        cls.log_likelihoods = [pints.GaussianLogLikelihood]
        cls.log_priors = [
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1)]

    def test_bad_input(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        with self.assertRaisesRegex(ValueError, 'Data has to be'):
            erlo.ProblemModellingController(data=data)

        # Wrong ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        with self.assertRaisesRegex(ValueError, 'Data does not have'):
            erlo.ProblemModellingController(data=data)

        # Wrong time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        with self.assertRaisesRegex(ValueError, 'Data does not have'):
            erlo.ProblemModellingController(data=data)

        # Wrong Biomarker key
        data = self.data.rename(columns={'Biomarker': 'SOME NON-STANDARD KEY'})

        with self.assertRaisesRegex(ValueError, 'Data does not have'):
            erlo.ProblemModellingController(data=data)

    def test_data_keys(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        erlo.ProblemModellingController(
            data=data, id_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            erlo.ProblemModellingController(
                data=data, id_key='SOME WRONG KEY')

        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        erlo.ProblemModellingController(
            data=data, time_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            erlo.ProblemModellingController(
                data=data, time_key='SOME WRONG KEY')

        # Rename biomarker key
        data = self.data.rename(columns={'Biomarker': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        erlo.ProblemModellingController(
            data=data, biom_keys=['SOME NON-STANDARD KEY'])

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            erlo.ProblemModellingController(
                data=data, biom_keys=['SOME WRONG KEY'])

    def test_fix_parameters(self):
        # Fix model parameters
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.log_likelihoods)
        name_value_dict = dict({
            'myokit.drug_concentration': 0,
            'myokit.kappa': 1})

        self.problem.fix_parameters(name_value_dict)

        self.assertEqual(self.problem.n_parameters(), 4)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 4)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.lambda_0')
        self.assertEqual(param_names[2], 'myokit.lambda_1')
        self.assertEqual(param_names[3], 'Noise param 1')

        param_values = self.problem._fixed_params_values
        self.assertEqual(len(param_values), 6)
        self.assertEqual(param_values[1], 0)
        self.assertEqual(param_values[2], 1)

        # Free kappa and fix lambda_1
        name_value_dict = dict({
            'myokit.lambda_1': 2,
            'myokit.kappa': None})

        self.problem.fix_parameters(name_value_dict)

        self.assertEqual(self.problem.n_parameters(), 4)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 4)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.kappa')
        self.assertEqual(param_names[2], 'myokit.lambda_0')
        self.assertEqual(param_names[3], 'Noise param 1')

        param_values = self.problem._fixed_params_values
        self.assertEqual(len(param_values), 6)
        self.assertEqual(param_values[1], 0)
        self.assertEqual(param_values[4], 2)

        # Free all parameters again
        name_value_dict = dict({
            'myokit.lambda_1': None,
            'myokit.drug_concentration': None})

        self.problem.fix_parameters(name_value_dict)

        self.assertEqual(self.problem.n_parameters(), 6)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 6)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Noise param 1')

        self.assertIsNone(self.problem._fixed_params_values)
        self.assertIsNone(self.problem._fixed_params_mask)

    def test_fix_parameters_bad_input(self):
        name_value_dict = dict({
            'myokit.lambda_1': 2,
            'myokit.kappa': None})

        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'The mechanistic'):
            problem.fix_parameters(name_value_dict)

        # No error model set
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem.set_mechanistic_model(model)

        with self.assertRaisesRegex(ValueError, 'The error model'):
            problem.fix_parameters(name_value_dict)

    def test_get_log_posteriors(self):
        # Create posterior with no fixed parameters
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.log_likelihoods)
        self.problem.set_log_prior(self.log_priors)
        posteriors = self.problem.get_log_posteriors()

        self.assertEqual(len(posteriors), 3)
        self.assertEqual(posteriors[0].n_parameters(), 6)
        self.assertEqual(posteriors[1].n_parameters(), 6)
        self.assertEqual(posteriors[2].n_parameters(), 6)

        # Fixe some parameters
        name_value_dict = dict({
            'myokit.drug_concentration': 0,
            'myokit.kappa': 1})
        self.problem.fix_parameters(name_value_dict)
        self.problem.set_log_prior(self.log_priors[:-2])
        posteriors = self.problem.get_log_posteriors()

        self.assertEqual(len(posteriors), 3)
        self.assertEqual(posteriors[0].n_parameters(), 4)
        self.assertEqual(posteriors[1].n_parameters(), 4)
        self.assertEqual(posteriors[2].n_parameters(), 4)

    def test_get_log_posteriors_bad_input(self):
        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'The mechanistic'):
            problem.get_log_posteriors()

        # No error model set
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem.set_mechanistic_model(model)

        with self.assertRaisesRegex(ValueError, 'The error model'):
            problem.get_log_posteriors()

        # No log-prior set
        problem.set_error_model(self.log_likelihoods)

        with self.assertRaisesRegex(ValueError, 'The log-prior'):
            problem.get_log_posteriors()

    def test_set_error_model(self):
        # Map error model to output automatically
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.log_likelihoods)

        error_models = self.problem._error_model
        n_ids = 3
        self.assertEqual(len(error_models), n_ids)
        self.assertIsInstance(error_models[0], pints.GaussianLogLikelihood)
        self.assertIsInstance(error_models[1], pints.GaussianLogLikelihood)
        self.assertIsInstance(error_models[2], pints.GaussianLogLikelihood)

        self.assertEqual(self.problem.n_parameters(), 6)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Noise param 1')

        # Set error model-output mapping explicitly
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker 1', 'Biomarker 2'])
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        output_biomarker_map = dict({
            'myokit.tumour_volume': 'Biomarker 2',
            'myokit.drug_concentration': 'Biomarker 1'})
        problem.set_mechanistic_model(model, output_biomarker_map)
        log_likelihoods = [
            pints.GaussianLogLikelihood,
            pints.MultiplicativeGaussianLogLikelihood]
        outputs = ['myokit.tumour_volume', 'myokit.drug_concentration']

        # TODO: To be implemented in complementary PR
        with self.assertRaisesRegex(NotImplementedError, 'Fitting'):
            problem.set_error_model(log_likelihoods, outputs)

    def test_set_error_model_bad_input(self):
        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_error_model(self.log_likelihoods)

        # Log-likelihoods have the wrong type
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem.set_mechanistic_model(model)

        likelihoods = [str, float, int]
        with self.assertRaisesRegex(ValueError, 'The log-likelihoods are'):
            problem.set_error_model(likelihoods)

        # Number of likelihoods does not match the number of outputs
        likelihoods = [pints.GaussianLogLikelihood, pints.AR1LogLikelihood]
        with self.assertRaisesRegex(ValueError, 'The number of log-'):
            problem.set_error_model(likelihoods)

        # The specified outputs do not match the model outputs
        likelihoods = [pints.GaussianLogLikelihood]
        outputs = ['wrong', 'outputs']
        with self.assertRaisesRegex(ValueError, 'The specified outputs'):
            problem.set_error_model(likelihoods, outputs)

    def test_set_log_prior(self):
        # Map priors to parameters automatically
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.log_likelihoods)
        priors = self.log_priors
        self.problem.set_log_prior(priors)

        self.assertIsInstance(self.problem._log_prior, pints.ComposedLogPrior)

        # Specify prior parameter map explicitly
        parameters = [
            'myokit.kappa',
            'Noise param 1',
            'myokit.tumour_volume',
            'myokit.lambda_1',
            'myokit.drug_concentration',
            'myokit.lambda_0']
        self.problem.set_log_prior(priors, parameters)

        self.assertIsInstance(self.problem._log_prior, pints.ComposedLogPrior)

    def test_set_log_prior_bad_input(self):
        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_log_prior(self.log_priors)

        # No error model set
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem.set_mechanistic_model(model)

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_log_prior(self.log_priors)

        # Wrong log-prior type
        problem.set_error_model(self.log_likelihoods)
        priors = ['Wrong', 'type']
        with self.assertRaisesRegex(ValueError, 'All marginal log-priors'):
            problem.set_log_prior(priors)

        # Number of log priors does not match number of parameters
        priors = [pints.GaussianLogPrior(0, 1), pints.HalfCauchyLogPrior(0, 1)]
        with self.assertRaisesRegex(ValueError, 'One marginal log-prior'):
            problem.set_log_prior(priors)

        # Dimensionality of joint log-pior does not match number of params
        prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(0, 1), pints.GaussianLogPrior(0, 1))
        priors = [
            prior,
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1)]
        with self.assertRaisesRegex(ValueError, 'The joint log-prior'):
            problem.set_log_prior(priors)

        # Specified parameter names do not match the model parameters
        params = ['wrong', 'params']
        with self.assertRaisesRegex(ValueError, 'The specified parameter'):
            problem.set_log_prior(self.log_priors, params)

    def test_set_mechanistic_model(self):
        # Set output biomarker mapping automatically
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker 1'])

        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem.set_mechanistic_model(model=model)

        self.assertEqual(problem._mechanistic_model, model)
        outputs = problem._mechanistic_model.outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'myokit.tumour_volume')
        self.assertEqual(len(problem._biom_keys), 1)
        self.assertEqual(problem._biom_keys[0], 'Biomarker 1')

        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker 1', 'Biomarker 2'])

        # Set output biomarker mapping explicitly
        output_biomarker_map = dict({
            'myokit.tumour_volume': 'Biomarker 2',
            'myokit.drug_concentration': 'Biomarker 1'})

        problem.set_mechanistic_model(model, output_biomarker_map)

        self.assertEqual(problem._mechanistic_model, model)
        outputs = problem._mechanistic_model.outputs()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], 'myokit.tumour_volume')
        self.assertEqual(outputs[1], 'myokit.drug_concentration')
        self.assertEqual(len(problem._biom_keys), 2)
        self.assertEqual(problem._biom_keys[0], 'Biomarker 2')
        self.assertEqual(problem._biom_keys[1], 'Biomarker 1')

    def test_set_mechanistic_model_bad_input(self):
        # Wrong model type
        model = 'some model'

        with self.assertRaisesRegex(ValueError, 'The model has to be'):
            self.problem.set_mechanistic_model(model=model)

        # Wrong number of model outputs
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker 1', 'Biomarker 2'])

        with self.assertRaisesRegex(ValueError, 'The model does not have'):
            problem.set_mechanistic_model(model=self.model)

        # Wrong map type
        output_biomarker_map = 'bad map'

        with self.assertRaisesRegex(ValueError, 'The output-biomarker'):
            self.problem.set_mechanistic_model(
                self.model, output_biomarker_map)

        # Map does not contain biomarkers specfied by the dataset
        output_biomarker_map = dict({'Some variable': 'Some biomarker'})

        with self.assertRaisesRegex(ValueError, 'The provided output'):
            self.problem.set_mechanistic_model(
                self.model, output_biomarker_map)


class TestInverseProblem(unittest.TestCase):
    """
    Tests the erlotinib.InverseProblem class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        cls.times = [1, 2, 3, 4, 5]
        cls.values = [1, 2, 3, 4, 5]

        # Set up inverse problem
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.model = erlo.PharmacodynamicModel(path)
        cls.problem = erlo.InverseProblem(cls.model, cls.times, cls.values)

    def test_bad_model_input(self):
        model = 'bad model'

        with self.assertRaisesRegex(ValueError, 'Model has to be an instance'):
            erlo.InverseProblem(model, self.times, self.values)

    def test_bad_times_input(self):
        times = [-1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, 'Times cannot be negative.'):
            erlo.InverseProblem(self.model, times, self.values)

        times = [5, 4, 3, 2, 1]
        with self.assertRaisesRegex(ValueError, 'Times must be increasing.'):
            erlo.InverseProblem(self.model, times, self.values)

    def test_bad_values_input(self):
        values = [1, 2, 3, 4, 5, 6, 7]
        with self.assertRaisesRegex(ValueError, 'Values array must have'):
            erlo.InverseProblem(self.model, self.times, values)

        values = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        with self.assertRaisesRegex(ValueError, 'Values array must have'):
            erlo.InverseProblem(self.model, self.times, values)

    def test_evaluate(self):
        parameters = [0.1, 1, 1, 1, 1]
        output = self.problem.evaluate(parameters)

        n_times = 5
        n_outputs = 1
        self.assertEqual(output.shape, (n_times, n_outputs))

    def test_evaluateS1(self):
        parameters = [0.1, 1, 1, 1, 1]
        with self.assertRaises(NotImplementedError):
            self.problem.evaluateS1(parameters)

    def test_n_ouputs(self):
        self.assertEqual(self.problem.n_outputs(), 1)

    def test_n_parameters(self):
        self.assertEqual(self.problem.n_parameters(), 5)

    def test_n_times(self):
        n_times = len(self.times)
        self.assertEqual(self.problem.n_times(), n_times)

    def test_times(self):
        times = self.problem.times()
        n_times = len(times)

        self.assertEqual(n_times, 5)

        self.assertEqual(times[0], self.times[0])
        self.assertEqual(times[1], self.times[1])
        self.assertEqual(times[2], self.times[2])
        self.assertEqual(times[3], self.times[3])
        self.assertEqual(times[4], self.times[4])

    def test_values(self):
        values = self.problem.values()

        n_times = 5
        n_outputs = 1
        self.assertEqual(values.shape, (n_times, n_outputs))

        self.assertEqual(values[0], self.values[0])
        self.assertEqual(values[1], self.values[1])
        self.assertEqual(values[2], self.values[2])
        self.assertEqual(values[3], self.values[3])
        self.assertEqual(values[4], self.values[4])


if __name__ == '__main__':
    unittest.main()
