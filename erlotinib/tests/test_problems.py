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


class TestProblemModellingControllerPDProblem(unittest.TestCase):
    """
    Tests the erlotinib.ProblemModellingController class on a PD modelling
    problem.
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
        cls.error_models = [pints.GaussianLogLikelihood]
        cls.log_priors = [
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1)]
        n_params = 6
        cls.pop_models = [erlo.PooledModel] * n_params

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
        self.problem.set_error_model(self.error_models)
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

        # Fix parameters before setting a population model
        name_value_dict = dict({
            'myokit.tumour_volume': 1,
            'myokit.drug_concentration': 0,
            'myokit.kappa': 1,
            'myokit.lambda_1': 2})
        self.problem.fix_parameters(name_value_dict)
        self.problem.set_population_model(
            pop_models=[erlo.HeterogeneousModel, erlo.PooledModel])

        n_ids = 3
        self.assertEqual(self.problem.n_parameters(), n_ids + 1)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 4)  # n_ids + 1
        self.assertEqual(param_names[0], 'ID 0: myokit.lambda_0')
        self.assertEqual(param_names[1], 'ID 1: myokit.lambda_0')
        self.assertEqual(param_names[2], 'ID 2: myokit.lambda_0')
        self.assertEqual(param_names[3], 'Pooled Noise param 1')

        self.assertIsNone(self.problem._fixed_params_values)
        self.assertIsNone(self.problem._fixed_params_mask)

        # Fix parameters after setting a population model
        name_value_dict = dict({
            'ID 1: myokit.lambda_0': 1,
            'ID 2: myokit.lambda_0': 4})
        self.problem.fix_parameters(name_value_dict)

        self.assertEqual(self.problem.n_parameters(), 2)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 2)
        self.assertEqual(param_names[0], 'ID 0: myokit.lambda_0')
        self.assertEqual(param_names[1], 'Pooled Noise param 1')

        param_values = self.problem._fixed_params_values
        self.assertEqual(len(param_values), 4)
        self.assertEqual(param_values[1], 1)
        self.assertEqual(param_values[2], 4)

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
        self.problem.set_error_model(self.error_models)
        self.problem.set_log_prior(self.log_priors)
        posteriors = self.problem.get_log_posteriors()

        self.assertEqual(len(posteriors), 3)
        self.assertEqual(posteriors[0].n_parameters(), 6)
        self.assertEqual(posteriors[0].get_id(), '0')
        self.assertEqual(posteriors[1].n_parameters(), 6)
        self.assertEqual(posteriors[1].get_id(), '1')
        self.assertEqual(posteriors[2].n_parameters(), 6)
        self.assertEqual(posteriors[2].get_id(), '2')

        # Fixe some parameters
        name_value_dict = dict({
            'myokit.drug_concentration': 0,
            'myokit.kappa': 1})
        self.problem.fix_parameters(name_value_dict)
        self.problem.set_log_prior(self.log_priors[:-2])
        posteriors = self.problem.get_log_posteriors()

        self.assertEqual(len(posteriors), 3)
        self.assertEqual(posteriors[0].n_parameters(), 4)
        self.assertEqual(posteriors[0].get_id(), '0')
        self.assertEqual(posteriors[1].n_parameters(), 4)
        self.assertEqual(posteriors[1].get_id(), '1')
        self.assertEqual(posteriors[2].n_parameters(), 4)
        self.assertEqual(posteriors[2].get_id(), '2')

    def test_get_dosing_regimens(self):
        regimens = self.problem.get_dosing_regimens()

        self.assertIsNone(regimens)

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
        problem.set_error_model(self.error_models)

        with self.assertRaisesRegex(ValueError, 'The log-prior'):
            problem.get_log_posteriors()

    def test_get_parameter_names(self):
        # Test with a mechanistic-error model pair only
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)

        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 6)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Noise param 1')

        # Check that also works with exclude pop params flag
        param_names = self.problem.get_parameter_names(exclude_pop_model=True)
        self.assertEqual(len(param_names), 6)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Noise param 1')

        # Test with fixed parameters
        name_value_dict = dict({
            'myokit.drug_concentration': 0,
            'myokit.kappa': 1})
        self.problem.fix_parameters(name_value_dict)

        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 4)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.lambda_0')
        self.assertEqual(param_names[2], 'myokit.lambda_1')
        self.assertEqual(param_names[3], 'Noise param 1')

        # Test with setting a population model
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)
        pop_models = [
            erlo.PooledModel,
            erlo.PooledModel,
            erlo.HeterogeneousModel,
            erlo.PooledModel,
            erlo.PooledModel,
            erlo.PooledModel]
        self.problem.set_population_model(pop_models)

        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 2 + 3 + 3)
        self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[2], 'ID 0: myokit.kappa')
        self.assertEqual(param_names[3], 'ID 1: myokit.kappa')
        self.assertEqual(param_names[4], 'ID 2: myokit.kappa')
        self.assertEqual(param_names[5], 'Pooled myokit.lambda_0')
        self.assertEqual(param_names[6], 'Pooled myokit.lambda_1')
        self.assertEqual(param_names[7], 'Pooled Noise param 1')

        # Test whether exclude population model works
        param_names = self.problem.get_parameter_names(exclude_pop_model=True)
        self.assertEqual(len(param_names), 6)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Noise param 1')

    def test_set_error_model(self):
        # Map error model to output automatically
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)

        log_likelihoods = list(self.problem._log_likelihoods.values())
        n_ids = 3
        self.assertEqual(len(log_likelihoods), n_ids)
        self.assertIsInstance(log_likelihoods[0], pints.GaussianLogLikelihood)
        self.assertIsInstance(log_likelihoods[1], pints.GaussianLogLikelihood)
        self.assertIsInstance(log_likelihoods[2], pints.GaussianLogLikelihood)

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
            pints.GaussianLogLikelihood]
        outputs = ['myokit.tumour_volume', 'myokit.drug_concentration']
        problem.set_error_model(log_likelihoods, outputs)

        log_likelihoods = list(problem._log_likelihoods.values())
        n_ids = 3
        self.assertEqual(len(log_likelihoods), n_ids)
        self.assertIsInstance(log_likelihoods[0], pints.PooledLogPDF)
        self.assertIsInstance(log_likelihoods[1], pints.PooledLogPDF)
        self.assertIsInstance(log_likelihoods[2], pints.PooledLogPDF)

        self.assertEqual(problem.n_parameters(), 7)
        param_names = problem.get_parameter_names()
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Noise param 1')
        self.assertEqual(param_names[6], 'Noise param 2')

    def test_set_error_model_bad_input(self):
        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_error_model(self.error_models)

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

        # The likelihoods need arguments for instantiation
        likelihoods = [pints.GaussianKnownSigmaLogLikelihood]
        with self.assertRaisesRegex(ValueError, 'Pints.ProblemLoglikelihoods'):
            problem.set_error_model(likelihoods)

        # Non-identical error models
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

        with self.assertRaisesRegex(ValueError, 'Only structurally identical'):
            problem.set_error_model(log_likelihoods, outputs)

    def test_set_log_prior(self):
        # Map priors to parameters automatically
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)
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
        problem.set_error_model(self.error_models)
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

    def test_set_population_model(self):
        # Map population model to parameters automatically
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)
        self.problem.set_population_model(self.pop_models)

        pop_models = self.problem._population_models
        n_parameters = 6
        self.assertEqual(len(pop_models), n_parameters)
        self.assertIsInstance(pop_models[0], erlo.PooledModel)
        self.assertIsInstance(pop_models[1], erlo.PooledModel)
        self.assertIsInstance(pop_models[2], erlo.PooledModel)
        self.assertIsInstance(pop_models[3], erlo.PooledModel)
        self.assertIsInstance(pop_models[4], erlo.PooledModel)
        self.assertIsInstance(pop_models[5], erlo.PooledModel)

        self.assertEqual(self.problem.n_parameters(), 6)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[2], 'Pooled myokit.kappa')
        self.assertEqual(param_names[3], 'Pooled myokit.lambda_0')
        self.assertEqual(param_names[4], 'Pooled myokit.lambda_1')
        self.assertEqual(param_names[5], 'Pooled Noise param 1')

        # Map population model to parameters explicitly (with blanks)
        pop_models = [erlo.PooledModel] * 5  # 6 paramaters in total
        params = [
            'myokit.drug_concentration',
            'myokit.kappa',
            'myokit.lambda_0',
            'myokit.lambda_1',
            'Noise param 1']
        self.problem.set_population_model(pop_models, params)

        pop_models = self.problem._population_models
        n_parameters = 6
        self.assertEqual(len(pop_models), n_parameters)
        self.assertIsInstance(pop_models[0], erlo.HeterogeneousModel)
        self.assertIsInstance(pop_models[1], erlo.PooledModel)
        self.assertIsInstance(pop_models[2], erlo.PooledModel)
        self.assertIsInstance(pop_models[3], erlo.PooledModel)
        self.assertIsInstance(pop_models[4], erlo.PooledModel)
        self.assertIsInstance(pop_models[5], erlo.PooledModel)

        n_ids = 3
        n_parameters = n_ids + (n_parameters - 1)  # 3 Heterogeneous + 5 Pooled
        self.assertEqual(self.problem.n_parameters(), 8)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(param_names[0], 'ID 0: myokit.tumour_volume')
        self.assertEqual(param_names[1], 'ID 1: myokit.tumour_volume')
        self.assertEqual(param_names[2], 'ID 2: myokit.tumour_volume')
        self.assertEqual(param_names[3], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[4], 'Pooled myokit.kappa')
        self.assertEqual(param_names[5], 'Pooled myokit.lambda_0')
        self.assertEqual(param_names[6], 'Pooled myokit.lambda_1')
        self.assertEqual(param_names[7], 'Pooled Noise param 1')

    def test_set_population_model_bad_input(self):
        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_population_model(self.pop_models)

        # No error model set
        problem.set_mechanistic_model(self.model)

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_population_model(self.pop_models)

        # Population models have the wrong type
        problem.set_error_model(self.error_models)

        pop_models = [str, float, int]
        with self.assertRaisesRegex(ValueError, 'The provided population'):
            problem.set_error_model(pop_models)

        #TODO:
        # Number of likelihoods does not match the number of outputs
        likelihoods = [pints.GaussianLogLikelihood, pints.AR1LogLikelihood]
        with self.assertRaisesRegex(ValueError, 'The number of log-'):
            problem.set_error_model(likelihoods)

        # The specified outputs do not match the model outputs
        likelihoods = [pints.GaussianLogLikelihood]
        outputs = ['wrong', 'outputs']
        with self.assertRaisesRegex(ValueError, 'The specified outputs'):
            problem.set_error_model(likelihoods, outputs)

        # The likelihoods need arguments for instantiation
        likelihoods = [pints.GaussianKnownSigmaLogLikelihood]
        with self.assertRaisesRegex(ValueError, 'Pints.ProblemLoglikelihoods'):
            problem.set_error_model(likelihoods)

        # Non-identical error models
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

        with self.assertRaisesRegex(ValueError, 'Only structurally identical'):
            problem.set_error_model(log_likelihoods, outputs)


class TestProblemModellingControllerPKProblem(unittest.TestCase):
    """
    Tests the erlotinib.ProblemModellingController class on a PK modelling
    problem.
    """
    @classmethod
    def setUpClass(cls):
        # Create test dataset
        ids = [0, 0, 0, 1, 1, 1, 2, 2]
        times = [0, 1, 2, 2, np.nan, 4, 1, 3]
        plasma_conc = [np.nan, 0.3, 0.2, 0.5, 0.1, 0.2, 0.234, np.nan]
        dose = [3.4, np.nan, np.nan, 0.5, 0.5, 0.5, np.nan, np.nan]
        cls.data = pd.DataFrame({
            'ID': ids,
            'Time': times,
            'Biomarker': plasma_conc,
            'Biomarker 2': plasma_conc,
            'Dose': dose})

        # Create test problem
        cls.problem = erlo.ProblemModellingController(
            cls.data, dose_key='Dose')

        # Create test model
        path = erlo.ModelLibrary().one_compartment_pk_model()
        cls.model = erlo.PharmacokineticModel(path)
        cls.model.set_administration(compartment='central', direct=False)
        cls.error_models = [pints.GaussianLogLikelihood]
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

        # Wrong dose key
        with self.assertRaisesRegex(ValueError, 'Data does not have'):
            erlo.ProblemModellingController(data=self.data, dose_key='New key')

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

        # Rename biomarker key
        data = self.data.rename(columns={'Dose': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        erlo.ProblemModellingController(
            data=data, dose_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            erlo.ProblemModellingController(
                data=data, dose_key='SOME WRONG KEY')

        # Test that it works with no dose key
        erlo.ProblemModellingController(data=data)

    def test_fix_parameters(self):
        # Fix model parameters
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)
        name_value_dict = dict({
            'central.drug_amount': 0,
            'myokit.elimination_rate': 1})

        self.problem.fix_parameters(name_value_dict)

        self.assertEqual(self.problem.n_parameters(), 4)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 4)
        self.assertEqual(param_names[0], 'dose.drug_amount')
        self.assertEqual(param_names[1], 'central.size')
        self.assertEqual(param_names[2], 'dose.absorption_rate')
        self.assertEqual(param_names[3], 'Noise param 1')

        param_values = self.problem._fixed_params_values
        self.assertEqual(len(param_values), 6)
        self.assertEqual(param_values[0], 0)
        self.assertEqual(param_values[4], 1)

        # Free elimination rate and fix dose drug amount
        name_value_dict = dict({
            'dose.drug_amount': 0,
            'myokit.elimination_rate': None})

        self.problem.fix_parameters(name_value_dict)

        self.assertEqual(self.problem.n_parameters(), 4)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 4)
        self.assertEqual(param_names[0], 'central.size')
        self.assertEqual(param_names[1], 'dose.absorption_rate')
        self.assertEqual(param_names[2], 'myokit.elimination_rate')
        self.assertEqual(param_names[3], 'Noise param 1')

        param_values = self.problem._fixed_params_values
        self.assertEqual(len(param_values), 6)
        self.assertEqual(param_values[0], 0)
        self.assertEqual(param_values[1], 0)

        # Free all parameters again
        name_value_dict = dict({
            'dose.drug_amount': None,
            'central.drug_amount': None})

        self.problem.fix_parameters(name_value_dict)

        self.assertEqual(self.problem.n_parameters(), 6)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(len(param_names), 6)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'dose.drug_amount')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'dose.absorption_rate')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'Noise param 1')

        self.assertIsNone(self.problem._fixed_params_values)
        self.assertIsNone(self.problem._fixed_params_mask)

    def test_fix_parameters_bad_input(self):
        name_value_dict = dict({
            'dose.drug_amount': 0,
            'myokit.elimination_rate': None})

        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'The mechanistic'):
            problem.fix_parameters(name_value_dict)

        # No error model set
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        problem.set_mechanistic_model(model)

        with self.assertRaisesRegex(ValueError, 'The error model'):
            problem.fix_parameters(name_value_dict)

    def test_get_dosing_regimens(self):
        regimens = self.problem.get_dosing_regimens()

        n_ids = 3
        ids = list(regimens.keys())
        self.assertEqual(len(ids), n_ids)
        self.assertEqual(ids[0], '0')
        self.assertEqual(ids[1], '1')
        self.assertEqual(ids[2], '2')

        # Check protocols
        events = regimens['0'].events()
        self.assertEqual(len(events), 1)

        event = events[0]
        dose = 3.4
        duration = 0.01
        self.assertEqual(event.level(), dose / duration)
        self.assertEqual(event.start(), 0)
        self.assertEqual(event.duration(), duration)
        self.assertEqual(event.period(), 0)
        self.assertEqual(event.multiplier(), 0)

        events = regimens['1'].events()
        self.assertEqual(len(events), 2)

        event = events[0]
        dose = 0.5
        duration = 0.01
        self.assertEqual(event.level(), dose / duration)
        self.assertEqual(event.start(), 2)
        self.assertEqual(event.duration(), duration)
        self.assertEqual(event.period(), 0)
        self.assertEqual(event.multiplier(), 0)

        event = events[1]
        dose = 0.5
        duration = 0.01
        self.assertEqual(event.level(), dose / duration)
        self.assertEqual(event.start(), 4)
        self.assertEqual(event.duration(), duration)
        self.assertEqual(event.period(), 0)
        self.assertEqual(event.multiplier(), 0)

        events = regimens['2'].events()
        self.assertEqual(len(events), 0)

    def test_get_log_posteriors(self):
        # Create posterior with no fixed parameters
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)
        self.problem.set_log_prior(self.log_priors)
        posteriors = self.problem.get_log_posteriors()

        self.assertEqual(len(posteriors), 3)
        self.assertEqual(posteriors[0].n_parameters(), 6)
        self.assertEqual(posteriors[0].get_id(), '0')
        self.assertEqual(posteriors[1].n_parameters(), 6)
        self.assertEqual(posteriors[1].get_id(), '1')
        self.assertEqual(posteriors[2].n_parameters(), 6)
        self.assertEqual(posteriors[2].get_id(), '2')

        # Fix some parameters
        name_value_dict = dict({
            'central.drug_amount': 0,
            'dose.absorption_rate': 1})
        self.problem.fix_parameters(name_value_dict)
        self.problem.set_log_prior(self.log_priors[:-2])
        posteriors = self.problem.get_log_posteriors()

        self.assertEqual(len(posteriors), 3)
        self.assertEqual(posteriors[0].n_parameters(), 4)
        self.assertEqual(posteriors[0].get_id(), '0')
        self.assertEqual(posteriors[1].n_parameters(), 4)
        self.assertEqual(posteriors[1].get_id(), '1')
        self.assertEqual(posteriors[2].n_parameters(), 4)
        self.assertEqual(posteriors[2].get_id(), '2')

    def test_get_log_posteriors_bad_input(self):
        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'The mechanistic'):
            problem.get_log_posteriors()

        # No error model set
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        problem.set_mechanistic_model(model)

        with self.assertRaisesRegex(ValueError, 'The error model'):
            problem.get_log_posteriors()

        # No log-prior set
        problem.set_error_model(self.error_models)

        with self.assertRaisesRegex(ValueError, 'The log-prior'):
            problem.get_log_posteriors()

    def test_set_error_model(self):
        # Map error model to output automatically
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)

        log_likelihoods = list(self.problem._log_likelihoods.values())
        n_ids = 3
        self.assertEqual(len(log_likelihoods), n_ids)
        self.assertIsInstance(log_likelihoods[0], pints.GaussianLogLikelihood)
        self.assertIsInstance(log_likelihoods[1], pints.GaussianLogLikelihood)
        self.assertIsInstance(log_likelihoods[2], pints.GaussianLogLikelihood)

        self.assertEqual(self.problem.n_parameters(), 6)
        param_names = self.problem.get_parameter_names()
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'dose.drug_amount')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'dose.absorption_rate')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'Noise param 1')

        # Set error model-output mapping explicitly
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker', 'Biomarker 2'])
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        output_biomarker_map = dict({
            'dose.drug_amount': 'Biomarker 2',
            'central.drug_concentration': 'Biomarker'})
        problem.set_mechanistic_model(model, output_biomarker_map)
        log_likelihoods = [
            pints.GaussianLogLikelihood,
            pints.GaussianLogLikelihood]
        outputs = ['dose.drug_amount', 'central.drug_concentration']
        problem.set_error_model(log_likelihoods, outputs)

        log_likelihoods = list(problem._log_likelihoods.values())
        n_ids = 3
        self.assertEqual(len(log_likelihoods), n_ids)
        self.assertIsInstance(log_likelihoods[0], pints.PooledLogPDF)
        self.assertIsInstance(log_likelihoods[1], pints.PooledLogPDF)
        self.assertIsInstance(log_likelihoods[2], pints.PooledLogPDF)

        self.assertEqual(problem.n_parameters(), 7)
        param_names = problem.get_parameter_names()
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'dose.drug_amount')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'dose.absorption_rate')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'Noise param 1')
        self.assertEqual(param_names[6], 'Noise param 2')

    def test_set_error_model_bad_input(self):
        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_error_model(self.error_models)

        # Log-likelihoods have the wrong type
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
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

        # The likelihoods need arguments for instantiation
        likelihoods = [pints.GaussianKnownSigmaLogLikelihood]
        with self.assertRaisesRegex(ValueError, 'Pints.ProblemLoglikelihoods'):
            problem.set_error_model(likelihoods)

        # Non-identical error models
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker', 'Biomarker 2'])
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        output_biomarker_map = dict({
            'central.drug_concentration': 'Biomarker 2',
            'dose.drug_amount': 'Biomarker'})
        problem.set_mechanistic_model(model, output_biomarker_map)
        log_likelihoods = [
            pints.GaussianLogLikelihood,
            pints.MultiplicativeGaussianLogLikelihood]
        outputs = ['central.drug_concentration', 'dose.drug_amount']

        with self.assertRaisesRegex(ValueError, 'Only structurally identical'):
            problem.set_error_model(log_likelihoods, outputs)

    def test_set_log_prior(self):
        # Map priors to parameters automatically
        self.problem.set_mechanistic_model(self.model)
        self.problem.set_error_model(self.error_models)
        priors = self.log_priors
        self.problem.set_log_prior(priors)

        self.assertIsInstance(self.problem._log_prior, pints.ComposedLogPrior)

        # Specify prior parameter map explicitly
        parameters = [
            'central.size',
            'Noise param 1',
            'myokit.elimination_rate',
            'central.drug_amount',
            'dose.absorption_rate',
            'dose.drug_amount']
        self.problem.set_log_prior(priors, parameters)

        self.assertIsInstance(self.problem._log_prior, pints.ComposedLogPrior)

    def test_set_log_prior_bad_input(self):
        # No mechanistic model set
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker'])

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_log_prior(self.log_priors)

        # No error model set
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        problem.set_mechanistic_model(model)

        with self.assertRaisesRegex(ValueError, 'Before setting'):
            problem.set_log_prior(self.log_priors)

        # Wrong log-prior type
        problem.set_error_model(self.error_models)
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
            self.data, dose_key='Dose')

        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration(compartment='central', direct=False)
        problem.set_mechanistic_model(model)

        self.assertEqual(problem._mechanistic_model, model)
        outputs = problem._mechanistic_model.outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'central.drug_concentration')
        self.assertEqual(len(problem._biom_keys), 1)
        self.assertEqual(problem._biom_keys[0], 'Biomarker')

        # Set output biomarker mapping explicitly
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker', 'Biomarker 2'], dose_key='Dose')
        output_biomarker_map = dict({
            'dose.drug_amount': 'Biomarker 2',
            'central.drug_concentration': 'Biomarker'})

        problem.set_mechanistic_model(model, output_biomarker_map)

        self.assertEqual(problem._mechanistic_model, model)
        outputs = problem._mechanistic_model.outputs()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], 'dose.drug_amount')
        self.assertEqual(outputs[1], 'central.drug_concentration')
        self.assertEqual(len(problem._biom_keys), 2)
        self.assertEqual(problem._biom_keys[0], 'Biomarker 2')
        self.assertEqual(problem._biom_keys[1], 'Biomarker')

    def test_set_mechanistic_model_bad_input(self):
        # Wrong model type
        model = 'some model'

        with self.assertRaisesRegex(ValueError, 'The model has to be'):
            self.problem.set_mechanistic_model(model=model)

        # Wrong number of model outputs
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker', 'Biomarker 2'])

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
