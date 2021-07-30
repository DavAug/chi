#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy
import unittest

import numpy as np
import pandas as pd
import pints

import chi
from chi.library import ModelLibrary


class TestProblemModellingControllerPDProblem(unittest.TestCase):
    """
    Tests the chi.ProblemModellingController class on a PD modelling
    problem.
    """
    @classmethod
    def setUpClass(cls):
        # Create test dataset
        ids_v = [0, 0, 0, 1, 1, 1, 2, 2]
        times_v = [0, 1, 2, 2, np.nan, 4, 1, 3]
        volumes = [np.nan, 0.3, 0.2, 0.5, 0.1, 0.2, 0.234, np.nan]
        ids_c = [0, 0, 1, 1]
        times_c = [0, 1, 2, np.nan]
        cytokines = [3.4, 0.3, 0.5, np.nan]
        ids_d = [0, 1, 1, 1, 2, 2]
        times_d = [0, np.nan, 4, 1, 3, 3]
        dose = [3.4, np.nan, 0.5, 0.5, np.nan, np.nan]
        duration = [0.01, np.nan, 0.31, np.nan, 0.5, np.nan]
        cls.data = pd.DataFrame({
            'ID': ids_v + ids_c + ids_d,
            'Time': times_v + times_c + times_d,
            'Biomarker':
                ['Tumour volume']*8 + ['IL 6']*4 + [np.nan]*6,
            'Measurement': volumes + cytokines + [np.nan]*6,
            'Dose': [np.nan]*12 + dose,
            'Duration': [np.nan]*12 + duration})

        # Test case I: create PD modelling problem
        lib = ModelLibrary()
        path = lib.tumour_growth_inhibition_model_koch()
        cls.pd_model = chi.PharmacodynamicModel(path)
        cls.error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()
        cls.pd_problem = chi.ProblemModellingController(
            cls.pd_model, cls.error_model)

        # Test case II: create PKPD modelling problem
        lib = ModelLibrary()
        path = lib.erlotinib_tumour_growth_inhibition_model()
        cls.pkpd_model = chi.PharmacokineticModel(path)
        cls.pkpd_model.set_outputs([
            'central.drug_concentration',
            'myokit.tumour_volume'])
        cls.error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel(),
            chi.ConstantAndMultiplicativeGaussianErrorModel()]
        cls.pkpd_problem = chi.ProblemModellingController(
            cls.pkpd_model, cls.error_models,
            outputs=[
                'central.drug_concentration',
                'myokit.tumour_volume'])

    def test_bad_input(self):
        # Mechanistic model has wrong type
        mechanistic_model = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The mechanistic model'):
            chi.ProblemModellingController(
                mechanistic_model, self.error_model)

        # Error model has wrong type
        error_model = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'Error models have to be'):
            chi.ProblemModellingController(
                self.pd_model, error_model)

        error_models = ['wrong', 'type']
        with self.assertRaisesRegex(TypeError, 'Error models have to be'):
            chi.ProblemModellingController(
                self.pd_model, error_models)

        # Wrong number of error models
        error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()
        with self.assertRaisesRegex(ValueError, 'Wrong number of error'):
            chi.ProblemModellingController(
                self.pkpd_model, error_model)

        error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel(),
            chi.ConstantAndMultiplicativeGaussianErrorModel()]
        with self.assertRaisesRegex(ValueError, 'Wrong number of error'):
            chi.ProblemModellingController(
                self.pd_model, error_models)

    def test_fix_parameters(self):
        # Test case I: PD model
        # Fix model parameters
        name_value_dict = dict({
            'myokit.drug_concentration': 0,
            'Sigma base': 1})
        self.pd_problem.fix_parameters(name_value_dict)

        self.assertEqual(self.pd_problem.get_n_parameters(), 5)
        param_names = self.pd_problem.get_parameter_names()
        self.assertEqual(len(param_names), 5)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.kappa')
        self.assertEqual(param_names[2], 'myokit.lambda_0')
        self.assertEqual(param_names[3], 'myokit.lambda_1')
        self.assertEqual(param_names[4], 'Sigma rel.')

        # Free and fix a parameter
        name_value_dict = dict({
            'myokit.lambda_1': 2,
            'Sigma base': None})
        self.pd_problem.fix_parameters(name_value_dict)

        self.assertEqual(self.pd_problem.get_n_parameters(), 5)
        param_names = self.pd_problem.get_parameter_names()
        self.assertEqual(len(param_names), 5)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.kappa')
        self.assertEqual(param_names[2], 'myokit.lambda_0')
        self.assertEqual(param_names[3], 'Sigma base')
        self.assertEqual(param_names[4], 'Sigma rel.')

        # Free all parameters again
        name_value_dict = dict({
            'myokit.lambda_1': None,
            'myokit.drug_concentration': None})
        self.pd_problem.fix_parameters(name_value_dict)

        self.assertEqual(self.pd_problem.get_n_parameters(), 7)
        param_names = self.pd_problem.get_parameter_names()
        self.assertEqual(len(param_names), 7)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Sigma base')
        self.assertEqual(param_names[6], 'Sigma rel.')

        # Fix parameters before setting a population model
        problem = copy.copy(self.pd_problem)
        name_value_dict = dict({
            'myokit.tumour_volume': 1,
            'myokit.drug_concentration': 0,
            'myokit.kappa': 1,
            'myokit.lambda_1': 2})
        problem.fix_parameters(name_value_dict)
        problem.set_population_model(
            pop_models=[
                chi.HeterogeneousModel(),
                chi.PooledModel(),
                chi.LogNormalModel()])
        problem.set_data(
            self.data,
            output_biomarker_dict={'myokit.tumour_volume': 'Tumour volume'})

        n_ids = 3
        self.assertEqual(problem.get_n_parameters(), 2 * n_ids + 1 + 2)
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 9)
        self.assertEqual(param_names[0], 'ID 0: myokit.lambda_0')
        self.assertEqual(param_names[1], 'ID 1: myokit.lambda_0')
        self.assertEqual(param_names[2], 'ID 2: myokit.lambda_0')
        self.assertEqual(param_names[3], 'Pooled Sigma base')
        self.assertEqual(param_names[4], 'ID 0: Sigma rel.')
        self.assertEqual(param_names[5], 'ID 1: Sigma rel.')
        self.assertEqual(param_names[6], 'ID 2: Sigma rel.')
        self.assertEqual(param_names[7], 'Mean log Sigma rel.')
        self.assertEqual(param_names[8], 'Std. log Sigma rel.')

        # Fix parameters after setting a population model
        # (Only population models can be fixed)
        name_value_dict = dict({
            'ID 1: myokit.lambda_0': 1,
            'ID 2: myokit.lambda_0': 4,
            'Pooled Sigma base': 2})
        problem.fix_parameters(name_value_dict)

        # self.assertEqual(problem.get_n_parameters(), 8)
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 8)
        self.assertEqual(param_names[0], 'ID 0: myokit.lambda_0')
        self.assertEqual(param_names[1], 'ID 1: myokit.lambda_0')
        self.assertEqual(param_names[2], 'ID 2: myokit.lambda_0')
        self.assertEqual(param_names[3], 'ID 0: Sigma rel.')
        self.assertEqual(param_names[4], 'ID 1: Sigma rel.')
        self.assertEqual(param_names[5], 'ID 2: Sigma rel.')
        self.assertEqual(param_names[6], 'Mean log Sigma rel.')
        self.assertEqual(param_names[7], 'Std. log Sigma rel.')

        # Test case II: PKPD model
        # Fix model parameters
        name_value_dict = dict({
            'myokit.kappa': 0,
            'central.drug_concentration Sigma base': 1})
        self.pkpd_problem.fix_parameters(name_value_dict)

        self.assertEqual(self.pkpd_problem.get_n_parameters(), 9)
        param_names = self.pkpd_problem.get_parameter_names()
        self.assertEqual(len(param_names), 9)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'myokit.tumour_volume')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'myokit.critical_volume')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'myokit.lambda')
        self.assertEqual(
            param_names[6], 'central.drug_concentration Sigma rel.')
        self.assertEqual(param_names[7], 'myokit.tumour_volume Sigma base')
        self.assertEqual(param_names[8], 'myokit.tumour_volume Sigma rel.')

        # Free and fix a parameter
        name_value_dict = dict({
            'myokit.lambda': 2,
            'myokit.kappa': None})
        self.pkpd_problem.fix_parameters(name_value_dict)

        self.assertEqual(self.pkpd_problem.get_n_parameters(), 9)
        param_names = self.pkpd_problem.get_parameter_names()
        self.assertEqual(len(param_names), 9)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'myokit.tumour_volume')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'myokit.critical_volume')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'myokit.kappa')
        self.assertEqual(
            param_names[6], 'central.drug_concentration Sigma rel.')
        self.assertEqual(param_names[7], 'myokit.tumour_volume Sigma base')
        self.assertEqual(param_names[8], 'myokit.tumour_volume Sigma rel.')

        # Free all parameters again
        name_value_dict = dict({
            'myokit.lambda': None,
            'central.drug_concentration Sigma base': None})
        self.pkpd_problem.fix_parameters(name_value_dict)

        self.assertEqual(self.pkpd_problem.get_n_parameters(), 11)
        param_names = self.pkpd_problem.get_parameter_names()
        self.assertEqual(len(param_names), 11)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'myokit.tumour_volume')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'myokit.critical_volume')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'myokit.kappa')
        self.assertEqual(param_names[6], 'myokit.lambda')
        self.assertEqual(
            param_names[7], 'central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[8], 'central.drug_concentration Sigma rel.')
        self.assertEqual(param_names[9], 'myokit.tumour_volume Sigma base')
        self.assertEqual(param_names[10], 'myokit.tumour_volume Sigma rel.')

    def test_fix_parameters_bad_input(self):
        # Input is not a dictionary
        name_value_dict = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The name-value dictionary'):
            self.pd_problem.fix_parameters(name_value_dict)

    def test_get_dosing_regimens(self):
        # Test case I: PD problem
        problem = copy.deepcopy(self.pd_problem)

        # No data has been set
        regimens = problem.get_dosing_regimens()
        self.assertIsNone(regimens)

        # Set data, but because PD model, no dosing regimen can be set
        problem.set_data(self.data, {'myokit.tumour_volume': 'Tumour volume'})
        regimens = problem.get_dosing_regimens()
        self.assertIsNone(regimens)

        # Test case II: PKPD problem
        problem = copy.deepcopy(self.pkpd_problem)

        # No data has been set
        regimens = problem.get_dosing_regimens()
        self.assertIsNone(regimens)

        # Data has been set, but duration is ignored
        problem.set_data(
            self.data,
            output_biomarker_dict={
                'myokit.tumour_volume': 'Tumour volume',
                'central.drug_concentration': 'IL 6'},
            dose_duration_key=None)
        regimens = problem.get_dosing_regimens()
        self.assertIsInstance(regimens, dict)

        # Data has been set with duration information
        problem.set_data(
            self.data,
            output_biomarker_dict={
                'myokit.tumour_volume': 'Tumour volume',
                'central.drug_concentration': 'IL 6'})
        regimens = problem.get_dosing_regimens()
        self.assertIsInstance(regimens, dict)

    def test_get_log_prior(self):
        # Log-prior is extensively tested with get_log_posterior
        # method
        self.assertIsNone(self.pd_problem.get_log_prior())

    def test_get_log_posterior(self):
        # Test case I: Create posterior with no fixed parameters
        problem = copy.deepcopy(self.pd_problem)

        # Set data which does not provide measurements for all IDs
        problem.set_data(
            self.data,
            output_biomarker_dict={'myokit.tumour_volume': 'IL 6'})
        problem.set_log_prior([
            pints.HalfCauchyLogPrior(0, 1)]*7)

        # Get all posteriors
        posteriors = problem.get_log_posterior()

        self.assertEqual(len(posteriors), 2)
        self.assertEqual(posteriors[0].n_parameters(), 7)
        self.assertEqual(posteriors[0].get_id(), 'ID 0')
        self.assertEqual(posteriors[1].n_parameters(), 7)
        self.assertEqual(posteriors[1].get_id(), 'ID 1')

        # Set data that has measurements for all IDs
        problem.set_data(
            self.data,
            output_biomarker_dict={'myokit.tumour_volume': 'Tumour volume'})
        problem.set_log_prior([
            pints.HalfCauchyLogPrior(0, 1)]*7)

        # Get all posteriors
        posteriors = problem.get_log_posterior()

        self.assertEqual(len(posteriors), 3)
        self.assertEqual(posteriors[0].n_parameters(), 7)
        self.assertEqual(posteriors[0].get_id(), 'ID 0')
        self.assertEqual(posteriors[1].n_parameters(), 7)
        self.assertEqual(posteriors[1].get_id(), 'ID 1')
        self.assertEqual(posteriors[2].n_parameters(), 7)
        self.assertEqual(posteriors[2].get_id(), 'ID 2')

        # Get only one posterior
        posterior = problem.get_log_posterior(individual='0')

        self.assertIsInstance(posterior, chi.LogPosterior)
        self.assertEqual(posterior.n_parameters(), 7)
        self.assertEqual(posterior.get_id(), 'ID 0')

        # Test case II: Fix some parameters
        name_value_dict = dict({
            'myokit.drug_concentration': 0,
            'myokit.kappa': 1})
        problem.fix_parameters(name_value_dict)
        problem.set_log_prior([
            pints.HalfCauchyLogPrior(0, 1)]*5)

        # Get all posteriors
        posteriors = problem.get_log_posterior()

        self.assertEqual(len(posteriors), 3)
        self.assertEqual(posteriors[0].n_parameters(), 5)
        self.assertEqual(posteriors[0].get_id(), 'ID 0')
        self.assertEqual(posteriors[1].n_parameters(), 5)
        self.assertEqual(posteriors[1].get_id(), 'ID 1')
        self.assertEqual(posteriors[2].n_parameters(), 5)
        self.assertEqual(posteriors[2].get_id(), 'ID 2')

        # Get only one posterior
        posterior = problem.get_log_posterior(individual='1')

        self.assertIsInstance(posterior, chi.LogPosterior)
        self.assertEqual(posterior.n_parameters(), 5)
        self.assertEqual(posterior.get_id(), 'ID 1')

        # Set a population model
        pop_models = [
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel()]
        problem.set_population_model(pop_models)
        problem.set_log_prior([
            pints.HalfCauchyLogPrior(0, 1)]*8)
        posterior = problem.get_log_posterior()

        self.assertIsInstance(posterior, chi.HierarchicalLogPosterior)
        self.assertEqual(posterior.n_parameters(), 11)

        names = posterior.get_parameter_names()
        ids = posterior.get_id()
        self.assertEqual(len(names), 11)
        self.assertEqual(len(ids), 11)

        self.assertEqual(names[0], 'Pooled myokit.tumour_volume')
        self.assertIsNone(ids[0])
        self.assertEqual(names[1], 'myokit.lambda_0')
        self.assertEqual(ids[1], 'ID 0')
        self.assertEqual(names[2], 'myokit.lambda_0')
        self.assertEqual(ids[2], 'ID 1')
        self.assertEqual(names[3], 'myokit.lambda_0')
        self.assertEqual(ids[3], 'ID 2')
        self.assertEqual(names[4], 'Pooled myokit.lambda_1')
        self.assertIsNone(ids[4])
        self.assertEqual(names[5], 'Pooled Sigma base')
        self.assertIsNone(ids[0])
        self.assertEqual(names[6], 'Sigma rel.')
        self.assertEqual(ids[6], 'ID 0')
        self.assertEqual(names[7], 'Sigma rel.')
        self.assertEqual(ids[7], 'ID 1')
        self.assertEqual(names[8], 'Sigma rel.')
        self.assertEqual(ids[8], 'ID 2')
        self.assertEqual(names[9], 'Mean log Sigma rel.')
        self.assertIsNone(ids[9])
        self.assertEqual(names[10], 'Std. log Sigma rel.')
        self.assertIsNone(ids[10])

        # Make sure that selecting an individual is ignored for population
        # models
        posterior = problem.get_log_posterior(individual='some individual')

        self.assertIsInstance(posterior, chi.HierarchicalLogPosterior)
        self.assertEqual(posterior.n_parameters(), 11)

        names = posterior.get_parameter_names()
        ids = posterior.get_id()
        self.assertEqual(len(names), 11)
        self.assertEqual(len(ids), 11)

        self.assertEqual(names[0], 'Pooled myokit.tumour_volume')
        self.assertIsNone(ids[0])
        self.assertEqual(names[1], 'myokit.lambda_0')
        self.assertEqual(ids[1], 'ID 0')
        self.assertEqual(names[2], 'myokit.lambda_0')
        self.assertEqual(ids[2], 'ID 1')
        self.assertEqual(names[3], 'myokit.lambda_0')
        self.assertEqual(ids[3], 'ID 2')
        self.assertEqual(names[4], 'Pooled myokit.lambda_1')
        self.assertIsNone(ids[4])
        self.assertEqual(names[5], 'Pooled Sigma base')
        self.assertIsNone(ids[0])
        self.assertEqual(names[6], 'Sigma rel.')
        self.assertEqual(ids[6], 'ID 0')
        self.assertEqual(names[7], 'Sigma rel.')
        self.assertEqual(ids[7], 'ID 1')
        self.assertEqual(names[8], 'Sigma rel.')
        self.assertEqual(ids[8], 'ID 2')
        self.assertEqual(names[9], 'Mean log Sigma rel.')
        self.assertIsNone(ids[9])
        self.assertEqual(names[10], 'Std. log Sigma rel.')
        self.assertIsNone(ids[10])

    def test_get_log_posteriors_bad_input(self):
        problem = copy.deepcopy(self.pd_problem)

        # No log-prior has been set
        problem.set_data(
            self.data,
            output_biomarker_dict={'myokit.tumour_volume': 'Tumour volume'})

        with self.assertRaisesRegex(ValueError, 'The log-prior has not'):
            problem.get_log_posterior()

        # The selected individual does not exist
        individual = 'Not existent'
        problem.set_log_prior([pints.HalfCauchyLogPrior(0, 1)]*7)

        with self.assertRaisesRegex(ValueError, 'The individual cannot'):
            problem.get_log_posterior(individual)

    def test_get_n_parameters(self):
        # Test case I: PD model
        # Test case I.1: No population model
        # Test default flag
        problem = copy.deepcopy(self.pd_problem)
        n_parameters = problem.get_n_parameters()
        self.assertEqual(n_parameters, 7)

        # Test exclude population model True
        n_parameters = problem.get_n_parameters(exclude_pop_model=True)
        self.assertEqual(n_parameters, 7)

        # Test exclude bottom-level model True
        n_parameters = problem.get_n_parameters(exclude_bottom_level=True)
        self.assertEqual(n_parameters, 7)

        # Test case I.2: Population model
        pop_models = [
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(),
            chi.LogNormalModel()]
        problem.set_population_model(pop_models)
        n_parameters = problem.get_n_parameters()
        self.assertEqual(n_parameters, 8)

        # Test exclude population model True
        n_parameters = problem.get_n_parameters(exclude_pop_model=True)
        self.assertEqual(n_parameters, 7)

        # Test exclude bottom-level model True
        n_parameters = problem.get_n_parameters(exclude_bottom_level=True)
        self.assertEqual(n_parameters, 8)

        # Test case I.3: Set data
        problem.set_data(
            self.data,
            output_biomarker_dict={'myokit.tumour_volume': 'Tumour volume'})
        n_parameters = problem.get_n_parameters()
        self.assertEqual(n_parameters, 17)

        # Test exclude population model True
        n_parameters = problem.get_n_parameters(exclude_pop_model=True)
        self.assertEqual(n_parameters, 7)

        # Test exclude bottom-level model True
        n_parameters = problem.get_n_parameters(exclude_bottom_level=True)
        self.assertEqual(n_parameters, 11)

        # Test case II: PKPD model
        # Test case II.1: No population model
        # Test default flag
        problem = copy.deepcopy(self.pkpd_problem)
        n_parameters = problem.get_n_parameters()
        self.assertEqual(n_parameters, 11)

        # Test exclude population model True
        n_parameters = problem.get_n_parameters(exclude_pop_model=True)
        self.assertEqual(n_parameters, 11)

        # Test exclude bottom-level model True
        n_parameters = problem.get_n_parameters(exclude_bottom_level=True)
        self.assertEqual(n_parameters, 11)

        # Test case II.2: Population model
        pop_models = [
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel()]
        problem.set_population_model(pop_models)
        n_parameters = problem.get_n_parameters()
        self.assertEqual(n_parameters, 12)

        # Test exclude population model True
        n_parameters = problem.get_n_parameters(exclude_pop_model=True)
        self.assertEqual(n_parameters, 11)

        # Test exclude bottom-level model True
        n_parameters = problem.get_n_parameters(exclude_bottom_level=True)
        self.assertEqual(n_parameters, 12)

        # Test case II.3: Set data
        problem.set_data(
            self.data,
            output_biomarker_dict={
                'myokit.tumour_volume': 'Tumour volume',
                'central.drug_concentration': 'IL 6'})
        n_parameters = problem.get_n_parameters()
        self.assertEqual(n_parameters, 21)

        # Test exclude population model True
        n_parameters = problem.get_n_parameters(exclude_pop_model=True)
        self.assertEqual(n_parameters, 11)

        # Test exclude bottom-level model True
        n_parameters = problem.get_n_parameters(exclude_bottom_level=True)
        self.assertEqual(n_parameters, 15)

    def test_get_parameter_names(self):
        # Test case I: PD model
        problem = copy.deepcopy(self.pd_problem)

        # Test case I.1: No population model
        # Test default flag
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 7)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Sigma base')
        self.assertEqual(param_names[6], 'Sigma rel.')

        # Check that also works with exclude pop params flag
        param_names = problem.get_parameter_names(exclude_pop_model=True)
        self.assertEqual(len(param_names), 7)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Sigma base')
        self.assertEqual(param_names[6], 'Sigma rel.')

        # Check that also works with exclude bottom-level flag
        param_names = problem.get_parameter_names(exclude_bottom_level=True)
        self.assertEqual(len(param_names), 7)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Sigma base')
        self.assertEqual(param_names[6], 'Sigma rel.')

        # Test case I.2: Population model
        pop_models = [
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(),
            chi.LogNormalModel()]
        problem.set_population_model(pop_models)
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 8)
        self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[2], 'Pooled myokit.lambda_0')
        self.assertEqual(param_names[3], 'Pooled myokit.lambda_1')
        self.assertEqual(param_names[4], 'Mean log Sigma base')
        self.assertEqual(param_names[5], 'Std. log Sigma base')
        self.assertEqual(param_names[6], 'Mean log Sigma rel.')
        self.assertEqual(param_names[7], 'Std. log Sigma rel.')

        # Test exclude population model True
        param_names = problem.get_parameter_names(exclude_pop_model=True)
        self.assertEqual(len(param_names), 7)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Sigma base')
        self.assertEqual(param_names[6], 'Sigma rel.')

        # Test exclude bottom-level True
        param_names = problem.get_parameter_names(exclude_bottom_level=True)
        self.assertEqual(len(param_names), 8)
        self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[2], 'Pooled myokit.lambda_0')
        self.assertEqual(param_names[3], 'Pooled myokit.lambda_1')
        self.assertEqual(param_names[4], 'Mean log Sigma base')
        self.assertEqual(param_names[5], 'Std. log Sigma base')
        self.assertEqual(param_names[6], 'Mean log Sigma rel.')
        self.assertEqual(param_names[7], 'Std. log Sigma rel.')

        # Test case I.3: Set data
        problem.set_data(
            self.data,
            output_biomarker_dict={'myokit.tumour_volume': 'Tumour volume'})
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 17)
        self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[2], 'ID 0: myokit.kappa')
        self.assertEqual(param_names[3], 'ID 1: myokit.kappa')
        self.assertEqual(param_names[4], 'ID 2: myokit.kappa')
        self.assertEqual(param_names[5], 'Pooled myokit.lambda_0')
        self.assertEqual(param_names[6], 'Pooled myokit.lambda_1')
        self.assertEqual(param_names[7], 'ID 0: Sigma base')
        self.assertEqual(param_names[8], 'ID 1: Sigma base')
        self.assertEqual(param_names[9], 'ID 2: Sigma base')
        self.assertEqual(param_names[10], 'Mean log Sigma base')
        self.assertEqual(param_names[11], 'Std. log Sigma base')
        self.assertEqual(param_names[12], 'ID 0: Sigma rel.')
        self.assertEqual(param_names[13], 'ID 1: Sigma rel.')
        self.assertEqual(param_names[14], 'ID 2: Sigma rel.')
        self.assertEqual(param_names[15], 'Mean log Sigma rel.')
        self.assertEqual(param_names[16], 'Std. log Sigma rel.')

        # Test exclude population model True
        param_names = problem.get_parameter_names(exclude_pop_model=True)
        self.assertEqual(len(param_names), 7)
        self.assertEqual(param_names[0], 'myokit.tumour_volume')
        self.assertEqual(param_names[1], 'myokit.drug_concentration')
        self.assertEqual(param_names[2], 'myokit.kappa')
        self.assertEqual(param_names[3], 'myokit.lambda_0')
        self.assertEqual(param_names[4], 'myokit.lambda_1')
        self.assertEqual(param_names[5], 'Sigma base')
        self.assertEqual(param_names[6], 'Sigma rel.')

        # Test exclude bottom-level True
        param_names = problem.get_parameter_names(exclude_bottom_level=True)
        self.assertEqual(len(param_names), 11)
        self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[2], 'ID 0: myokit.kappa')
        self.assertEqual(param_names[3], 'ID 1: myokit.kappa')
        self.assertEqual(param_names[4], 'ID 2: myokit.kappa')
        self.assertEqual(param_names[5], 'Pooled myokit.lambda_0')
        self.assertEqual(param_names[6], 'Pooled myokit.lambda_1')
        self.assertEqual(param_names[7], 'Mean log Sigma base')
        self.assertEqual(param_names[8], 'Std. log Sigma base')
        self.assertEqual(param_names[9], 'Mean log Sigma rel.')
        self.assertEqual(param_names[10], 'Std. log Sigma rel.')

        # Test case II: PKPD model
        problem = copy.deepcopy(self.pkpd_problem)

        # Test case II.1: No population model
        # Test default flag
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 11)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'myokit.tumour_volume')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'myokit.critical_volume')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'myokit.kappa')
        self.assertEqual(param_names[6], 'myokit.lambda')
        self.assertEqual(
            param_names[7], 'central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[8], 'central.drug_concentration Sigma rel.')
        self.assertEqual(param_names[9], 'myokit.tumour_volume Sigma base')
        self.assertEqual(param_names[10], 'myokit.tumour_volume Sigma rel.')

        # Test exclude population model True
        param_names = problem.get_parameter_names(exclude_pop_model=True)
        self.assertEqual(len(param_names), 11)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'myokit.tumour_volume')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'myokit.critical_volume')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'myokit.kappa')
        self.assertEqual(param_names[6], 'myokit.lambda')
        self.assertEqual(
            param_names[7], 'central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[8], 'central.drug_concentration Sigma rel.')
        self.assertEqual(param_names[9], 'myokit.tumour_volume Sigma base')
        self.assertEqual(param_names[10], 'myokit.tumour_volume Sigma rel.')

        # Test exclude population model True
        param_names = problem.get_parameter_names(exclude_bottom_level=True)
        self.assertEqual(len(param_names), 11)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'myokit.tumour_volume')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'myokit.critical_volume')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'myokit.kappa')
        self.assertEqual(param_names[6], 'myokit.lambda')
        self.assertEqual(
            param_names[7], 'central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[8], 'central.drug_concentration Sigma rel.')
        self.assertEqual(param_names[9], 'myokit.tumour_volume Sigma base')
        self.assertEqual(param_names[10], 'myokit.tumour_volume Sigma rel.')

        # Test case II.2: Population model
        pop_models = [
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel()]
        problem.set_population_model(pop_models)
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 12)
        self.assertEqual(param_names[0], 'Pooled central.drug_amount')
        self.assertEqual(param_names[1], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[2], 'Pooled myokit.critical_volume')
        self.assertEqual(param_names[3], 'Pooled myokit.elimination_rate')
        self.assertEqual(param_names[4], 'Mean log myokit.kappa')
        self.assertEqual(param_names[5], 'Std. log myokit.kappa')
        self.assertEqual(param_names[6], 'Mean log myokit.lambda')
        self.assertEqual(param_names[7], 'Std. log myokit.lambda')
        self.assertEqual(
            param_names[8], 'Pooled central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[9], 'Pooled central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[10], 'Pooled myokit.tumour_volume Sigma base')
        self.assertEqual(
            param_names[11], 'Pooled myokit.tumour_volume Sigma rel.')

        # Test exclude population model True
        param_names = problem.get_parameter_names(exclude_pop_model=True)
        self.assertEqual(len(param_names), 11)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'myokit.tumour_volume')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'myokit.critical_volume')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'myokit.kappa')
        self.assertEqual(param_names[6], 'myokit.lambda')
        self.assertEqual(
            param_names[7], 'central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[8], 'central.drug_concentration Sigma rel.')
        self.assertEqual(param_names[9], 'myokit.tumour_volume Sigma base')
        self.assertEqual(param_names[10], 'myokit.tumour_volume Sigma rel.')

        # Test exclude bottom-level True
        param_names = problem.get_parameter_names(exclude_bottom_level=True)
        self.assertEqual(len(param_names), 12)
        self.assertEqual(param_names[0], 'Pooled central.drug_amount')
        self.assertEqual(param_names[1], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[2], 'Pooled myokit.critical_volume')
        self.assertEqual(param_names[3], 'Pooled myokit.elimination_rate')
        self.assertEqual(param_names[4], 'Mean log myokit.kappa')
        self.assertEqual(param_names[5], 'Std. log myokit.kappa')
        self.assertEqual(param_names[6], 'Mean log myokit.lambda')
        self.assertEqual(param_names[7], 'Std. log myokit.lambda')
        self.assertEqual(
            param_names[8], 'Pooled central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[9], 'Pooled central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[10], 'Pooled myokit.tumour_volume Sigma base')
        self.assertEqual(
            param_names[11], 'Pooled myokit.tumour_volume Sigma rel.')

        # Test case II.3: Set data
        problem.set_data(
            self.data,
            output_biomarker_dict={
                'myokit.tumour_volume': 'Tumour volume',
                'central.drug_concentration': 'IL 6'})
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 21)
        self.assertEqual(param_names[0], 'Pooled central.drug_amount')
        self.assertEqual(param_names[1], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[2], 'ID 0: central.size')
        self.assertEqual(param_names[3], 'ID 1: central.size')
        self.assertEqual(param_names[4], 'ID 2: central.size')
        self.assertEqual(param_names[5], 'Pooled myokit.critical_volume')
        self.assertEqual(param_names[6], 'Pooled myokit.elimination_rate')
        self.assertEqual(param_names[7], 'ID 0: myokit.kappa')
        self.assertEqual(param_names[8], 'ID 1: myokit.kappa')
        self.assertEqual(param_names[9], 'ID 2: myokit.kappa')
        self.assertEqual(param_names[10], 'Mean log myokit.kappa')
        self.assertEqual(param_names[11], 'Std. log myokit.kappa')
        self.assertEqual(param_names[12], 'ID 0: myokit.lambda')
        self.assertEqual(param_names[13], 'ID 1: myokit.lambda')
        self.assertEqual(param_names[14], 'ID 2: myokit.lambda')
        self.assertEqual(param_names[15], 'Mean log myokit.lambda')
        self.assertEqual(param_names[16], 'Std. log myokit.lambda')
        self.assertEqual(
            param_names[17], 'Pooled central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[18], 'Pooled central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[19], 'Pooled myokit.tumour_volume Sigma base')
        self.assertEqual(
            param_names[20], 'Pooled myokit.tumour_volume Sigma rel.')

        # Test exclude population model True
        param_names = problem.get_parameter_names(exclude_pop_model=True)
        self.assertEqual(len(param_names), 11)
        self.assertEqual(param_names[0], 'central.drug_amount')
        self.assertEqual(param_names[1], 'myokit.tumour_volume')
        self.assertEqual(param_names[2], 'central.size')
        self.assertEqual(param_names[3], 'myokit.critical_volume')
        self.assertEqual(param_names[4], 'myokit.elimination_rate')
        self.assertEqual(param_names[5], 'myokit.kappa')
        self.assertEqual(param_names[6], 'myokit.lambda')
        self.assertEqual(
            param_names[7], 'central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[8], 'central.drug_concentration Sigma rel.')
        self.assertEqual(param_names[9], 'myokit.tumour_volume Sigma base')
        self.assertEqual(param_names[10], 'myokit.tumour_volume Sigma rel.')

        # Test exclude bottom-level True
        param_names = problem.get_parameter_names(exclude_bottom_level=True)
        self.assertEqual(len(param_names), 15)
        self.assertEqual(param_names[0], 'Pooled central.drug_amount')
        self.assertEqual(param_names[1], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[2], 'ID 0: central.size')
        self.assertEqual(param_names[3], 'ID 1: central.size')
        self.assertEqual(param_names[4], 'ID 2: central.size')
        self.assertEqual(param_names[5], 'Pooled myokit.critical_volume')
        self.assertEqual(param_names[6], 'Pooled myokit.elimination_rate')
        self.assertEqual(param_names[7], 'Mean log myokit.kappa')
        self.assertEqual(param_names[8], 'Std. log myokit.kappa')
        self.assertEqual(param_names[9], 'Mean log myokit.lambda')
        self.assertEqual(param_names[10], 'Std. log myokit.lambda')
        self.assertEqual(
            param_names[11], 'Pooled central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[12], 'Pooled central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[13], 'Pooled myokit.tumour_volume Sigma base')
        self.assertEqual(
            param_names[14], 'Pooled myokit.tumour_volume Sigma rel.')

    def test_get_predictive_model(self):
        # Test case I: PD model
        problem = copy.deepcopy(self.pd_problem)

        # Test case I.1: No population model
        predictive_model = problem.get_predictive_model()
        self.assertIsInstance(predictive_model, chi.PredictiveModel)

        # Exclude population model
        predictive_model = problem.get_predictive_model(
            exclude_pop_model=True)
        self.assertIsInstance(predictive_model, chi.PredictiveModel)

        # Test case I.2: Population model
        problem.set_population_model([
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(),
            chi.LogNormalModel()])
        predictive_model = problem.get_predictive_model()
        self.assertIsInstance(
            predictive_model, chi.PredictivePopulationModel)

        # Exclude population model
        predictive_model = problem.get_predictive_model(
            exclude_pop_model=True)
        self.assertNotIsInstance(
            predictive_model, chi.PredictivePopulationModel)
        self.assertIsInstance(predictive_model, chi.PredictiveModel)

        # Test case II: PKPD model
        problem = copy.deepcopy(self.pkpd_problem)

        # Test case II.1: No population model
        predictive_model = problem.get_predictive_model()
        self.assertIsInstance(predictive_model, chi.PredictiveModel)

        # Exclude population model
        predictive_model = problem.get_predictive_model(
            exclude_pop_model=True)
        self.assertIsInstance(predictive_model, chi.PredictiveModel)

        # Test case II.2: Population model
        problem.set_population_model([
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel()])
        predictive_model = problem.get_predictive_model()
        self.assertIsInstance(
            predictive_model, chi.PredictivePopulationModel)

        # Exclude population model
        predictive_model = problem.get_predictive_model(
            exclude_pop_model=True)
        self.assertNotIsInstance(
            predictive_model, chi.PredictivePopulationModel)
        self.assertIsInstance(predictive_model, chi.PredictiveModel)

    def test_set_data(self):
        # Set data with explicit output-biomarker map
        problem = copy.deepcopy(self.pd_problem)
        output_biomarker_dict = {'myokit.tumour_volume': 'Tumour volume'}
        problem.set_data(self.data, output_biomarker_dict)

        # Set data with implicit output-biomarker map
        mask = self.data['Biomarker'] == 'Tumour volume'
        problem.set_data(self.data[mask])

    def test_set_data_bad_input(self):
        # Data has the wrong type
        data = 'Wrong type'
        with self.assertRaisesRegex(TypeError, 'Data has to be a'):
            self.pd_problem.set_data(data)

        # Data has the wrong ID key
        data = self.data.rename(columns={'ID': 'Some key'})
        with self.assertRaisesRegex(ValueError, 'Data does not have the'):
            self.pkpd_problem.set_data(data)

        # Data has the wrong time key
        data = self.data.rename(columns={'Time': 'Some key'})
        with self.assertRaisesRegex(ValueError, 'Data does not have the'):
            self.pkpd_problem.set_data(data)

        # Data has the wrong biomarker key
        data = self.data.rename(columns={'Biomarker': 'Some key'})
        with self.assertRaisesRegex(ValueError, 'Data does not have the'):
            self.pkpd_problem.set_data(data)

        # Data has the wrong measurement key
        data = self.data.rename(columns={'Measurement': 'Some key'})
        with self.assertRaisesRegex(ValueError, 'Data does not have the'):
            self.pkpd_problem.set_data(data)

        # Data has the wrong dose key
        data = self.data.rename(columns={'Dose': 'Some key'})
        with self.assertRaisesRegex(ValueError, 'Data does not have the'):
            self.pkpd_problem.set_data(data)

        # Data has the wrong duration key
        data = self.data.rename(columns={'Duration': 'Some key'})
        with self.assertRaisesRegex(ValueError, 'Data does not have the'):
            self.pkpd_problem.set_data(data)

        # The output-biomarker map does not contain a model output
        output_biomarker_dict = {'some ouput': 'some biomarker'}
        with self.assertRaisesRegex(ValueError, 'The output <central.drug'):
            self.pkpd_problem.set_data(self.data, output_biomarker_dict)

        # The output-biomarker map references a biomarker that is not in the
        # dataframe
        output_biomarker_dict = {'myokit.tumour_volume': 'some biomarker'}
        with self.assertRaisesRegex(ValueError, 'The biomarker <some'):
            self.pd_problem.set_data(self.data, output_biomarker_dict)

        # The model outputs and dataframe biomarker cannot be trivially mapped
        with self.assertRaisesRegex(ValueError, 'The biomarker <central.'):
            self.pkpd_problem.set_data(self.data)

    def test_set_log_prior(self):
        # Test case I: PD model
        problem = copy.deepcopy(self.pd_problem)
        problem.set_data(self.data, {'myokit.tumour_volume': 'Tumour volume'})
        log_priors = [pints.HalfCauchyLogPrior(0, 1)] * 7

        # Map priors to parameters automatically
        problem.set_log_prior(log_priors)

        # Specify prior parameter map explicitly
        param_names = [
            'myokit.kappa',
            'Sigma base',
            'Sigma rel.',
            'myokit.tumour_volume',
            'myokit.lambda_1',
            'myokit.drug_concentration',
            'myokit.lambda_0']
        problem.set_log_prior(log_priors, param_names)

    def test_set_log_prior_bad_input(self):
        problem = copy.deepcopy(self.pd_problem)

        # No data has been set
        with self.assertRaisesRegex(ValueError, 'The data has not'):
            problem.set_log_prior('some prior')

        # Wrong log-prior type
        problem.set_data(self.data, {'myokit.tumour_volume': 'Tumour volume'})
        log_priors = ['Wrong', 'type']
        with self.assertRaisesRegex(ValueError, 'All marginal log-priors'):
            problem.set_log_prior(log_priors)

        # Number of log priors does not match number of parameters
        log_priors = [
            pints.GaussianLogPrior(0, 1), pints.HalfCauchyLogPrior(0, 1)]
        with self.assertRaisesRegex(ValueError, 'One marginal log-prior'):
            problem.set_log_prior(log_priors)

        # Dimensionality of joint log-pior does not match number of parameters
        prior = pints.ComposedLogPrior(
            pints.GaussianLogPrior(0, 1), pints.GaussianLogPrior(0, 1))
        log_priors = [
            prior,
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1)]
        with self.assertRaisesRegex(ValueError, 'The joint log-prior'):
            problem.set_log_prior(log_priors)

        # Specified parameter names do not match the model parameters
        params = ['wrong', 'params']
        log_priors = [pints.HalfCauchyLogPrior(0, 1)] * 7
        with self.assertRaisesRegex(ValueError, 'The specified parameter'):
            problem.set_log_prior(log_priors, params)

    def test_set_population_model(self):
        # Test case I: PD model
        problem = copy.deepcopy(self.pd_problem)
        problem.set_data(self.data, {'myokit.tumour_volume': 'Tumour volume'})
        pop_models = [
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel()]

        # Test case I.1: Don't specify order
        problem.set_population_model(pop_models)

        self.assertEqual(problem.get_n_parameters(), 13)
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 13)
        self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[2], 'ID 0: myokit.kappa')
        self.assertEqual(param_names[3], 'ID 1: myokit.kappa')
        self.assertEqual(param_names[4], 'ID 2: myokit.kappa')
        self.assertEqual(param_names[5], 'Pooled myokit.lambda_0')
        self.assertEqual(param_names[6], 'Pooled myokit.lambda_1')
        self.assertEqual(param_names[7], 'Pooled Sigma base')
        self.assertEqual(param_names[8], 'ID 0: Sigma rel.')
        self.assertEqual(param_names[9], 'ID 1: Sigma rel.')
        self.assertEqual(param_names[10], 'ID 2: Sigma rel.')
        self.assertEqual(param_names[11], 'Mean log Sigma rel.')
        self.assertEqual(param_names[12], 'Std. log Sigma rel.')

        # Test case I.2: Specify order
        parameter_names = [
            'Sigma base',
            'myokit.drug_concentration',
            'myokit.lambda_1',
            'myokit.kappa',
            'myokit.tumour_volume',
            'Sigma rel.',
            'myokit.lambda_0']
        problem.set_population_model(pop_models, parameter_names)

        self.assertEqual(problem.get_n_parameters(), 13)
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 13)
        self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
        self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
        self.assertEqual(param_names[2], 'Pooled myokit.kappa')
        self.assertEqual(param_names[3], 'ID 0: myokit.lambda_0')
        self.assertEqual(param_names[4], 'ID 1: myokit.lambda_0')
        self.assertEqual(param_names[5], 'ID 2: myokit.lambda_0')
        self.assertEqual(param_names[6], 'Mean log myokit.lambda_0')
        self.assertEqual(param_names[7], 'Std. log myokit.lambda_0')
        self.assertEqual(param_names[8], 'ID 0: myokit.lambda_1')
        self.assertEqual(param_names[9], 'ID 1: myokit.lambda_1')
        self.assertEqual(param_names[10], 'ID 2: myokit.lambda_1')
        self.assertEqual(param_names[11], 'Pooled Sigma base')
        self.assertEqual(param_names[12], 'Pooled Sigma rel.')

        # Test case II: PKPD model
        problem = copy.deepcopy(self.pkpd_problem)
        problem.set_data(
            self.data,
            output_biomarker_dict={
                'central.drug_concentration': 'IL 6',
                'myokit.tumour_volume': 'Tumour volume'})
        pop_models = [
            chi.LogNormalModel(),
            chi.LogNormalModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.LogNormalModel()]

        # Test case I.1: Don't specify order
        problem.set_population_model(pop_models)

        self.assertEqual(problem.get_n_parameters(), 33)
        param_names = problem.get_parameter_names()
        self.assertEqual(len(param_names), 33)
        self.assertEqual(param_names[0], 'ID 0: central.drug_amount')
        self.assertEqual(param_names[1], 'ID 1: central.drug_amount')
        self.assertEqual(param_names[2], 'ID 2: central.drug_amount')
        self.assertEqual(param_names[3], 'Mean log central.drug_amount')
        self.assertEqual(param_names[4], 'Std. log central.drug_amount')
        self.assertEqual(param_names[5], 'ID 0: myokit.tumour_volume')
        self.assertEqual(param_names[6], 'ID 1: myokit.tumour_volume')
        self.assertEqual(param_names[7], 'ID 2: myokit.tumour_volume')
        self.assertEqual(param_names[8], 'Mean log myokit.tumour_volume')
        self.assertEqual(param_names[9], 'Std. log myokit.tumour_volume')
        self.assertEqual(param_names[10], 'ID 0: central.size')
        self.assertEqual(param_names[11], 'ID 1: central.size')
        self.assertEqual(param_names[12], 'ID 2: central.size')
        self.assertEqual(param_names[13], 'Mean log central.size')
        self.assertEqual(param_names[14], 'Std. log central.size')
        self.assertEqual(param_names[15], 'Pooled myokit.critical_volume')
        self.assertEqual(param_names[16], 'Pooled myokit.elimination_rate')
        self.assertEqual(param_names[17], 'ID 0: myokit.kappa')
        self.assertEqual(param_names[18], 'ID 1: myokit.kappa')
        self.assertEqual(param_names[19], 'ID 2: myokit.kappa')
        self.assertEqual(param_names[20], 'Pooled myokit.lambda')
        self.assertEqual(
            param_names[21], 'Pooled central.drug_concentration Sigma base')
        self.assertEqual(
            param_names[22], 'ID 0: central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[23], 'ID 1: central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[24], 'ID 2: central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[25], 'Mean log central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[26], 'Std. log central.drug_concentration Sigma rel.')
        self.assertEqual(
            param_names[27], 'Pooled myokit.tumour_volume Sigma base')
        self.assertEqual(
            param_names[28], 'ID 0: myokit.tumour_volume Sigma rel.')
        self.assertEqual(
            param_names[29], 'ID 1: myokit.tumour_volume Sigma rel.')
        self.assertEqual(
            param_names[30], 'ID 2: myokit.tumour_volume Sigma rel.')
        self.assertEqual(
            param_names[31], 'Mean log myokit.tumour_volume Sigma rel.')
        self.assertEqual(
            param_names[32], 'Std. log myokit.tumour_volume Sigma rel.')

    def test_set_population_model_bad_input(self):
        # Population models have the wrong type
        pop_models = ['bad', 'type']
        with self.assertRaisesRegex(TypeError, 'The population models'):
            self.pd_problem.set_population_model(pop_models)

        # Number of population models is not correct
        pop_models = [chi.PooledModel()]
        with self.assertRaisesRegex(ValueError, 'The number of population'):
            self.pd_problem.set_population_model(pop_models)

        # Specified parameter names do not coincide with model
        pop_models = [chi.PooledModel()] * 7
        parameter_names = ['wrong names'] * 7
        with self.assertRaisesRegex(ValueError, 'The parameter names'):
            self.pd_problem.set_population_model(pop_models, parameter_names)


class TestInverseProblem(unittest.TestCase):
    """
    Tests the chi.InverseProblem class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        cls.times = [1, 2, 3, 4, 5]
        cls.values = [1, 2, 3, 4, 5]

        # Set up inverse problem
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.model = chi.PharmacodynamicModel(path)
        cls.problem = chi.InverseProblem(cls.model, cls.times, cls.values)

    def test_bad_model_input(self):
        model = 'bad model'

        with self.assertRaisesRegex(ValueError, 'Model has to be an instance'):
            chi.InverseProblem(model, self.times, self.values)

    def test_bad_times_input(self):
        times = [-1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, 'Times cannot be negative.'):
            chi.InverseProblem(self.model, times, self.values)

        times = [5, 4, 3, 2, 1]
        with self.assertRaisesRegex(ValueError, 'Times must be increasing.'):
            chi.InverseProblem(self.model, times, self.values)

    def test_bad_values_input(self):
        values = [1, 2, 3, 4, 5, 6, 7]
        with self.assertRaisesRegex(ValueError, 'Values array must have'):
            chi.InverseProblem(self.model, self.times, values)

        values = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        with self.assertRaisesRegex(ValueError, 'Values array must have'):
            chi.InverseProblem(self.model, self.times, values)

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
