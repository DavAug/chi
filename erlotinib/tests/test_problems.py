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


class TestProblemModellingControllerPDProblem(unittest.TestCase):
    """
    Tests the erlotinib.ProblemModellingController class on a PD modelling
    problem.
    """
    @classmethod
    def setUpClass(cls):
        # Create test dataset
        ids_v = [0, 0, 0, 1, 1, 1, 2, 2]
        times_v = [0, 1, 2, 2, np.nan, 4, 1, 3]
        volumes = [np.nan, 0.3, 0.2, 0.5, 0.1, 0.2, 0.234, np.nan]
        ids_c = [0, 0, 1, 1, 2, 2]
        times_c = [0, 1, 2, np.nan, 1, 3]
        cytokines = [3.4, 0.3, 0.5, np.nan, 0.234, 0]
        ids_d = [0, 1, 1, 1, 2, 2]
        times_d = [0, np.nan, 4, 1, 3, 3]
        dose = [3.4, np.nan, 0.5, 0.5, np.nan, np.nan]
        duration = [0.01, np.nan, 0.31, np.nan, 0.5, np.nan]
        cls.data = pd.DataFrame({
            'ID': ids_v + ids_c + ids_d,
            'Time': times_v + times_c + times_d,
            'Biomarker':
                ['Tumour volume']*8 + ['IL 6']*6 + [np.nan]*6,
            'Measurement': volumes + cytokines + [np.nan]*6,
            'Dose': [np.nan]*14 + dose,
            'Duration': [np.nan]*14 + duration})

        # Test case I: create PD modelling problem
        lib = erlo.ModelLibrary()
        path = lib.tumour_growth_inhibition_model_koch()
        cls.pd_model = erlo.PharmacodynamicModel(path)
        cls.error_model = erlo.ConstantAndMultiplicativeGaussianErrorModel()
        cls.pd_problem = erlo.ProblemModellingController(
            cls.pd_model, cls.error_model)

        # Test case II: create PKPD modelling problem
        lib = erlo.ModelLibrary()
        path = lib.erlotinib_tumour_growth_inhibition_model()
        cls.pkpd_model = erlo.PharmacokineticModel(path)
        cls.pkpd_model.set_outputs([
            'central.drug_concentration',
            'myokit.tumour_volume'])
        cls.error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel(),
            erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        cls.pkpd_problem = erlo.ProblemModellingController(
            cls.pkpd_model, cls.error_models,
            outputs=[
                'central.drug_concentration',
                'myokit.tumour_volume'])

    def test_bad_input(self):
        # Mechanistic model has wrong type
        mechanistic_model = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The mechanistic model'):
            erlo.ProblemModellingController(
                mechanistic_model, self.error_model)

        # Error model has wrong type
        error_model = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'Error models have to be'):
            erlo.ProblemModellingController(
                self.pd_model, error_model)

        error_models = ['wrong', 'type']
        with self.assertRaisesRegex(TypeError, 'Error models have to be'):
            erlo.ProblemModellingController(
                self.pd_model, error_models)

        # Wrong number of error models
        error_model = erlo.ConstantAndMultiplicativeGaussianErrorModel()
        with self.assertRaisesRegex(ValueError, 'Wrong number of error'):
            erlo.ProblemModellingController(
                self.pkpd_model, error_model)

        error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel(),
            erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        with self.assertRaisesRegex(ValueError, 'Wrong number of error'):
            erlo.ProblemModellingController(
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
                erlo.HeterogeneousModel(),
                erlo.PooledModel(),
                erlo.LogNormalModel()])
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
        self.assertEqual(param_names[7], 'Mean Sigma rel.')
        self.assertEqual(param_names[8], 'Std. Sigma rel.')

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
        self.assertEqual(param_names[6], 'Mean Sigma rel.')
        self.assertEqual(param_names[7], 'Std. Sigma rel.')

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
        regimens = self.pd_problem.get_dosing_regimens()
        self.assertIsNone(regimens)

        # Test case II: PKPD problem
        regimens = self.pkpd_problem.get_dosing_regimens()
        self.assertIsNone(regimens)

    def test_get_log_posteriors(self):
        # Test case I: Create posterior with no fixed parameters
        problem = copy.deepcopy(self.pd_problem)
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

        self.assertIsInstance(posterior, erlo.LogPosterior)
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

        self.assertIsInstance(posterior, erlo.LogPosterior)
        self.assertEqual(posterior.n_parameters(), 5)
        self.assertEqual(posterior.get_id(), 'ID 1')

        # Set a population model
        pop_models = [
            erlo.PooledModel(),
            erlo.HeterogeneousModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.LogNormalModel()]
        problem.set_population_model(pop_models)
        problem.set_log_prior([
            pints.HalfCauchyLogPrior(0, 1)]*11)
        posterior = problem.get_log_posterior()

        self.assertIsInstance(posterior, erlo.LogPosterior)
        self.assertEqual(posterior.n_parameters(), 11)

        names = posterior.get_parameter_names()
        ids = posterior.get_id()
        self.assertEqual(len(names), 11)
        self.assertEqual(len(ids), 11)

        self.assertEqual(names[0], 'myokit.tumour_volume')
        self.assertEqual(ids[0], 'Pooled')
        self.assertEqual(names[1], 'myokit.lambda_0')
        self.assertEqual(ids[1], 'ID 0')
        self.assertEqual(names[2], 'myokit.lambda_0')
        self.assertEqual(ids[2], 'ID 1')
        self.assertEqual(names[3], 'myokit.lambda_0')
        self.assertEqual(ids[3], 'ID 2')
        self.assertEqual(names[4], 'myokit.lambda_1')
        self.assertEqual(ids[4], 'Pooled')
        self.assertEqual(names[5], 'Sigma base')
        self.assertEqual(ids[5], 'Pooled')
        self.assertEqual(names[6], 'Sigma rel.')
        self.assertEqual(ids[6], 'ID 0')
        self.assertEqual(names[7], 'Sigma rel.')
        self.assertEqual(ids[7], 'ID 1')
        self.assertEqual(names[8], 'Sigma rel.')
        self.assertEqual(ids[8], 'ID 2')
        self.assertEqual(names[9], 'Sigma rel.')
        self.assertEqual(ids[9], 'Mean')
        self.assertEqual(names[10], 'Sigma rel.')
        self.assertEqual(ids[10], 'Std.')

        # Make sure that selecting an individual is ignored for population
        # models
        posterior = problem.get_log_posterior(individual='some individual')

        self.assertIsInstance(posterior, erlo.LogPosterior)
        self.assertEqual(posterior.n_parameters(), 11)

        names = posterior.get_parameter_names()
        ids = posterior.get_id()
        self.assertEqual(len(names), 11)
        self.assertEqual(len(ids), 11)

        self.assertEqual(names[0], 'myokit.tumour_volume')
        self.assertEqual(ids[0], 'Pooled')
        self.assertEqual(names[1], 'myokit.lambda_0')
        self.assertEqual(ids[1], 'ID 0')
        self.assertEqual(names[2], 'myokit.lambda_0')
        self.assertEqual(ids[2], 'ID 1')
        self.assertEqual(names[3], 'myokit.lambda_0')
        self.assertEqual(ids[3], 'ID 2')
        self.assertEqual(names[4], 'myokit.lambda_1')
        self.assertEqual(ids[4], 'Pooled')
        self.assertEqual(names[5], 'Sigma base')
        self.assertEqual(ids[5], 'Pooled')
        self.assertEqual(names[6], 'Sigma rel.')
        self.assertEqual(ids[6], 'ID 0')
        self.assertEqual(names[7], 'Sigma rel.')
        self.assertEqual(ids[7], 'ID 1')
        self.assertEqual(names[8], 'Sigma rel.')
        self.assertEqual(ids[8], 'ID 2')
        self.assertEqual(names[9], 'Sigma rel.')
        self.assertEqual(ids[9], 'Mean')
        self.assertEqual(names[10], 'Sigma rel.')
        self.assertEqual(ids[10], 'Std.')

    def test_get_log_posteriors_bad_input(self):
        problem = copy.deepcopy(self.pd_problem)

        # No data has been set
        with self.assertRaisesRegex(ValueError, 'The data has not'):
            problem.get_log_posterior()

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

    # def test_get_n_parameters(self):
    #     # Test whether exclude pop models work
    #     self.problem.set_mechanistic_model(self.model)
    #     self.problem.set_error_model(self.error_models)
    #     pop_models = [
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.HeterogeneousModel(),
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.LogNormalModel()]
    #     self.problem.set_population_model(pop_models)

    #     self.assertEqual(self.problem.get_n_parameters(), 13)
    #     self.assertEqual(
    #         self.problem.get_n_parameters(exclude_pop_model=True),
    #         7)

    # def test_get_parameter_names(self):
    #     # Test with a mechanistic-error model pair only
    #     self.problem.set_mechanistic_model(self.model)
    #     self.problem.set_error_model(self.error_models)

    #     param_names = self.problem.get_parameter_names()
    #     self.assertEqual(len(param_names), 7)
    #     self.assertEqual(param_names[0], 'myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'myokit.drug_concentration')
    #     self.assertEqual(param_names[2], 'myokit.kappa')
    #     self.assertEqual(param_names[3], 'myokit.lambda_0')
    #     self.assertEqual(param_names[4], 'myokit.lambda_1')
    #     self.assertEqual(param_names[5], 'Sigma base')
    #     self.assertEqual(param_names[6], 'Sigma rel.')

    #     # Check that also works with exclude pop params flag
    #     param_names = self.problem.get_parameter_names(exclude_pop_model=True)
    #     self.assertEqual(len(param_names), 7)
    #     self.assertEqual(param_names[0], 'myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'myokit.drug_concentration')
    #     self.assertEqual(param_names[2], 'myokit.kappa')
    #     self.assertEqual(param_names[3], 'myokit.lambda_0')
    #     self.assertEqual(param_names[4], 'myokit.lambda_1')
    #     self.assertEqual(param_names[5], 'Sigma base')
    #     self.assertEqual(param_names[6], 'Sigma rel.')

    #     # Test with fixed parameters
    #     name_value_dict = dict({
    #         'myokit.drug_concentration': 0,
    #         'myokit.kappa': 1})
    #     self.problem.fix_parameters(name_value_dict)

    #     param_names = self.problem.get_parameter_names()
    #     self.assertEqual(len(param_names), 5)
    #     self.assertEqual(param_names[0], 'myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'myokit.lambda_0')
    #     self.assertEqual(param_names[2], 'myokit.lambda_1')
    #     self.assertEqual(param_names[3], 'Sigma base')
    #     self.assertEqual(param_names[4], 'Sigma rel.')

    #     # Test with setting a population model
    #     self.problem.set_mechanistic_model(self.model)
    #     self.problem.set_error_model(self.error_models)
    #     pop_models = [
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.HeterogeneousModel(),
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.LogNormalModel()]
    #     self.problem.set_population_model(pop_models)

    #     param_names = self.problem.get_parameter_names()
    #     self.assertEqual(len(param_names), 13)
    #     self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
    #     self.assertEqual(param_names[2], 'ID 0: myokit.kappa')
    #     self.assertEqual(param_names[3], 'ID 1: myokit.kappa')
    #     self.assertEqual(param_names[4], 'ID 2: myokit.kappa')
    #     self.assertEqual(param_names[5], 'Pooled myokit.lambda_0')
    #     self.assertEqual(param_names[6], 'Pooled myokit.lambda_1')
    #     self.assertEqual(param_names[7], 'Pooled Sigma base')
    #     self.assertEqual(param_names[8], 'ID 0: Sigma rel.')
    #     self.assertEqual(param_names[9], 'ID 1: Sigma rel.')
    #     self.assertEqual(param_names[10], 'ID 2: Sigma rel.')
    #     self.assertEqual(param_names[11], 'Mean Sigma rel.')
    #     self.assertEqual(param_names[12], 'Std. Sigma rel.')

    #     # Test whether exclude population model works
    #     param_names = self.problem.get_parameter_names(exclude_pop_model=True)
    #     self.assertEqual(len(param_names), 7)
    #     self.assertEqual(param_names[0], 'myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'myokit.drug_concentration')
    #     self.assertEqual(param_names[2], 'myokit.kappa')
    #     self.assertEqual(param_names[3], 'myokit.lambda_0')
    #     self.assertEqual(param_names[4], 'myokit.lambda_1')
    #     self.assertEqual(param_names[5], 'Sigma base')
    #     self.assertEqual(param_names[6], 'Sigma rel.')

    # def test_set_error_model(self):
    #     # Map error model to output automatically
    #     self.problem.set_mechanistic_model(self.model)
    #     self.problem.set_error_model(self.error_models)

    #     self.assertEqual(self.problem.get_n_parameters(), 7)
    #     param_names = self.problem.get_parameter_names()
    #     self.assertEqual(param_names[0], 'myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'myokit.drug_concentration')
    #     self.assertEqual(param_names[2], 'myokit.kappa')
    #     self.assertEqual(param_names[3], 'myokit.lambda_0')
    #     self.assertEqual(param_names[4], 'myokit.lambda_1')
    #     self.assertEqual(param_names[5], 'Sigma base')
    #     self.assertEqual(param_names[6], 'Sigma rel.')

    #     # Set error model-output mapping explicitly
    #     problem = erlo.ProblemModellingController(
    #         self.data, biom_keys=['Biomarker 1', 'Biomarker 2'])
    #     path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
    #     model = erlo.PharmacodynamicModel(path)
    #     output_biomarker_map = dict({
    #         'myokit.tumour_volume': 'Biomarker 2',
    #         'myokit.drug_concentration': 'Biomarker 1'})
    #     problem.set_mechanistic_model(model, output_biomarker_map)
    #     log_likelihoods = [
    #         erlo.ConstantAndMultiplicativeGaussianErrorModel(),
    #         erlo.ConstantAndMultiplicativeGaussianErrorModel()]
    #     outputs = ['myokit.tumour_volume', 'myokit.drug_concentration']
    #     problem.set_error_model(log_likelihoods, outputs)

    #     self.assertEqual(problem.get_n_parameters(), 9)
    #     param_names = problem.get_parameter_names()
    #     self.assertEqual(param_names[0], 'myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'myokit.drug_concentration')
    #     self.assertEqual(param_names[2], 'myokit.kappa')
    #     self.assertEqual(param_names[3], 'myokit.lambda_0')
    #     self.assertEqual(param_names[4], 'myokit.lambda_1')
    #     self.assertEqual(param_names[5], 'myokit.tumour_volume Sigma base')
    #     self.assertEqual(param_names[6], 'myokit.tumour_volume Sigma rel.')
    #     self.assertEqual(
    #         param_names[7], 'myokit.drug_concentration Sigma base')
    #     self.assertEqual(
    #         param_names[8], 'myokit.drug_concentration Sigma rel.')

    # def test_set_error_model_bad_input(self):
    #     # No mechanistic model set
    #     problem = erlo.ProblemModellingController(
    #         self.data, biom_keys=['Biomarker'])

    #     with self.assertRaisesRegex(ValueError, 'Before setting'):
    #         problem.set_error_model(self.error_models)

    #     # Error models have the wrong type
    #     path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
    #     model = erlo.PharmacodynamicModel(path)
    #     problem.set_mechanistic_model(model)

    #     error_models = [str, float, int]
    #     with self.assertRaisesRegex(ValueError, 'The error models have'):
    #         problem.set_error_model(error_models)

    #     # Number of error_models does not match the number of outputs
    #     error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2
    #     with self.assertRaisesRegex(ValueError, 'The number of error'):
    #         problem.set_error_model(error_models)

    #     # The specified outputs do not match the model outputs
    #     outputs = ['wrong', 'outputs']
    #     with self.assertRaisesRegex(ValueError, 'The specified outputs'):
    #         problem.set_error_model(self.error_models, outputs)

    # def test_set_log_prior(self):
    #     # Map priors to parameters automatically
    #     self.problem.set_mechanistic_model(self.model)
    #     self.problem.set_error_model(self.error_models)
    #     self.problem.set_log_prior(self.log_priors)

    #     # Specify prior parameter map explicitly
    #     parameters = [
    #         'myokit.kappa',
    #         'Sigma base',
    #         'Sigma rel.',
    #         'myokit.tumour_volume',
    #         'myokit.lambda_1',
    #         'myokit.drug_concentration',
    #         'myokit.lambda_0']
    #     self.problem.set_log_prior(self.log_priors, parameters)

    # def test_set_log_prior_bad_input(self):
    #     # No mechanistic model set
    #     problem = erlo.ProblemModellingController(
    #         self.data, biom_keys=['Biomarker'])

    #     with self.assertRaisesRegex(ValueError, 'Before setting'):
    #         problem.set_log_prior(self.log_priors)

    #     # No error model set
    #     path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
    #     model = erlo.PharmacodynamicModel(path)
    #     problem.set_mechanistic_model(model)

    #     with self.assertRaisesRegex(ValueError, 'Before setting'):
    #         problem.set_log_prior(self.log_priors)

    #     # Wrong log-prior type
    #     problem.set_error_model(self.error_models)
    #     priors = ['Wrong', 'type']
    #     with self.assertRaisesRegex(ValueError, 'All marginal log-priors'):
    #         problem.set_log_prior(priors)

    #     # Number of log priors does not match number of parameters
    #     priors = [pints.GaussianLogPrior(0, 1), pints.HalfCauchyLogPrior(0, 1)]
    #     with self.assertRaisesRegex(ValueError, 'One marginal log-prior'):
    #         problem.set_log_prior(priors)

    #     # Dimensionality of joint log-pior does not match number of params
    #     prior = pints.ComposedLogPrior(
    #         pints.GaussianLogPrior(0, 1), pints.GaussianLogPrior(0, 1))
    #     priors = [
    #         prior,
    #         pints.UniformLogPrior(0, 1),
    #         pints.UniformLogPrior(0, 1),
    #         pints.UniformLogPrior(0, 1),
    #         pints.UniformLogPrior(0, 1),
    #         pints.UniformLogPrior(0, 1),
    #         pints.UniformLogPrior(0, 1)]
    #     with self.assertRaisesRegex(ValueError, 'The joint log-prior'):
    #         problem.set_log_prior(priors)

    #     # Specified parameter names do not match the model parameters
    #     params = ['wrong', 'params']
    #     with self.assertRaisesRegex(ValueError, 'The specified parameter'):
    #         problem.set_log_prior(self.log_priors, params)

    # def test_set_mechanistic_model(self):
    #     # Set output biomarker mapping automatically
    #     problem = erlo.ProblemModellingController(
    #         self.data, biom_keys=['Biomarker 1'])

    #     path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
    #     model = erlo.PharmacodynamicModel(path)
    #     problem.set_mechanistic_model(model=model)

    #     self.assertEqual(problem._mechanistic_model, model)
    #     outputs = problem._mechanistic_model.outputs()
    #     self.assertEqual(len(outputs), 1)
    #     self.assertEqual(outputs[0], 'myokit.tumour_volume')
    #     self.assertEqual(len(problem._biom_keys), 1)
    #     self.assertEqual(problem._biom_keys[0], 'Biomarker 1')

    #     problem = erlo.ProblemModellingController(
    #         self.data, biom_keys=['Biomarker 1', 'Biomarker 2'])

    #     # Set output biomarker mapping explicitly
    #     output_biomarker_map = dict({
    #         'myokit.tumour_volume': 'Biomarker 2',
    #         'myokit.drug_concentration': 'Biomarker 1'})

    #     problem.set_mechanistic_model(model, output_biomarker_map)

    #     self.assertEqual(problem._mechanistic_model, model)
    #     outputs = problem._mechanistic_model.outputs()
    #     self.assertEqual(len(outputs), 2)
    #     self.assertEqual(outputs[0], 'myokit.tumour_volume')
    #     self.assertEqual(outputs[1], 'myokit.drug_concentration')
    #     self.assertEqual(len(problem._biom_keys), 2)
    #     self.assertEqual(problem._biom_keys[0], 'Biomarker 2')
    #     self.assertEqual(problem._biom_keys[1], 'Biomarker 1')

    # def test_set_mechanistic_model_bad_input(self):
    #     # Wrong model type
    #     model = 'some model'

    #     with self.assertRaisesRegex(ValueError, 'The model has to be'):
    #         self.problem.set_mechanistic_model(model=model)

    #     # Wrong number of model outputs
    #     problem = erlo.ProblemModellingController(
    #         self.data, biom_keys=['Biomarker 1', 'Biomarker 2'])

    #     with self.assertRaisesRegex(ValueError, 'The model does not have'):
    #         problem.set_mechanistic_model(model=self.model)

    #     # Wrong map type
    #     output_biomarker_map = 'bad map'

    #     with self.assertRaisesRegex(ValueError, 'The output-biomarker'):
    #         self.problem.set_mechanistic_model(
    #             self.model, output_biomarker_map)

    #     # Map does not contain biomarkers specfied by the dataset
    #     output_biomarker_map = dict({'Some variable': 'Some biomarker'})

    #     with self.assertRaisesRegex(ValueError, 'The provided output'):
    #         self.problem.set_mechanistic_model(
    #             self.model, output_biomarker_map)

    # def test_set_population_model(self):
    #     # Map population model to parameters automatically
    #     self.problem.set_mechanistic_model(self.model)
    #     self.problem.set_error_model(self.error_models)
    #     pop_models = [
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.HeterogeneousModel(),
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.PooledModel(),
    #         erlo.LogNormalModel()]
    #     self.problem.set_population_model(pop_models)

    #     self.assertEqual(self.problem.get_n_parameters(), 13)
    #     param_names = self.problem.get_parameter_names()
    #     self.assertEqual(len(param_names), 13)
    #     self.assertEqual(param_names[0], 'Pooled myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'Pooled myokit.drug_concentration')
    #     self.assertEqual(param_names[2], 'ID 0: myokit.kappa')
    #     self.assertEqual(param_names[3], 'ID 1: myokit.kappa')
    #     self.assertEqual(param_names[4], 'ID 2: myokit.kappa')
    #     self.assertEqual(param_names[5], 'Pooled myokit.lambda_0')
    #     self.assertEqual(param_names[6], 'Pooled myokit.lambda_1')
    #     self.assertEqual(param_names[7], 'Pooled Sigma base')
    #     self.assertEqual(param_names[8], 'ID 0: Sigma rel.')
    #     self.assertEqual(param_names[9], 'ID 1: Sigma rel.')
    #     self.assertEqual(param_names[10], 'ID 2: Sigma rel.')
    #     self.assertEqual(param_names[11], 'Mean Sigma rel.')
    #     self.assertEqual(param_names[12], 'Std. Sigma rel.')

    #     # Map population model to parameters explicitly (with blanks)
    #     pop_models = [erlo.PooledModel()] * 5  # 6 paramaters in total
    #     params = [
    #         'myokit.drug_concentration',
    #         'myokit.kappa',
    #         'myokit.lambda_0',
    #         'myokit.lambda_1',
    #         'Sigma rel.']
    #     self.problem.set_population_model(pop_models, params)

    #     self.assertEqual(self.problem.get_n_parameters(), 11)
    #     param_names = self.problem.get_parameter_names()
    #     self.assertEqual(param_names[0], 'ID 0: myokit.tumour_volume')
    #     self.assertEqual(param_names[1], 'ID 1: myokit.tumour_volume')
    #     self.assertEqual(param_names[2], 'ID 2: myokit.tumour_volume')
    #     self.assertEqual(param_names[3], 'Pooled myokit.drug_concentration')
    #     self.assertEqual(param_names[4], 'Pooled myokit.kappa')
    #     self.assertEqual(param_names[5], 'Pooled myokit.lambda_0')
    #     self.assertEqual(param_names[6], 'Pooled myokit.lambda_1')
    #     self.assertEqual(param_names[7], 'ID 0: Sigma base')
    #     self.assertEqual(param_names[8], 'ID 1: Sigma base')
    #     self.assertEqual(param_names[9], 'ID 2: Sigma base')
    #     self.assertEqual(param_names[10], 'Pooled Sigma rel.')

    # def test_set_population_model_bad_input(self):
    #     # No mechanistic model set
    #     problem = erlo.ProblemModellingController(
    #         self.data, biom_keys=['Biomarker'])

    #     with self.assertRaisesRegex(ValueError, 'Before setting'):
    #         problem.set_population_model(self.pop_models)

    #     # No error model set
    #     problem.set_mechanistic_model(self.model)

    #     with self.assertRaisesRegex(ValueError, 'Before setting'):
    #         problem.set_population_model(self.pop_models)

    #     # Population models have the wrong type
    #     problem.set_error_model(self.error_models)

    #     pop_models = ['bad', 'type']
    #     with self.assertRaisesRegex(ValueError, 'The population models'):
    #         problem.set_population_model(pop_models)

    #     # The specified parameters do not match the model parameters
    #     pop_models = [erlo.PooledModel(), erlo.HeterogeneousModel()]
    #     params = ['wrong', 'outputs']
    #     with self.assertRaisesRegex(ValueError, 'The parameter <wrong>'):
    #         problem.set_population_model(pop_models, params)

    #     # Not one population model for each parameter
    #     with self.assertRaisesRegex(ValueError, 'If no parameter names are'):
    #         problem.set_population_model(pop_models)


# class TestProblemModellingControllerPKProblem(unittest.TestCase):
#     """
#     Tests the erlotinib.ProblemModellingController class on a PK modelling
#     problem.
#     """
#     @classmethod
#     def setUpClass(cls):
#         # Create test dataset
#         ids = [0, 0, 0, 1, 1, 1, 2, 2]
#         times = [0, 1, 2, 2, np.nan, 4, 1, 3]
#         plasma_conc = [np.nan, 0.3, 0.2, 0.5, 0.1, 0.2, 0.234, np.nan]
#         dose = [3.4, np.nan, np.nan, 0.5, 0.5, 0.5, np.nan, np.nan]
#         duration = [
#             0.01, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
#         cls.data = pd.DataFrame({
#             'ID': ids,
#             'Time': times,
#             'Biomarker': plasma_conc,
#             'Biomarker 2': plasma_conc,
#             'Dose': dose,
#             'Duration': duration})

#         # Create test problem
#         cls.problem = erlo.ProblemModellingController(
#             cls.data, dose_key='Dose', dose_duration_key='Duration')

#         # Create test model
#         path = erlo.ModelLibrary().one_compartment_pk_model()
#         cls.model = erlo.PharmacokineticModel(path)
#         cls.model.set_administration(compartment='central', direct=False)
#         cls.error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]
#         cls.log_priors = [
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1)]

#     def test_bad_input(self):
#         # Create data of wrong type
#         data = np.ones(shape=(10, 4))

#         with self.assertRaisesRegex(ValueError, 'Data has to be'):
#             erlo.ProblemModellingController(data=data)

#         # Wrong ID key
#         data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

#         with self.assertRaisesRegex(ValueError, 'Data does not have'):
#             erlo.ProblemModellingController(data=data)

#         # Wrong time key
#         data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

#         with self.assertRaisesRegex(ValueError, 'Data does not have'):
#             erlo.ProblemModellingController(data=data)

#         # Wrong Biomarker key
#         data = self.data.rename(columns={'Biomarker': 'SOME NON-STANDARD KEY'})

#         with self.assertRaisesRegex(ValueError, 'Data does not have'):
#             erlo.ProblemModellingController(data=data)

#         # Wrong dose key
#         with self.assertRaisesRegex(ValueError, 'Data does not have'):
#             erlo.ProblemModellingController(data=self.data, dose_key='New key')

#     def test_data_keys(self):
#         # Rename ID key
#         data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

#         # Test that it works with correct mapping
#         erlo.ProblemModellingController(
#             data=data, id_key='SOME NON-STANDARD KEY')

#         # Test that it fails with wrong mapping
#         with self.assertRaisesRegex(
#                 ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
#             erlo.ProblemModellingController(
#                 data=data, id_key='SOME WRONG KEY')

#         # Rename time key
#         data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

#         # Test that it works with correct mapping
#         erlo.ProblemModellingController(
#             data=data, time_key='SOME NON-STANDARD KEY')

#         # Test that it fails with wrong mapping
#         with self.assertRaisesRegex(
#                 ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
#             erlo.ProblemModellingController(
#                 data=data, time_key='SOME WRONG KEY')

#         # Rename biomarker key
#         data = self.data.rename(columns={'Biomarker': 'SOME NON-STANDARD KEY'})

#         # Test that it works with correct mapping
#         erlo.ProblemModellingController(
#             data=data, biom_keys=['SOME NON-STANDARD KEY'])

#         # Test that it fails with wrong mapping
#         with self.assertRaisesRegex(
#                 ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
#             erlo.ProblemModellingController(
#                 data=data, biom_keys=['SOME WRONG KEY'])

#         # Rename biomarker key
#         data = self.data.rename(columns={'Dose': 'SOME NON-STANDARD KEY'})

#         # Test that it works with correct mapping
#         erlo.ProblemModellingController(
#             data=data, dose_key='SOME NON-STANDARD KEY')

#         # Test that it fails with wrong mapping
#         with self.assertRaisesRegex(
#                 ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
#             erlo.ProblemModellingController(
#                 data=data, dose_key='SOME WRONG KEY')

#         # Test that it works with no dose key
#         erlo.ProblemModellingController(data=data)

#     def test_fix_parameters(self):
#         # Fix model parameters
#         self.problem.set_mechanistic_model(self.model)
#         self.problem.set_error_model(self.error_models)
#         name_value_dict = dict({
#             'central.drug_amount': 0,
#             'myokit.elimination_rate': 1})

#         self.problem.fix_parameters(name_value_dict)

#         self.assertEqual(self.problem.get_n_parameters(), 5)
#         param_names = self.problem.get_parameter_names()
#         self.assertEqual(len(param_names), 5)
#         self.assertEqual(param_names[0], 'dose.drug_amount')
#         self.assertEqual(param_names[1], 'central.size')
#         self.assertEqual(param_names[2], 'dose.absorption_rate')
#         self.assertEqual(param_names[3], 'Sigma base')
#         self.assertEqual(param_names[4], 'Sigma rel.')

#         # Free elimination rate and fix dose drug amount
#         name_value_dict = dict({
#             'dose.drug_amount': 0,
#             'myokit.elimination_rate': None})

#         self.problem.fix_parameters(name_value_dict)

#         self.assertEqual(self.problem.get_n_parameters(), 5)
#         param_names = self.problem.get_parameter_names()
#         self.assertEqual(len(param_names), 5)
#         self.assertEqual(param_names[0], 'central.size')
#         self.assertEqual(param_names[1], 'dose.absorption_rate')
#         self.assertEqual(param_names[2], 'myokit.elimination_rate')
#         self.assertEqual(param_names[3], 'Sigma base')
#         self.assertEqual(param_names[4], 'Sigma rel.')

#         # Free all parameters again
#         name_value_dict = dict({
#             'dose.drug_amount': None,
#             'central.drug_amount': None})

#         self.problem.fix_parameters(name_value_dict)

#         self.assertEqual(self.problem.get_n_parameters(), 7)
#         param_names = self.problem.get_parameter_names()
#         self.assertEqual(len(param_names), 7)
#         self.assertEqual(param_names[0], 'central.drug_amount')
#         self.assertEqual(param_names[1], 'dose.drug_amount')
#         self.assertEqual(param_names[2], 'central.size')
#         self.assertEqual(param_names[3], 'dose.absorption_rate')
#         self.assertEqual(param_names[4], 'myokit.elimination_rate')
#         self.assertEqual(param_names[5], 'Sigma base')
#         self.assertEqual(param_names[6], 'Sigma rel.')

#     def test_fix_parameters_bad_input(self):
#         name_value_dict = dict({
#             'dose.drug_amount': 0,
#             'myokit.elimination_rate': None})

#         # No mechanistic model set
#         problem = erlo.ProblemModellingController(
#             self.data, biom_keys=['Biomarker'])

#         with self.assertRaisesRegex(ValueError, 'The mechanistic'):
#             problem.fix_parameters(name_value_dict)

#         # No error model set
#         path = erlo.ModelLibrary().one_compartment_pk_model()
#         model = erlo.PharmacokineticModel(path)
#         problem.set_mechanistic_model(model)

#         with self.assertRaisesRegex(ValueError, 'The error model'):
#             problem.fix_parameters(name_value_dict)

#     def test_get_dosing_regimens(self):
#         regimens = self.problem.get_dosing_regimens()

#         n_ids = 3
#         ids = list(regimens.keys())
#         self.assertEqual(len(ids), n_ids)
#         self.assertEqual(ids[0], '0')
#         self.assertEqual(ids[1], '1')
#         self.assertEqual(ids[2], '2')

#         # Check protocols
#         events = regimens['0'].events()
#         self.assertEqual(len(events), 1)

#         event = events[0]
#         dose = 3.4
#         duration = 0.01
#         self.assertEqual(event.level(), dose / duration)
#         self.assertEqual(event.start(), 0)
#         self.assertEqual(event.duration(), duration)
#         self.assertEqual(event.period(), 0)
#         self.assertEqual(event.multiplier(), 0)

#         events = regimens['1'].events()
#         self.assertEqual(len(events), 2)

#         event = events[0]
#         dose = 0.5
#         duration = 0.01
#         self.assertEqual(event.level(), dose / duration)
#         self.assertEqual(event.start(), 2)
#         self.assertEqual(event.duration(), duration)
#         self.assertEqual(event.period(), 0)
#         self.assertEqual(event.multiplier(), 0)

#         event = events[1]
#         dose = 0.5
#         duration = 0.01
#         self.assertEqual(event.level(), dose / duration)
#         self.assertEqual(event.start(), 4)
#         self.assertEqual(event.duration(), duration)
#         self.assertEqual(event.period(), 0)
#         self.assertEqual(event.multiplier(), 0)

#         events = regimens['2'].events()
#         self.assertEqual(len(events), 0)

#     def test_get_log_posteriors(self):
#         # Create posterior with no fixed parameters
#         self.problem.set_mechanistic_model(self.model)
#         self.problem.set_error_model(self.error_models)
#         self.problem.set_log_prior(self.log_priors)
#         posteriors = self.problem.get_log_posteriors()

#         self.assertEqual(len(posteriors), 3)
#         self.assertEqual(posteriors[0].n_parameters(), 7)
#         self.assertEqual(posteriors[0].get_id(), 'ID 0')
#         self.assertEqual(posteriors[1].n_parameters(), 7)
#         self.assertEqual(posteriors[1].get_id(), 'ID 1')
#         self.assertEqual(posteriors[2].n_parameters(), 7)
#         self.assertEqual(posteriors[2].get_id(), 'ID 2')

#         # Fix some parameters
#         name_value_dict = dict({
#             'central.drug_amount': 0,
#             'dose.absorption_rate': 1})
#         self.problem.fix_parameters(name_value_dict)
#         self.problem.set_log_prior(self.log_priors[:-2])
#         posteriors = self.problem.get_log_posteriors()

#         self.assertEqual(len(posteriors), 3)
#         self.assertEqual(posteriors[0].n_parameters(), 5)
#         self.assertEqual(posteriors[0].get_id(), 'ID 0')
#         self.assertEqual(posteriors[1].n_parameters(), 5)
#         self.assertEqual(posteriors[1].get_id(), 'ID 1')
#         self.assertEqual(posteriors[2].n_parameters(), 5)
#         self.assertEqual(posteriors[2].get_id(), 'ID 2')

#     def test_get_log_posteriors_bad_input(self):
#         # No mechanistic model set
#         problem = erlo.ProblemModellingController(
#             self.data, biom_keys=['Biomarker'])

#         with self.assertRaisesRegex(ValueError, 'The mechanistic'):
#             problem.get_log_posteriors()

#         # No error model set
#         path = erlo.ModelLibrary().one_compartment_pk_model()
#         model = erlo.PharmacokineticModel(path)
#         problem.set_mechanistic_model(model)

#         with self.assertRaisesRegex(ValueError, 'The error model'):
#             problem.get_log_posteriors()

#         # No log-prior set
#         problem.set_error_model(self.error_models)

#         with self.assertRaisesRegex(ValueError, 'The log-prior'):
#             problem.get_log_posteriors()

#     def test_set_error_model(self):
#         # Map error model to output automatically
#         self.problem.set_mechanistic_model(self.model)
#         self.problem.set_error_model(self.error_models)

#         self.assertEqual(self.problem.get_n_parameters(), 7)
#         param_names = self.problem.get_parameter_names()
#         self.assertEqual(param_names[0], 'central.drug_amount')
#         self.assertEqual(param_names[1], 'dose.drug_amount')
#         self.assertEqual(param_names[2], 'central.size')
#         self.assertEqual(param_names[3], 'dose.absorption_rate')
#         self.assertEqual(param_names[4], 'myokit.elimination_rate')
#         self.assertEqual(param_names[5], 'Sigma base')
#         self.assertEqual(param_names[6], 'Sigma rel.')

#         # Set error model-output mapping explicitly
#         problem = erlo.ProblemModellingController(
#             self.data, biom_keys=['Biomarker', 'Biomarker 2'])
#         path = erlo.ModelLibrary().one_compartment_pk_model()
#         model = erlo.PharmacokineticModel(path)
#         model.set_administration('central', direct=False)
#         output_biomarker_map = dict({
#             'dose.drug_amount': 'Biomarker 2',
#             'central.drug_concentration': 'Biomarker'})
#         problem.set_mechanistic_model(model, output_biomarker_map)
#         log_likelihoods = [
#             erlo.ConstantAndMultiplicativeGaussianErrorModel(),
#             erlo.ConstantAndMultiplicativeGaussianErrorModel()]
#         outputs = ['dose.drug_amount', 'central.drug_concentration']
#         problem.set_error_model(log_likelihoods, outputs)

#         self.assertEqual(problem.get_n_parameters(), 9)
#         param_names = problem.get_parameter_names()
#         self.assertEqual(param_names[0], 'central.drug_amount')
#         self.assertEqual(param_names[1], 'dose.drug_amount')
#         self.assertEqual(param_names[2], 'central.size')
#         self.assertEqual(param_names[3], 'dose.absorption_rate')
#         self.assertEqual(param_names[4], 'myokit.elimination_rate')
#         self.assertEqual(param_names[5], 'dose.drug_amount Sigma base')
#         self.assertEqual(param_names[6], 'dose.drug_amount Sigma rel.')
#         self.assertEqual(
#             param_names[7], 'central.drug_concentration Sigma base')
#         self.assertEqual(
#             param_names[8], 'central.drug_concentration Sigma rel.')

#     def test_set_error_model_bad_input(self):
#         # No mechanistic model set
#         problem = erlo.ProblemModellingController(
#             self.data, biom_keys=['Biomarker'])

#         with self.assertRaisesRegex(ValueError, 'Before setting'):
#             problem.set_error_model(self.error_models)

#         # Error models have the wrong type
#         path = erlo.ModelLibrary().one_compartment_pk_model()
#         model = erlo.PharmacokineticModel(path)
#         problem.set_mechanistic_model(model)

#         error_models = [str, float, int]
#         with self.assertRaisesRegex(ValueError, 'The error models have'):
#             problem.set_error_model(error_models)

#         # Number of error_models does not match the number of outputs
#         error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2
#         with self.assertRaisesRegex(ValueError, 'The number of error'):
#             problem.set_error_model(error_models)

#         # The specified outputs do not match the model outputs
#         outputs = ['wrong', 'outputs']
#         with self.assertRaisesRegex(ValueError, 'The specified outputs'):
#             problem.set_error_model(self.error_models, outputs)

#     def test_set_log_prior(self):
#         # Map priors to parameters automatically
#         self.problem.set_mechanistic_model(self.model)
#         self.problem.set_error_model(self.error_models)
#         self.problem.set_log_prior(self.log_priors)

#         # Specify prior parameter map explicitly
#         parameters = [
#             'central.size',
#             'Sigma base',
#             'Sigma rel.',
#             'myokit.elimination_rate',
#             'central.drug_amount',
#             'dose.absorption_rate',
#             'dose.drug_amount']
#         self.problem.set_log_prior(self.log_priors, parameters)

#     def test_set_log_prior_bad_input(self):
#         # No mechanistic model set
#         problem = erlo.ProblemModellingController(
#             self.data, biom_keys=['Biomarker'])

#         with self.assertRaisesRegex(ValueError, 'Before setting'):
#             problem.set_log_prior(self.log_priors)

#         # No error model set
#         path = erlo.ModelLibrary().one_compartment_pk_model()
#         model = erlo.PharmacokineticModel(path)
#         model.set_administration('central', direct=False)
#         problem.set_mechanistic_model(model)

#         with self.assertRaisesRegex(ValueError, 'Before setting'):
#             problem.set_log_prior(self.log_priors)

#         # Wrong log-prior type
#         problem.set_error_model(self.error_models)
#         priors = ['Wrong', 'type']
#         with self.assertRaisesRegex(ValueError, 'All marginal log-priors'):
#             problem.set_log_prior(priors)

#         # Number of log priors does not match number of parameters
#         priors = [pints.GaussianLogPrior(0, 1), pints.HalfCauchyLogPrior(0, 1)]
#         with self.assertRaisesRegex(ValueError, 'One marginal log-prior'):
#             problem.set_log_prior(priors)

#         # Dimensionality of joint log-pior does not match number of params
#         prior = pints.ComposedLogPrior(
#             pints.GaussianLogPrior(0, 1), pints.GaussianLogPrior(0, 1))
#         priors = [
#             prior,
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1),
#             pints.UniformLogPrior(0, 1)]
#         with self.assertRaisesRegex(ValueError, 'The joint log-prior'):
#             problem.set_log_prior(priors)

#         # Specified parameter names do not match the model parameters
#         params = ['wrong', 'params']
#         with self.assertRaisesRegex(ValueError, 'The specified parameter'):
#             problem.set_log_prior(self.log_priors, params)

#     def test_set_mechanistic_model(self):
#         # Set output biomarker mapping automatically
#         problem = erlo.ProblemModellingController(
#             self.data, dose_key='Dose')

#         path = erlo.ModelLibrary().one_compartment_pk_model()
#         model = erlo.PharmacokineticModel(path)
#         model.set_administration(compartment='central', direct=False)
#         problem.set_mechanistic_model(model)

#         self.assertEqual(problem._mechanistic_model, model)
#         outputs = problem._mechanistic_model.outputs()
#         self.assertEqual(len(outputs), 1)
#         self.assertEqual(outputs[0], 'central.drug_concentration')
#         self.assertEqual(len(problem._biom_keys), 1)
#         self.assertEqual(problem._biom_keys[0], 'Biomarker')

#         # Set output biomarker mapping explicitly
#         problem = erlo.ProblemModellingController(
#             self.data, biom_keys=['Biomarker', 'Biomarker 2'], dose_key='Dose')
#         output_biomarker_map = dict({
#             'dose.drug_amount': 'Biomarker 2',
#             'central.drug_concentration': 'Biomarker'})

#         problem.set_mechanistic_model(model, output_biomarker_map)

#         self.assertEqual(problem._mechanistic_model, model)
#         outputs = problem._mechanistic_model.outputs()
#         self.assertEqual(len(outputs), 2)
#         self.assertEqual(outputs[0], 'dose.drug_amount')
#         self.assertEqual(outputs[1], 'central.drug_concentration')
#         self.assertEqual(len(problem._biom_keys), 2)
#         self.assertEqual(problem._biom_keys[0], 'Biomarker 2')
#         self.assertEqual(problem._biom_keys[1], 'Biomarker')

#     def test_set_mechanistic_model_bad_input(self):
#         # Wrong model type
#         model = 'some model'

#         with self.assertRaisesRegex(ValueError, 'The model has to be'):
#             self.problem.set_mechanistic_model(model=model)

#         # Wrong number of model outputs
#         problem = erlo.ProblemModellingController(
#             self.data, biom_keys=['Biomarker', 'Biomarker 2'])

#         with self.assertRaisesRegex(ValueError, 'The model does not have'):
#             problem.set_mechanistic_model(model=self.model)

#         # Wrong map type
#         output_biomarker_map = 'bad map'

#         with self.assertRaisesRegex(ValueError, 'The output-biomarker'):
#             self.problem.set_mechanistic_model(
#                 self.model, output_biomarker_map)

#         # Map does not contain biomarkers specfied by the dataset
#         output_biomarker_map = dict({'Some variable': 'Some biomarker'})

#         with self.assertRaisesRegex(ValueError, 'The provided output'):
#             self.problem.set_mechanistic_model(
#                 self.model, output_biomarker_map)


# class TestInverseProblem(unittest.TestCase):
#     """
#     Tests the erlotinib.InverseProblem class.
#     """

#     @classmethod
#     def setUpClass(cls):
#         # Create test data
#         cls.times = [1, 2, 3, 4, 5]
#         cls.values = [1, 2, 3, 4, 5]

#         # Set up inverse problem
#         path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
#         cls.model = erlo.PharmacodynamicModel(path)
#         cls.problem = erlo.InverseProblem(cls.model, cls.times, cls.values)

#     def test_bad_model_input(self):
#         model = 'bad model'

#         with self.assertRaisesRegex(ValueError, 'Model has to be an instance'):
#             erlo.InverseProblem(model, self.times, self.values)

#     def test_bad_times_input(self):
#         times = [-1, 2, 3, 4, 5]
#         with self.assertRaisesRegex(ValueError, 'Times cannot be negative.'):
#             erlo.InverseProblem(self.model, times, self.values)

#         times = [5, 4, 3, 2, 1]
#         with self.assertRaisesRegex(ValueError, 'Times must be increasing.'):
#             erlo.InverseProblem(self.model, times, self.values)

#     def test_bad_values_input(self):
#         values = [1, 2, 3, 4, 5, 6, 7]
#         with self.assertRaisesRegex(ValueError, 'Values array must have'):
#             erlo.InverseProblem(self.model, self.times, values)

#         values = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
#         with self.assertRaisesRegex(ValueError, 'Values array must have'):
#             erlo.InverseProblem(self.model, self.times, values)

#     def test_evaluate(self):
#         parameters = [0.1, 1, 1, 1, 1]
#         output = self.problem.evaluate(parameters)

#         n_times = 5
#         n_outputs = 1
#         self.assertEqual(output.shape, (n_times, n_outputs))

#     def test_evaluateS1(self):
#         parameters = [0.1, 1, 1, 1, 1]
#         with self.assertRaises(NotImplementedError):
#             self.problem.evaluateS1(parameters)

#     def test_n_ouputs(self):
#         self.assertEqual(self.problem.n_outputs(), 1)

#     def test_n_parameters(self):
#         self.assertEqual(self.problem.n_parameters(), 5)

#     def test_n_times(self):
#         n_times = len(self.times)
#         self.assertEqual(self.problem.n_times(), n_times)

#     def test_times(self):
#         times = self.problem.times()
#         n_times = len(times)

#         self.assertEqual(n_times, 5)

#         self.assertEqual(times[0], self.times[0])
#         self.assertEqual(times[1], self.times[1])
#         self.assertEqual(times[2], self.times[2])
#         self.assertEqual(times[3], self.times[3])
#         self.assertEqual(times[4], self.times[4])

#     def test_values(self):
#         values = self.problem.values()

#         n_times = 5
#         n_outputs = 1
#         self.assertEqual(values.shape, (n_times, n_outputs))

#         self.assertEqual(values[0], self.values[0])
#         self.assertEqual(values[1], self.values[1])
#         self.assertEqual(values[2], self.values[2])
#         self.assertEqual(values[3], self.values[3])
#         self.assertEqual(values[4], self.values[4])


if __name__ == '__main__':
    unittest.main()
