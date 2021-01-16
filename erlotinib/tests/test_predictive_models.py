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


class TestDataDrivenPredictiveModel(unittest.TestCase):
    """
    Tests the erlo.DataDrivenPredictiveModel class.

    Since most methods only call methods from the
    erlo.PredictiveModel the methods are tested rather superficially.
    """
    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = erlo.PharmacodynamicModel(path)

        # Define error models
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        predictive_model = erlo.PredictiveModel(
            mechanistic_model, error_models)

        # Create data driven predictive model
        cls.model = erlo.DataDrivenPredictiveModel(
            predictive_model)

    def test_get_dosing_regimen(self):
        # Pass no final time
        regimen = self.model.get_dosing_regimen()
        self.assertIsNone(regimen)

        # Pass final time
        final_time = 10
        regimen = self.model.get_dosing_regimen(final_time)
        self.assertIsNone(regimen)

    def test_get_n_outputs(self):
        n_outputs = self.model.get_n_outputs()
        self.assertEqual(n_outputs, 1)

    def test_get_output_names(self):
        names = self.model.get_output_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'myokit.tumour_volume')

    def test_get_submodels(self):
        submodels = self.model.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)

    def test_sample(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.model.sample('times')

    def test_set_dosing_regimen(self):
        with self.assertRaisesRegex(AttributeError, 'The mechanistic model'):
            self.model.set_dosing_regimen(10, 2)


class TestPosteriorPredictiveModel(unittest.TestCase):
    """
    Tests the erlotinib.PosteriorPredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Test model I: Individual predictive model
        # Create predictive model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = erlo.PharmacodynamicModel(path)
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        cls.pred_model = erlo.PredictiveModel(
            mechanistic_model, error_models)

        # Create a posterior samples dataframe
        parameter_names = cls.pred_model.get_parameter_names()
        n_parameters = cls.pred_model.n_parameters()
        iterations = \
            [1] * 2 * n_parameters + \
            [2] * 2 * n_parameters + \
            [3] * 2 * n_parameters
        runs = [1] * n_parameters + [2] * n_parameters
        runs = runs * 3
        cls.posterior_samples = pd.DataFrame({
            'ID': 1,
            'Parameter': parameter_names * 6,
            'Sample': 42,
            'Iteration': iterations,
            'Run': runs})

        # Create posterior predictive model
        cls.model = erlo.PosteriorPredictiveModel(
            cls.pred_model, cls.posterior_samples)

        # Test model II: PredictivePopulation model
        pop_models = [
            erlo.PooledModel(),
            erlo.HeterogeneousModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.LogNormalModel()]
        cls.pred_pop_model = erlo.PredictivePopulationModel(
            cls.pred_model, pop_models)

        ids = [
            'Pooled',
            1,
            2,
            'Pooled',
            'Pooled',
            'Pooled',
            'Pooled',
            'Pooled',
            'Mean',
            'Std.'] * 6
        parameter_names = [
            'myokit.tumour_volume',
            'myokit.drug_concentration',
            'myokit.drug_concentration',
            'myokit.kappa',
            'myokit.lambda_0',
            'myokit.lambda_1',
            'Sigma base',
            'Sigma rel.',
            'Sigma rel.',
            'Sigma rel.']
        n_parameters = len(parameter_names)
        iterations = \
            [1] * 2 * n_parameters + \
            [2] * 2 * n_parameters + \
            [3] * 2 * n_parameters
        runs = [1] * n_parameters + [2] * n_parameters
        runs = runs * 3
        cls.pop_post_samples = pd.DataFrame({
            'ID': ids,
            'Parameter': parameter_names * 6,
            'Sample': 42,
            'Iteration': iterations,
            'Run': runs})

        cls.pop_model = erlo.PosteriorPredictiveModel(
            cls.pred_pop_model, cls.pop_post_samples)

    def test_bad_instantiation(self):
        # Posterior samples have the wrong type
        posterior_samples = 'Bad type'
        with self.assertRaisesRegex(TypeError, 'The posterior samples'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, posterior_samples)

        # Bad ID key
        id_key = 'Bad key'
        with self.assertRaisesRegex(ValueError, 'The posterior samples'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, id_key=id_key)

        # Bad sample key
        sample_key = 'Bad key'
        with self.assertRaisesRegex(ValueError, 'The posterior samples'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, sample_key=sample_key)

        # Bad param key
        param_key = 'Bad key'
        with self.assertRaisesRegex(ValueError, 'The posterior samples'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, param_key=param_key)

        # Bad iter key
        iter_key = 'Bad key'
        with self.assertRaisesRegex(ValueError, 'The posterior samples'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, iter_key=iter_key)

        # Bad run key
        run_key = 'Bad key'
        with self.assertRaisesRegex(ValueError, 'The posterior samples'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, run_key=run_key)

        # Bad parameter map type
        param_map = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The parameter map'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, param_map=param_map)

        # Non existent ID
        _id = 'Does not exist'
        with self.assertRaisesRegex(ValueError, 'The individual <Does not'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, individual=_id)

        # Set ID despite using PredictivePopulationModel
        _id = 'Some ID'
        with self.assertRaisesRegex(ValueError, "Individual ID's cannot be"):
            erlo.PosteriorPredictiveModel(
                self.pred_pop_model, self.posterior_samples, individual=_id)

        # Negative warm-up iterations
        warmup = -10
        with self.assertRaisesRegex(ValueError, 'The number of warm-up'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, warm_up_iter=warmup)

        # Too large warm-up iterations
        warmup = 10
        with self.assertRaisesRegex(ValueError, 'The number of warm-up'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, warm_up_iter=warmup)

        # The posterior does not have samples for all parameters
        mask = self.posterior_samples['Parameter'] != 'myokit.tumour_volume'
        posterior_samples = self.posterior_samples[mask]
        with self.assertRaisesRegex(ValueError, 'The parameter <myokit.'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, posterior_samples)

        # The parameter map does not identify all parameters correctly
        param_map = {'myokit.tumour_volume': 'Does not exist'}
        with self.assertRaisesRegex(ValueError, 'The parameter <Does not'):
            erlo.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, param_map=param_map)

    def test_sample(self):
        # Test case I: Just one sample
        times = [1, 2, 3, 4, 5]
        seed = 42
        samples = self.model.sample(times, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 10.25763928014699)
        self.assertAlmostEqual(values[1], 28.48348944331076)
        self.assertAlmostEqual(values[2], -24.592234016787202)
        self.assertAlmostEqual(values[3], -38.164271171263756)
        self.assertAlmostEqual(values[4], -83.65720486709867)

        # Test case II: More than one sample
        n_samples = 4
        samples = self.model.sample(
            times, n_samples=n_samples, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 20)
        self.assertAlmostEqual(values[0], 10.257639280108956)
        self.assertAlmostEqual(values[1], -83.6572048670368)
        self.assertAlmostEqual(values[2], -32.91098481323581)
        self.assertAlmostEqual(values[18], -39.007231487083104)
        self.assertAlmostEqual(values[19], -0.2477241745349836)

        # Test case III: include dosing regimen

        # Test case III.1: PD model
        samples = self.model.sample(times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 10.25763928014699)
        self.assertAlmostEqual(values[1], 28.48348944331076)
        self.assertAlmostEqual(values[2], -24.592234016787202)
        self.assertAlmostEqual(values[3], -38.164271171263756)
        self.assertAlmostEqual(values[4], -83.65720486709867)

        # Test case III.2: PK model, regimen not set
        path = erlo.ModelLibrary().one_compartment_pk_model()
        mechanistic_model = erlo.PharmacokineticModel(path)
        mechanistic_model.set_administration('central', direct=False)
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        predictive_model = erlo.PredictiveModel(
            mechanistic_model, error_models)

        # Define a map between the parameters to recycle posterior samples
        param_map = {
            'central.size': 'myokit.tumour_volume',
            'dose.absorption_rate': 'myokit.lambda_0',
            'myokit.elimination_rate': 'myokit.lambda_1',
            'central.drug_amount': 'myokit.drug_concentration',
            'dose.drug_amount': 'myokit.kappa'}
        model = erlo.PosteriorPredictiveModel(
            predictive_model, self.posterior_samples, param_map=param_map)

        # Sample
        samples = model.sample(times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 10.257639409972427)
        self.assertAlmostEqual(values[1], 28.483489440306272)
        self.assertAlmostEqual(values[2], -24.592234019552883)
        self.assertAlmostEqual(values[3], -38.16427115639449)
        self.assertAlmostEqual(values[4], -83.65720493975232)

        # Test case III.3: PK model, regimen set
        model.set_dosing_regimen(1, 1, duration=2, period=2, num=2)

        # Sample
        samples = model.sample(times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertTrue(np.isnan(sample_ids[1]))

        biomarkers = samples['Biomarker'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].dropna().unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 10.257639409972427)
        self.assertAlmostEqual(values[1], 28.48397119211002)
        self.assertAlmostEqual(values[2], -24.589501386669337)
        self.assertAlmostEqual(values[3], -38.17331623508392)
        self.assertAlmostEqual(values[4], -83.64232025329919)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 2)


class TestPredictiveModel(unittest.TestCase):
    """
    Tests the erlo.PredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.mechanistic_model = erlo.PharmacodynamicModel(path)

        # Define error models
        cls.error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        cls.model = erlo.PredictiveModel(
            cls.mechanistic_model, cls.error_models)

    def test_bad_instantiation(self):
        # Mechanistic model has wrong type
        mechanistic_model = 'wrong type'

        with self.assertRaisesRegex(TypeError, 'The mechanistic model'):
            erlo.PredictiveModel(mechanistic_model, self.error_models)

        # Error model has wrong type
        error_models = ['wrong type']

        with self.assertRaisesRegex(TypeError, 'All error models'):
            erlo.PredictiveModel(self.mechanistic_model, error_models)

        # Non-existent outputs
        outputs = ['Not', 'existent']

        with self.assertRaisesRegex(KeyError, 'The variable <Not> does not'):
            erlo.PredictiveModel(
                self.mechanistic_model, self.error_models, outputs)

        # Wrong number of error models
        error_models = [erlo.ErrorModel(), erlo.ErrorModel()]

        with self.assertRaisesRegex(ValueError, 'Wrong number of error'):
            erlo.PredictiveModel(self.mechanistic_model, error_models)

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        self.model.fix_parameters(name_value_dict={
            'myokit.tumour_volume': 1,
            'myokit.kappa': 1})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 5)

        parameter_names = self.model.get_parameter_names()
        self.assertEqual(len(parameter_names), 5)
        self.assertEqual(parameter_names[0], 'myokit.drug_concentration')
        self.assertEqual(parameter_names[1], 'myokit.lambda_0')
        self.assertEqual(parameter_names[2], 'myokit.lambda_1')
        self.assertEqual(parameter_names[3], 'Sigma base')
        self.assertEqual(parameter_names[4], 'Sigma rel.')

        # Test case II: fix overlapping set of parameters
        self.model.fix_parameters(name_value_dict={
            'myokit.kappa': None,
            'myokit.lambda_0': 0.5,
            'Sigma rel.': 0.3})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 4)

        parameter_names = self.model.get_parameter_names()
        self.assertEqual(len(parameter_names), 4)
        self.assertEqual(parameter_names[0], 'myokit.drug_concentration')
        self.assertEqual(parameter_names[1], 'myokit.kappa')
        self.assertEqual(parameter_names[2], 'myokit.lambda_1')
        self.assertEqual(parameter_names[3], 'Sigma base')

        # Test case III: unfix all parameters
        self.model.fix_parameters(name_value_dict={
            'myokit.tumour_volume': None,
            'myokit.lambda_0': None,
            'Sigma rel.': None})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 7)

        parameter_names = self.model.get_parameter_names()
        self.assertEqual(len(parameter_names), 7)
        self.assertEqual(parameter_names[0], 'myokit.tumour_volume')
        self.assertEqual(parameter_names[1], 'myokit.drug_concentration')
        self.assertEqual(parameter_names[2], 'myokit.kappa')
        self.assertEqual(parameter_names[3], 'myokit.lambda_0')
        self.assertEqual(parameter_names[4], 'myokit.lambda_1')
        self.assertEqual(parameter_names[5], 'Sigma base')
        self.assertEqual(parameter_names[6], 'Sigma rel.')

    def test_fix_parameters_bad_input(self):
        name_value_dict = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The name-value dictionary'):
            self.model.fix_parameters(name_value_dict)

    def test_get_n_outputs(self):
        self.assertEqual(self.model.get_n_outputs(), 1)

    def test_get_output_names(self):
        outputs = self.model.get_output_names()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'myokit.tumour_volume')

    def test_get_parameter_names(self):
        # Test case I: Single output problem
        names = self.model.get_parameter_names()

        self.assertEqual(len(names), 7)
        self.assertEqual(names[0], 'myokit.tumour_volume')
        self.assertEqual(names[1], 'myokit.drug_concentration')
        self.assertEqual(names[2], 'myokit.kappa')
        self.assertEqual(names[3], 'myokit.lambda_0')
        self.assertEqual(names[4], 'myokit.lambda_1')
        self.assertEqual(names[5], 'Sigma base')
        self.assertEqual(names[6], 'Sigma rel.')

        # Test case II: Multi-output problem
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel(),
            erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        model = erlo.PredictiveModel(model, error_models)

        names = model.get_parameter_names()

        self.assertEqual(len(names), 9)
        self.assertEqual(names[0], 'central.drug_amount')
        self.assertEqual(names[1], 'dose.drug_amount')
        self.assertEqual(names[2], 'central.size')
        self.assertEqual(names[3], 'dose.absorption_rate')
        self.assertEqual(names[4], 'myokit.elimination_rate')
        self.assertEqual(names[5], 'central.drug_amount Sigma base')
        self.assertEqual(names[6], 'central.drug_amount Sigma rel.')
        self.assertEqual(names[7], 'dose.drug_amount Sigma base')
        self.assertEqual(names[8], 'dose.drug_amount Sigma rel.')

    def test_get_set_dosing_regimen(self):
        # Test case I: Mechanistic model does not support dosing regimens
        # (PharmacodynaimcModel)
        with self.assertRaisesRegex(AttributeError, 'The mechanistic model'):
            self.model.set_dosing_regimen(1, 1, 1)

        self.assertIsNone(self.model.get_dosing_regimen())

        # Test case II: Mechanistic model supports dosing regimens
        path = erlo.ModelLibrary().one_compartment_pk_model()
        mechanistic_model = erlo.PharmacokineticModel(path)
        mechanistic_model.set_administration('central')
        model = erlo.PredictiveModel(
            mechanistic_model, self.error_models)

        # Test case II.1: Dosing regimen not set
        self.assertIsNone(model.get_dosing_regimen())

        # Test case II.2 Set single bolus dose
        model.set_dosing_regimen(dose=1, start=1)
        regimen_df = model.get_dosing_regimen()

        self.assertIsInstance(regimen_df, pd.DataFrame)

        keys = regimen_df.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], 'Time')
        self.assertEqual(keys[1], 'Duration')
        self.assertEqual(keys[2], 'Dose')

        times = regimen_df['Time'].to_numpy()
        self.assertEqual(len(times), 1)
        self.assertEqual(times[0], 1)

        durations = regimen_df['Duration'].unique()
        self.assertEqual(len(durations), 1)
        self.assertEqual(durations[0], 0.01)

        doses = regimen_df['Dose'].unique()
        self.assertEqual(len(doses), 1)
        self.assertEqual(doses[0], 1)

        # Test case II.3 Set single infusion
        model.set_dosing_regimen(dose=1, start=1, duration=1)
        regimen_df = model.get_dosing_regimen()

        self.assertIsInstance(regimen_df, pd.DataFrame)

        keys = regimen_df.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], 'Time')
        self.assertEqual(keys[1], 'Duration')
        self.assertEqual(keys[2], 'Dose')

        times = regimen_df['Time'].to_numpy()
        self.assertEqual(len(times), 1)
        self.assertEqual(times[0], 1)

        durations = regimen_df['Duration'].unique()
        self.assertEqual(len(durations), 1)
        self.assertEqual(durations[0], 1)

        doses = regimen_df['Dose'].unique()
        self.assertEqual(len(doses), 1)
        self.assertEqual(doses[0], 1)

        # Test case II.4 Multiple doses
        model.set_dosing_regimen(dose=1, start=1, period=1, num=3)
        regimen_df = model.get_dosing_regimen()

        self.assertIsInstance(regimen_df, pd.DataFrame)

        keys = regimen_df.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], 'Time')
        self.assertEqual(keys[1], 'Duration')
        self.assertEqual(keys[2], 'Dose')

        times = regimen_df['Time'].to_numpy()
        self.assertEqual(len(times), 3)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)

        durations = regimen_df['Duration'].unique()
        self.assertEqual(len(durations), 1)
        self.assertEqual(durations[0], 0.01)

        doses = regimen_df['Dose'].unique()
        self.assertEqual(len(doses), 1)
        self.assertEqual(doses[0], 1)

        # Set final time
        regimen_df = model.get_dosing_regimen(final_time=1.5)

        self.assertIsInstance(regimen_df, pd.DataFrame)

        keys = regimen_df.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], 'Time')
        self.assertEqual(keys[1], 'Duration')
        self.assertEqual(keys[2], 'Dose')

        times = regimen_df['Time'].to_numpy()
        self.assertEqual(len(times), 1)
        self.assertEqual(times[0], 1)

        durations = regimen_df['Duration'].unique()
        self.assertEqual(len(durations), 1)
        self.assertEqual(durations[0], 0.01)

        doses = regimen_df['Dose'].unique()
        self.assertEqual(len(doses), 1)
        self.assertEqual(doses[0], 1)

        # Set final time, such that regimen dataframe would be empty
        regimen_df = model.get_dosing_regimen(final_time=0)

        self.assertIsNone(regimen_df, pd.DataFrame)

        # Test case II.3 Indefinite dosing regimen
        model.set_dosing_regimen(dose=1, start=1, period=1)
        regimen_df = model.get_dosing_regimen()

        self.assertIsInstance(regimen_df, pd.DataFrame)

        keys = regimen_df.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], 'Time')
        self.assertEqual(keys[1], 'Duration')
        self.assertEqual(keys[2], 'Dose')

        times = regimen_df['Time'].to_numpy()
        self.assertEqual(len(times), 1)
        self.assertEqual(times[0], 1)

        durations = regimen_df['Duration'].unique()
        self.assertEqual(len(durations), 1)
        self.assertEqual(durations[0], 0.01)

        doses = regimen_df['Dose'].unique()
        self.assertEqual(len(doses), 1)
        self.assertEqual(doses[0], 1)

        # Set final time
        regimen_df = model.get_dosing_regimen(final_time=5)

        self.assertIsInstance(regimen_df, pd.DataFrame)

        keys = regimen_df.keys()
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], 'Time')
        self.assertEqual(keys[1], 'Duration')
        self.assertEqual(keys[2], 'Dose')

        times = regimen_df['Time'].to_numpy()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        durations = regimen_df['Duration'].unique()
        self.assertEqual(len(durations), 1)
        self.assertEqual(durations[0], 0.01)

        doses = regimen_df['Dose'].unique()
        self.assertEqual(len(doses), 1)
        self.assertEqual(doses[0], 1)

    def test_get_submodels(self):
        # Test case I: no fixed parameters
        submodels = self.model.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)

        # Test case II: some fixed parameters
        self.model.fix_parameters({
            'myokit.tumour_volume': 1,
            'Sigma rel.': 10})
        submodels = self.model.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)

        # Unfix parameter
        self.model.fix_parameters({
            'myokit.tumour_volume': None,
            'Sigma rel.': None})

    def test_n_parameters(self):
        self.assertEqual(self.model.n_parameters(), 7)

    def test_sample(self):
        # Test case I: Just one sample
        parameters = [1, 1, 1, 1, 1, 1, 0.1]
        times = [1, 2, 3, 4, 5]
        seed = 42

        # Test case I.1: Return as pd.DataFrame
        samples = self.model.sample(parameters, times, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.970159924388273)
        self.assertAlmostEqual(values[1], -0.3837168004345003)
        self.assertAlmostEqual(values[2], 1.3172158091846213)
        self.assertAlmostEqual(values[3], 1.4896478457110898)
        self.assertAlmostEqual(values[4], -1.4664469447762758)

        # Test case I.2: Return as numpy.ndarray
        samples = self.model.sample(
            parameters, times, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        n_samples = 1
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))
        self.assertAlmostEqual(samples[0, 0, 0], 0.970159924388273)
        self.assertAlmostEqual(samples[0, 1, 0], -0.3837168004345003)
        self.assertAlmostEqual(samples[0, 2, 0], 1.3172158091846213)
        self.assertAlmostEqual(samples[0, 3, 0], 1.4896478457110898)
        self.assertAlmostEqual(samples[0, 4, 0], -1.4664469447762758)

        # Test case II: More than one sample
        n_samples = 4

        # Test case .1: Return as pd.DataFrame
        samples = self.model.sample(
            parameters, times, n_samples=n_samples, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 20)
        self.assertAlmostEqual(values[0], 1.0556423390683263)
        self.assertAlmostEqual(values[1], -0.3270113841421633)
        self.assertAlmostEqual(values[2], 1.609052478543911)
        self.assertAlmostEqual(values[3], 1.6938106489072702)
        self.assertAlmostEqual(values[4], -1.3308066638991631)
        self.assertAlmostEqual(values[5], -0.6770137193349925)
        self.assertAlmostEqual(values[6], 0.8103166170457382)
        self.assertAlmostEqual(values[7], 0.3554210376910704)
        self.assertAlmostEqual(values[8], 0.5926284393333348)
        self.assertAlmostEqual(values[9], -0.24255566520628413)
        self.assertAlmostEqual(values[10], 1.5900163762325767)
        self.assertAlmostEqual(values[11], 1.3392789962107843)
        self.assertAlmostEqual(values[12], 0.5878641834748815)
        self.assertAlmostEqual(values[13], 1.6324903256719818)
        self.assertAlmostEqual(values[14], 1.0513958594002857)
        self.assertAlmostEqual(values[15], -0.24719096826112444)
        self.assertAlmostEqual(values[16], 0.8924949457952482)
        self.assertAlmostEqual(values[17], -0.47361160445867245)
        self.assertAlmostEqual(values[18], 1.364551743048893)
        self.assertAlmostEqual(values[19], 0.5143221311427919)

        # Test case II.2: Return as numpy.ndarray
        samples = self.model.sample(
            parameters, times, n_samples=n_samples, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))
        self.assertAlmostEqual(samples[0, 0, 0], 1.0556423390683263)
        self.assertAlmostEqual(samples[0, 0, 1], -0.3270113841421633)
        self.assertAlmostEqual(samples[0, 0, 2], 1.609052478543911)
        self.assertAlmostEqual(samples[0, 0, 3], 1.6938106489072702)
        self.assertAlmostEqual(samples[0, 1, 0], -1.3308066638991631)
        self.assertAlmostEqual(samples[0, 1, 1], -0.6770137193349925)
        self.assertAlmostEqual(samples[0, 1, 2], 0.8103166170457382)
        self.assertAlmostEqual(samples[0, 1, 3], 0.3554210376910704)
        self.assertAlmostEqual(samples[0, 2, 0], 0.5926284393333348)
        self.assertAlmostEqual(samples[0, 2, 1], -0.24255566520628413)
        self.assertAlmostEqual(samples[0, 2, 2], 1.5900163762325767)
        self.assertAlmostEqual(samples[0, 2, 3], 1.3392789962107843)
        self.assertAlmostEqual(samples[0, 3, 0], 0.5878641834748815)
        self.assertAlmostEqual(samples[0, 3, 1], 1.6324903256719818)
        self.assertAlmostEqual(samples[0, 3, 2], 1.0513958594002857)
        self.assertAlmostEqual(samples[0, 3, 3], -0.24719096826112444)
        self.assertAlmostEqual(samples[0, 4, 0], 0.8924949457952482)
        self.assertAlmostEqual(samples[0, 4, 1], -0.47361160445867245)
        self.assertAlmostEqual(samples[0, 4, 2], 1.364551743048893)
        self.assertAlmostEqual(samples[0, 4, 3], 0.5143221311427919)

        # Test case III: Return dosing regimen

        # Test case III.1: PDModel, dosing regimen is not returned even
        # if flag is True
        samples = self.model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.970159924388273)
        self.assertAlmostEqual(values[1], -0.3837168004345003)
        self.assertAlmostEqual(values[2], 1.3172158091846213)
        self.assertAlmostEqual(values[3], 1.4896478457110898)
        self.assertAlmostEqual(values[4], -1.4664469447762758)

        # Test case III.2: PKmodel, where the dosing regimen is not set
        path = erlo.ModelLibrary().one_compartment_pk_model()
        mechanistic_model = erlo.PharmacokineticModel(path)
        mechanistic_model.set_administration('central')
        model = erlo.PredictiveModel(
            mechanistic_model, self.error_models)

        # Sample
        parameters = [1, 1, 1, 1, 1]
        samples = model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.19357442536989605)
        self.assertAlmostEqual(values[1], -0.8873567434686567)
        self.assertAlmostEqual(values[2], 0.7844710370969462)
        self.assertAlmostEqual(values[3], 0.9585509622439399)
        self.assertAlmostEqual(values[4], -1.9500467417155718)

        # Test case III.3: PKmodel, dosing regimen is set
        model.set_dosing_regimen(1, 1, period=1, num=2)

        # Sample
        parameters = [1, 1, 1, 1, 1]
        samples = model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].dropna().unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.19357442536989605)
        self.assertAlmostEqual(values[1], -0.47051946530234423)
        self.assertAlmostEqual(values[2], 1.1301703133958951)
        self.assertAlmostEqual(values[3], 1.1414603643105294)
        self.assertAlmostEqual(values[4], -1.9399955984363169)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 0.01)

        # Test case III.4: PKmodel, dosing regimen is set, 2 samples
        # Sample
        parameters = [1, 1, 1, 1, 1]
        samples = model.sample(
            parameters, times, n_samples=2, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)

        biomarkers = samples['Biomarker'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].dropna().unique()
        self.assertEqual(len(values), 10)
        self.assertAlmostEqual(values[0], 0.9959660719183876)
        self.assertAlmostEqual(values[1], -0.3861061623036009)
        self.assertAlmostEqual(values[2], 1.2887071287477976)
        self.assertAlmostEqual(values[3], 2.0146427922545884)
        self.assertAlmostEqual(values[4], -1.1360658058662318)
        self.assertAlmostEqual(values[5], -1.2240387200366378)
        self.assertAlmostEqual(values[6], 0.4075153414639344)
        self.assertAlmostEqual(values[7], -0.3078411315299712)
        self.assertAlmostEqual(values[8], 0.12431122545485368)
        self.assertAlmostEqual(values[9], -0.7816727453841099)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 0.01)

    def test_sample_bad_input(self):
        # Parameters are not of length n_parameters
        parameters = ['wrong', 'length']
        times = [1, 2, 3, 4]

        with self.assertRaisesRegex(ValueError, 'The length of parameters'):
            self.model.sample(parameters, times)


class TestPredictivePopulationModel(unittest.TestCase):
    """
    Tests the erlo.PredictivePopulationModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic and error model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = erlo.PharmacodynamicModel(path)
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        cls.predictive_model = erlo.PredictiveModel(
            mechanistic_model, error_models)

        # Create population model
        cls.population_models = [
            erlo.HeterogeneousModel(),
            erlo.LogNormalModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel()]

        # Create predictive population model
        cls.model = erlo.PredictivePopulationModel(
            cls.predictive_model, cls.population_models)

    def test_instantiation(self):
        # Define order of population model with params
        # Get mechanistic and error model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = erlo.PharmacodynamicModel(path)
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        predictive_model = erlo.PredictiveModel(
            mechanistic_model, error_models)

        # Create population model
        population_models = [
            erlo.HeterogeneousModel(),
            erlo.LogNormalModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel()]

        params = [
            'Sigma base',
            'myokit.tumour_volume',
            'myokit.kappa',
            'myokit.lambda_0',
            'myokit.drug_concentration',
            'myokit.lambda_1',
            'Sigma rel.']

        # Create predictive population model
        model = erlo.PredictivePopulationModel(
            predictive_model, population_models, params)

        parameter_names = model.get_parameter_names()
        self.assertEqual(len(parameter_names), 8)
        self.assertEqual(parameter_names[0], 'Mean myokit.tumour_volume')
        self.assertEqual(parameter_names[1], 'Std. myokit.tumour_volume')
        self.assertEqual(
            parameter_names[2], 'Pooled myokit.drug_concentration')
        self.assertEqual(parameter_names[3], 'Pooled myokit.kappa')
        self.assertEqual(parameter_names[4], 'Pooled myokit.lambda_0')
        self.assertEqual(parameter_names[5], 'Pooled myokit.lambda_1')
        self.assertEqual(parameter_names[6], 'Heterogeneous Sigma base')
        self.assertEqual(parameter_names[7], 'Pooled Sigma rel.')

    def test_bad_instantiation(self):
        # Predictive model has wrong type
        predictive_model = 'wrong type'

        with self.assertRaisesRegex(TypeError, 'The predictive model'):
            erlo.PredictivePopulationModel(
                predictive_model, self.population_models)

        # Population model has wrong type
        pop_models = ['wrong type']

        with self.assertRaisesRegex(TypeError, 'All population models'):
            erlo.PredictivePopulationModel(
                self.predictive_model, pop_models)

        # Wrong number of population models
        pop_models = [erlo.HeterogeneousModel()] * 3

        with self.assertRaisesRegex(ValueError, 'One population model'):
            erlo.PredictivePopulationModel(
                self.predictive_model, pop_models)

        # Wrong number of parameters are specfied
        params = ['Too', 'few']

        with self.assertRaisesRegex(ValueError, 'Params does not have'):
            erlo.PredictivePopulationModel(
                self.predictive_model, self.population_models, params)

        # Params does not list existing parameters
        params = ['Do', 'not', 'exist', '!', '!', '!', '!']

        with self.assertRaisesRegex(ValueError, 'The parameter names in'):
            erlo.PredictivePopulationModel(
                self.predictive_model, self.population_models, params)

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        # (Heterogenous params cannot be fixed)
        self.model.fix_parameters(name_value_dict={
            'Heterogeneous myokit.tumour_volume': 1,
            'Mean myokit.drug_concentration': 1,
            'Pooled myokit.kappa': 1})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 6)

        parameter_names = self.model.get_parameter_names()
        self.assertEqual(len(parameter_names), 6)
        self.assertEqual(
            parameter_names[0], 'Heterogeneous myokit.tumour_volume')
        self.assertEqual(parameter_names[1], 'Std. myokit.drug_concentration')
        self.assertEqual(parameter_names[2], 'Pooled myokit.lambda_0')
        self.assertEqual(parameter_names[3], 'Pooled myokit.lambda_1')
        self.assertEqual(parameter_names[4], 'Pooled Sigma base')
        self.assertEqual(parameter_names[5], 'Pooled Sigma rel.')

        # Test case II: fix overlapping set of parameters
        self.model.fix_parameters(name_value_dict={
            'Pooled myokit.kappa': None,
            'Pooled myokit.lambda_0': 0.5,
            'Pooled Sigma rel.': 0.3})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 5)

        parameter_names = self.model.get_parameter_names()
        self.assertEqual(len(parameter_names), 5)
        self.assertEqual(
            parameter_names[0], 'Heterogeneous myokit.tumour_volume')
        self.assertEqual(parameter_names[1], 'Std. myokit.drug_concentration')
        self.assertEqual(parameter_names[2], 'Pooled myokit.kappa')
        self.assertEqual(parameter_names[3], 'Pooled myokit.lambda_1')
        self.assertEqual(parameter_names[4], 'Pooled Sigma base')

        # Test case III: unfix all parameters
        self.model.fix_parameters(name_value_dict={
            'Mean myokit.drug_concentration': None,
            'Pooled myokit.lambda_0': None,
            'Pooled Sigma rel.': None})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 8)

        parameter_names = self.model.get_parameter_names()
        self.assertEqual(len(parameter_names), 8)
        self.assertEqual(
            parameter_names[0], 'Heterogeneous myokit.tumour_volume')
        self.assertEqual(parameter_names[1], 'Mean myokit.drug_concentration')
        self.assertEqual(parameter_names[2], 'Std. myokit.drug_concentration')
        self.assertEqual(parameter_names[3], 'Pooled myokit.kappa')
        self.assertEqual(parameter_names[4], 'Pooled myokit.lambda_0')
        self.assertEqual(parameter_names[5], 'Pooled myokit.lambda_1')
        self.assertEqual(parameter_names[6], 'Pooled Sigma base')
        self.assertEqual(parameter_names[7], 'Pooled Sigma rel.')

    def test_fix_parameters_bad_input(self):
        name_value_dict = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The name-value dictionary'):
            self.model.fix_parameters(name_value_dict)

    def test_get_n_outputs(self):
        self.assertEqual(self.model.get_n_outputs(), 1)

    def test_get_output_names(self):
        outputs = self.model.get_output_names()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'myokit.tumour_volume')

    def test_get_parameter_names(self):
        # Test case I: Single output problem
        names = self.model.get_parameter_names()

        self.assertEqual(len(names), 8)
        self.assertEqual(
            names[0], 'Heterogeneous myokit.tumour_volume')
        self.assertEqual(names[1], 'Mean myokit.drug_concentration')
        self.assertEqual(names[2], 'Std. myokit.drug_concentration')
        self.assertEqual(names[3], 'Pooled myokit.kappa')
        self.assertEqual(names[4], 'Pooled myokit.lambda_0')
        self.assertEqual(names[5], 'Pooled myokit.lambda_1')
        self.assertEqual(names[6], 'Pooled Sigma base')
        self.assertEqual(names[7], 'Pooled Sigma rel.')

        # Test case II: Multi-output problem
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel(),
            erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        model = erlo.PredictiveModel(model, error_models)
        pop_models = self.population_models + [erlo.PooledModel()] * 2
        model = erlo.PredictivePopulationModel(model, pop_models)

        names = model.get_parameter_names()

        self.assertEqual(len(names), 10)
        self.assertEqual(
            names[0], 'Heterogeneous central.drug_amount')
        self.assertEqual(names[1], 'Mean dose.drug_amount')
        self.assertEqual(names[2], 'Std. dose.drug_amount')
        self.assertEqual(names[3], 'Pooled central.size')
        self.assertEqual(names[4], 'Pooled dose.absorption_rate')
        self.assertEqual(names[5], 'Pooled myokit.elimination_rate')
        self.assertEqual(names[6], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(names[7], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(names[8], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(names[9], 'Pooled dose.drug_amount Sigma rel.')

    def test_get_set_dosing_regimen(self):
        # This just wraps the method from the PredictiveModel. So shallow
        # tests should suffice.j
        ref_dosing_regimen = self.predictive_model.get_dosing_regimen()
        dosing_regimen = self.model.get_dosing_regimen()

        self.assertIsNone(ref_dosing_regimen)
        self.assertIsNone(dosing_regimen)

    def test_get_submodels(self):
        # Test case I: no fixed parameters
        submodels = self.model.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')
        self.assertEqual(keys[2], 'Population models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)

        pop_models = submodels['Population models']
        self.assertEqual(len(pop_models), 7)
        self.assertIsInstance(pop_models[0], erlo.PopulationModel)
        self.assertIsInstance(pop_models[1], erlo.PopulationModel)
        self.assertIsInstance(pop_models[2], erlo.PopulationModel)
        self.assertIsInstance(pop_models[3], erlo.PopulationModel)
        self.assertIsInstance(pop_models[4], erlo.PopulationModel)
        self.assertIsInstance(pop_models[5], erlo.PopulationModel)
        self.assertIsInstance(pop_models[6], erlo.PopulationModel)

        # Test case II: some fixed parameters
        self.model.fix_parameters({
            'Pooled myokit.kappa': 1,
            'Pooled Sigma rel.': 10})
        submodels = self.model.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')
        self.assertEqual(keys[2], 'Population models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)

        pop_models = submodels['Population models']
        self.assertEqual(len(pop_models), 7)
        self.assertIsInstance(pop_models[0], erlo.PopulationModel)
        self.assertIsInstance(pop_models[1], erlo.PopulationModel)
        self.assertIsInstance(pop_models[2], erlo.PopulationModel)
        self.assertIsInstance(pop_models[3], erlo.PopulationModel)
        self.assertIsInstance(pop_models[4], erlo.PopulationModel)
        self.assertIsInstance(pop_models[5], erlo.PopulationModel)
        self.assertIsInstance(pop_models[6], erlo.PopulationModel)

        # Unfix parameter
        self.model.fix_parameters({
            'Pooled myokit.kappa': None,
            'Pooled Sigma rel.': None})

    def test_n_parameters(self):
        self.assertEqual(self.model.n_parameters(), 8)

    def test_sample(self):
        # Test case I: Just one sample
        parameters = [1, 1, 1, 1, 1, 1, 0.1, 0.1]
        times = [1, 2, 3, 4, 5]
        seed = 42

        # Test case I.1: Return as pd.DataFrame
        samples = self.model.sample(parameters, times, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.706890541588044)
        self.assertAlmostEqual(values[1], 0.92377400455709)
        self.assertAlmostEqual(values[2], 0.6494614629969823)
        self.assertAlmostEqual(values[3], 0.8096804910998759)
        self.assertAlmostEqual(values[4], 0.7137169898523077)

        # Test case I.2: Return as numpy.ndarray
        samples = self.model.sample(
            parameters, times, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        n_samples = 1
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))
        self.assertAlmostEqual(samples[0, 0, 0], 0.706890541588044)
        self.assertAlmostEqual(samples[0, 1, 0], 0.92377400455709)
        self.assertAlmostEqual(samples[0, 2, 0], 0.6494614629969823)
        self.assertAlmostEqual(samples[0, 3, 0], 0.8096804910998759)
        self.assertAlmostEqual(samples[0, 4, 0], 0.7137169898523077)

        # Test case II: More than one sample
        n_samples = 4

        # Test case .1: Return as pd.DataFrame
        samples = self.model.sample(
            parameters, times, n_samples=n_samples, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 20)
        self.assertAlmostEqual(values[0], 0.706890541588044)
        self.assertAlmostEqual(values[1], 1.1819335813039527)
        self.assertAlmostEqual(values[18], 0.34250343758412044)
        self.assertAlmostEqual(values[19], 0.22479868804541864)

        # Test case II.2: Return as numpy.ndarray
        samples = self.model.sample(
            parameters, times, n_samples=n_samples, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))
        self.assertAlmostEqual(samples[0, 0, 0], 0.706890541588044)
        self.assertAlmostEqual(samples[0, 1, 2], 0.6028704149153622)
        self.assertAlmostEqual(samples[0, 2, 1], 1.801028233914049)
        self.assertAlmostEqual(samples[0, 3, 2], 0.46832698042688226)
        self.assertAlmostEqual(samples[0, 4, 3], 0.22479868804541864)

        # Test case III: Return dosing regimen

        # Test case III.1: PDModel, dosing regimen is not returned even
        # if flag is True
        samples = self.model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.706890541588044)
        self.assertAlmostEqual(values[1], 0.92377400455709)
        self.assertAlmostEqual(values[2], 0.6494614629969823)
        self.assertAlmostEqual(values[3], 0.8096804910998759)
        self.assertAlmostEqual(values[4], 0.7137169898523077)

        # Test case III.2: PKmodel, where the dosing regimen is not set
        path = erlo.ModelLibrary().one_compartment_pk_model()
        mechanistic_model = erlo.PharmacokineticModel(path)
        mechanistic_model.set_administration('central', direct=False)
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        predictive_model = erlo.PredictiveModel(
            mechanistic_model, error_models)
        model = erlo.PredictivePopulationModel(
            predictive_model, self.population_models)

        # Sample
        parameters = [1, 1, 1, 1, 1, 1, 0.1, 0.1]
        samples = model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.6010875382040474)
        self.assertAlmostEqual(values[1], 0.5498109476992113)
        self.assertAlmostEqual(values[2], 0.19328049998332394)
        self.assertAlmostEqual(values[3], 0.22922447690250458)
        self.assertAlmostEqual(values[4], 0.058015876021734025)

        # Test case III.3: PKmodel, dosing regimen is set
        model.set_dosing_regimen(1, 1, period=1, num=2)

        # Sample
        samples = model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].dropna().unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.6010875382040474)
        self.assertAlmostEqual(values[1], 0.9494472511510463)
        self.assertAlmostEqual(values[2], 0.7916608762100437)
        self.assertAlmostEqual(values[3], 0.6678547210989847)
        self.assertAlmostEqual(values[4], 0.30760362908683914)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 0.01)

        # Test case III.4: PKmodel, dosing regimen is set, 2 samples
        # Sample
        samples = model.sample(
            parameters, times, n_samples=2, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)

        biomarkers = samples['Biomarker'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].dropna().unique()
        self.assertEqual(len(values), 10)
        self.assertAlmostEqual(values[0], 0.6010875382040474)
        self.assertAlmostEqual(values[1], 0.3979077263135499)
        self.assertAlmostEqual(values[8], 0.30760362908683914)
        self.assertAlmostEqual(values[9], 0.28454670539055454)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 0.01)

    def test_sample_bad_input(self):
        # Parameters are not of length n_parameters
        parameters = ['wrong', 'length']
        times = [1, 2, 3, 4]

        with self.assertRaisesRegex(ValueError, 'The length of parameters'):
            self.model.sample(parameters, times)


class TestPriorPredictiveModel(unittest.TestCase):
    """
    Tests the erlo.PriorPredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = erlo.PharmacodynamicModel(path)

        # Define error models
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        cls.predictive_model = erlo.PredictiveModel(
            mechanistic_model, error_models)

        # Create prior
        cls.log_prior = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(1, 2),
            pints.UniformLogPrior(2, 3),
            pints.UniformLogPrior(3, 4),
            pints.UniformLogPrior(4, 5),
            pints.UniformLogPrior(5, 6),
            pints.UniformLogPrior(6, 7))

        # Create prior predictive model
        cls.model = erlo.PriorPredictiveModel(
            cls.predictive_model, cls.log_prior)

    def test_bad_instantiation(self):
        # Predictive model has wrong type
        predictive_model = 'wrong type'

        with self.assertRaisesRegex(ValueError, 'The provided predictive'):
            erlo.PriorPredictiveModel(predictive_model, self.log_prior)

        # Prior has woring type
        log_prior = 'wrong type'

        with self.assertRaisesRegex(ValueError, 'The provided log-prior'):
            erlo.PriorPredictiveModel(self.predictive_model, log_prior)

        # Dimension of predictive model and log-prior don't match
        log_prior = pints.UniformLogPrior(0, 1)  # dim 1, but 7 params

        with self.assertRaisesRegex(ValueError, 'The dimension of the'):
            erlo.PriorPredictiveModel(self.predictive_model, log_prior)

    def test_sample(self):
        # Test case I: Just one sample
        times = [1, 2, 3, 4, 5]
        seed = 42
        samples = self.model.sample(times, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 2.8622881485041396)
        self.assertAlmostEqual(values[1], 3.7272644272099664)
        self.assertAlmostEqual(values[2], -2.5604320890107455)
        self.assertAlmostEqual(values[3], -5.445074975020219)
        self.assertAlmostEqual(values[4], -8.562594546870663)

        # Test case II: More than one sample
        n_samples = 4
        samples = self.model.sample(
            times, n_samples=n_samples, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 20)
        self.assertAlmostEqual(values[0], 3.6599791844429284)
        self.assertAlmostEqual(values[1], -12.696938587267084)
        self.assertAlmostEqual(values[2], -3.82460662961628)
        self.assertAlmostEqual(values[3], -4.103207219325659)
        self.assertAlmostEqual(values[4], -5.196420346964001)
        self.assertAlmostEqual(values[5], 10.726522931974097)
        self.assertAlmostEqual(values[6], 1.4866633054676286)
        self.assertAlmostEqual(values[7], 5.48736409468915)
        self.assertAlmostEqual(values[8], -4.211329523375031)
        self.assertAlmostEqual(values[9], -2.38819374047191)
        self.assertAlmostEqual(values[10], -3.6298125294796812)
        self.assertAlmostEqual(values[11], 9.209895487514647)
        self.assertAlmostEqual(values[12], -6.256268368313989)
        self.assertAlmostEqual(values[13], -5.03014957524413)
        self.assertAlmostEqual(values[14], 6.367870976692225)
        self.assertAlmostEqual(values[15], -1.2252254747096893)
        self.assertAlmostEqual(values[16], -0.7853509638638059)
        self.assertAlmostEqual(values[17], 12.177527343575)
        self.assertAlmostEqual(values[18], -6.435165240274607)
        self.assertAlmostEqual(values[19], 10.471501140030037)

        # Test case III: include dosing regimen

        # Test case III.1: PD model
        samples = self.model.sample(times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 2.8622881485041396)
        self.assertAlmostEqual(values[1], 3.7272644272099664)
        self.assertAlmostEqual(values[2], -2.5604320890107455)
        self.assertAlmostEqual(values[3], -5.445074975020219)
        self.assertAlmostEqual(values[4], -8.562594546870663)

        # Test case III.2: PK model, regimen not set
        path = erlo.ModelLibrary().one_compartment_pk_model()
        mechanistic_model = erlo.PharmacokineticModel(path)
        mechanistic_model.set_administration('central')
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        predictive_model = erlo.PredictiveModel(
            mechanistic_model, error_models)
        log_prior = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(1, 2),
            pints.UniformLogPrior(2, 3),
            pints.UniformLogPrior(3, 4),
            pints.UniformLogPrior(4, 5))
        model = erlo.PriorPredictiveModel(predictive_model, log_prior)

        # Sample
        samples = model.sample(times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Biomarker'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.9418268811969496)
        self.assertAlmostEqual(values[1], 2.4414001899620565)
        self.assertAlmostEqual(values[2], -2.1070223214978583)
        self.assertAlmostEqual(values[3], -3.2700124414629426)
        self.assertAlmostEqual(values[4], -7.167939155896637)

        # Test case III.3: PK model, regimen set
        model.set_dosing_regimen(1, 1, duration=2, period=2, num=2)

        # Sample
        samples = model.sample(times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertTrue(np.isnan(sample_ids[1]))

        biomarkers = samples['Biomarker'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].dropna().unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.9418268811969484)
        self.assertAlmostEqual(values[1], 2.535202697375215)
        self.assertAlmostEqual(values[2], -1.9337139346520897)
        self.assertAlmostEqual(values[3], -3.481678813431062)
        self.assertAlmostEqual(values[4], -6.595926429217902)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 2)


if __name__ == '__main__':
    unittest.main()
