#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pandas as pd
import pints
import xarray as xr

import chi
from chi.library import ModelLibrary


class TestGenerativeModel(unittest.TestCase):
    """
    Tests the chi.GenerativeModel class.

    Since most methods only call methods from the
    chi.PredictiveModel the methods are tested rather superficially.
    """
    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(path)

        # Define error models
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Create data driven predictive model
        cls.model = chi.GenerativeModel(
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

    def test_get_predictive_model(self):
        predictive_model = self.model.get_predictive_model()

        self.assertIsInstance(predictive_model, chi.PredictiveModel)

    def test_sample(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.model.sample('times')

    def test_set_dosing_regimen(self):
        with self.assertRaisesRegex(AttributeError, 'The mechanistic model'):
            self.model.set_dosing_regimen(10, 2)


class TestPosteriorPredictiveModel(unittest.TestCase):
    """
    Tests the chi.PosteriorPredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Test model I: Individual predictive model
        # Create predictive model
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(path)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]
        cls.pred_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Create a posterior samples
        n_chains = 2
        n_draws = 3
        n_ids = 1
        samples = np.ones(shape=(n_chains, n_draws, n_ids))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 1']})
        cls.posterior_samples = xr.Dataset({
            param: samples for param
            in cls.pred_model.get_parameter_names()})

        # Create posterior predictive model
        cls.model = chi.PosteriorPredictiveModel(
            cls.pred_model, cls.posterior_samples)

        # Test model II: PredictivePopulation model
        pop_models = [
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel()]
        cls.pred_pop_model = chi.PredictivePopulationModel(
            cls.pred_model, pop_models)

        # Create a posterior samples
        n_chains = 2
        n_draws = 3
        n_ids = 2
        samples = np.ones(shape=(n_chains, n_draws, n_ids))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 1', 'ID 2']})
        pop_samples = xr.DataArray(
            data=samples[:, :, 0],
            dims=['chain', 'draw'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        cls.pop_post_samples = xr.Dataset({
            'Pooled myokit.tumour_volume': pop_samples,
            'myokit.drug_concentration': samples,
            'Pooled myokit.kappa': pop_samples,
            'Pooled myokit.lambda_0': pop_samples,
            'Pooled myokit.lambda_1': pop_samples,
            'Pooled Sigma base': pop_samples,
            'Sigma rel.': samples,
            'Mean log Sigma rel.': pop_samples,
            'Std. log Sigma rel.': pop_samples})

        cls.pop_model = chi.PosteriorPredictiveModel(
            cls.pred_pop_model, cls.pop_post_samples)

    def test_bad_instantiation(self):
        # Posterior samples have the wrong type
        posterior_samples = 'Bad type'
        with self.assertRaisesRegex(TypeError, 'The posterior samples'):
            chi.PosteriorPredictiveModel(
                self.pred_model, posterior_samples)

        # The dimensions have the wrong names (3 dimensions)
        posterior_samples = self.pop_post_samples.copy()
        posterior_samples = posterior_samples.rename(
            {'chain': 'wrong name'})
        with self.assertRaisesRegex(ValueError, 'The posterior samples'):
            chi.PosteriorPredictiveModel(
                self.pred_model, posterior_samples)

        # The dimensions have the wrong names (2 dimensions)
        posterior_samples = posterior_samples.drop_dims('individual')
        with self.assertRaisesRegex(ValueError, 'The posterior samples'):
            chi.PosteriorPredictiveModel(
                self.pred_model, posterior_samples)

        # The dimensions are just generally wrong
        samples = posterior_samples.drop_dims('draw')
        with self.assertRaisesRegex(ValueError, 'The posterior samples'):
            chi.PosteriorPredictiveModel(
                self.pred_model, samples)

        # Bad parameter map type
        param_map = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The parameter map'):
            chi.PosteriorPredictiveModel(
                self.pred_model, self.posterior_samples, param_map=param_map)

        # The posterior does not have samples for all parameters
        # (Drop dims removes all parameters with the dimension 'individual')
        posterior_samples = posterior_samples.rename(
            {'wrong name': 'chain'})
        with self.assertRaisesRegex(ValueError, 'The parameter <myokit.'):
            chi.PosteriorPredictiveModel(
                self.pred_model, posterior_samples)

    def test_sample(self):
        # Test case I: Just one sample
        times = [1, 2, 3, 4, 5]
        samples = self.model.sample(times)

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

        # Test case II: More than one sample
        n_samples = 4
        samples = self.model.sample(times, n_samples=n_samples)

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

        # Test case III: include dosing regimen

        # Test case III.1: PD model
        samples = self.model.sample(times, include_regimen=True)

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

        # Test case III.2: PK model, regimen not set
        path = ModelLibrary().one_compartment_pk_model()
        mechanistic_model = chi.PharmacokineticModel(path)
        mechanistic_model.set_administration('central', direct=False)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]
        predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Define a map between the parameters to recycle posterior samples
        param_map = {
            'central.size': 'myokit.tumour_volume',
            'dose.absorption_rate': 'myokit.lambda_0',
            'myokit.elimination_rate': 'myokit.lambda_1',
            'central.drug_amount': 'myokit.drug_concentration',
            'dose.drug_amount': 'myokit.kappa'}
        model = chi.PosteriorPredictiveModel(
            predictive_model, self.posterior_samples, param_map=param_map)

        # Sample
        samples = model.sample(times, include_regimen=True)

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

        # Test case III.3: PK model, regimen set
        model.set_dosing_regimen(1, 1, duration=2, period=2, num=2)

        # Sample
        samples = model.sample(times, include_regimen=True)

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

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 2)

        # Test IV: Pooled model
        # Create a posterior samples
        n_chains = 2
        n_draws = 3
        samples = np.ones(shape=(n_chains, n_draws))
        pop_samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        pop_post_samples = xr.Dataset({
            'Pooled myokit.tumour_volume': pop_samples,
            'Pooled myokit.drug_concentration': pop_samples,
            'Pooled myokit.kappa': pop_samples,
            'Pooled myokit.lambda_0': pop_samples,
            'Pooled myokit.lambda_1': pop_samples,
            'Pooled Sigma base': pop_samples,
            'Pooled Sigma rel.': pop_samples})

        pop_models = [
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel()]
        pred_pop_model = chi.PredictivePopulationModel(
            self.pred_model, pop_models)
        pop_model = chi.PosteriorPredictiveModel(
            pred_pop_model, pop_post_samples)

        samples = pop_model.sample(times, include_regimen=True)

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

    def test_sample_bad_individual(self):
        # Individual does not exist
        _id = 'some ID'
        times = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, 'The individual <some ID>'):
            self.model.sample(times, individual=_id)

        # Try to seclect individual despite having a population model
        with self.assertRaisesRegex(ValueError, "Individual ID's cannot be"):
            self.pop_model.sample(times, individual=_id)


class TestPredictiveModel(unittest.TestCase):
    """
    Tests the chi.PredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.mechanistic_model = chi.PharmacodynamicModel(path)

        # Define error models
        cls.error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        cls.model = chi.PredictiveModel(
            cls.mechanistic_model, cls.error_models)

    def test_bad_instantiation(self):
        # Mechanistic model has wrong type
        mechanistic_model = 'wrong type'

        with self.assertRaisesRegex(TypeError, 'The mechanistic model'):
            chi.PredictiveModel(mechanistic_model, self.error_models)

        # Error model has wrong type
        error_models = ['wrong type']

        with self.assertRaisesRegex(TypeError, 'All error models'):
            chi.PredictiveModel(self.mechanistic_model, error_models)

        # Non-existent outputs
        outputs = ['Not', 'existent']

        with self.assertRaisesRegex(KeyError, 'The variable <Not> does not'):
            chi.PredictiveModel(
                self.mechanistic_model, self.error_models, outputs)

        # Wrong number of error models
        error_models = [chi.ErrorModel(), chi.ErrorModel()]

        with self.assertRaisesRegex(ValueError, 'Wrong number of error'):
            chi.PredictiveModel(self.mechanistic_model, error_models)

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
        path = ModelLibrary().one_compartment_pk_model()
        model = chi.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel(),
            chi.ConstantAndMultiplicativeGaussianErrorModel()]
        model = chi.PredictiveModel(model, error_models)

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
        path = ModelLibrary().one_compartment_pk_model()
        mechanistic_model = chi.PharmacokineticModel(path)
        mechanistic_model.set_administration('central')
        model = chi.PredictiveModel(
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
        self.assertIsInstance(mechanistic_model, chi.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], chi.ErrorModel)

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
        self.assertIsInstance(mechanistic_model, chi.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], chi.ErrorModel)

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
        path = ModelLibrary().one_compartment_pk_model()
        mechanistic_model = chi.PharmacokineticModel(path)
        mechanistic_model.set_administration('central')
        model = chi.PredictiveModel(
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
    Tests the chi.PredictivePopulationModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic and error model
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(path)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        cls.predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Create population model
        cls.population_models = [
            chi.HeterogeneousModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel()]

        # Create predictive population model
        cls.model = chi.PredictivePopulationModel(
            cls.predictive_model, cls.population_models)

    def test_instantiation(self):
        # Define order of population model with params
        # Get mechanistic and error model
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(path)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Create population model
        population_models = [
            chi.HeterogeneousModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel()]

        params = [
            'Sigma base',
            'myokit.tumour_volume',
            'myokit.kappa',
            'myokit.lambda_0',
            'myokit.drug_concentration',
            'myokit.lambda_1',
            'Sigma rel.']

        # Create predictive population model
        model = chi.PredictivePopulationModel(
            predictive_model, population_models, params)

        parameter_names = model.get_parameter_names()
        self.assertEqual(len(parameter_names), 8)
        self.assertEqual(parameter_names[0], 'Mean log myokit.tumour_volume')
        self.assertEqual(parameter_names[1], 'Std. log myokit.tumour_volume')
        self.assertEqual(
            parameter_names[2], 'Pooled myokit.drug_concentration')
        self.assertEqual(parameter_names[3], 'Pooled myokit.kappa')
        self.assertEqual(parameter_names[4], 'Pooled myokit.lambda_0')
        self.assertEqual(parameter_names[5], 'Pooled myokit.lambda_1')
        self.assertEqual(parameter_names[6], 'Sigma base')
        self.assertEqual(parameter_names[7], 'Pooled Sigma rel.')

    def test_bad_instantiation(self):
        # Predictive model has wrong type
        predictive_model = 'wrong type'

        with self.assertRaisesRegex(TypeError, 'The predictive model'):
            chi.PredictivePopulationModel(
                predictive_model, self.population_models)

        # Population model has wrong type
        pop_models = ['wrong type']

        with self.assertRaisesRegex(TypeError, 'All population models'):
            chi.PredictivePopulationModel(
                self.predictive_model, pop_models)

        # Wrong number of population models
        pop_models = [chi.HeterogeneousModel()] * 3

        with self.assertRaisesRegex(ValueError, 'One population model'):
            chi.PredictivePopulationModel(
                self.predictive_model, pop_models)

        # Wrong number of parameters are specfied
        params = ['Too', 'few']

        with self.assertRaisesRegex(ValueError, 'Params does not have'):
            chi.PredictivePopulationModel(
                self.predictive_model, self.population_models, params)

        # Params does not list existing parameters
        params = ['Do', 'not', 'exist', '!', '!', '!', '!']

        with self.assertRaisesRegex(ValueError, 'The parameter names in'):
            chi.PredictivePopulationModel(
                self.predictive_model, self.population_models, params)

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        # (Heterogenous params cannot be fixed)
        self.model.fix_parameters(name_value_dict={
            'myokit.tumour_volume': 1,
            'Mean log myokit.drug_concentration': 1,
            'Pooled myokit.kappa': 1})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 6)

        parameter_names = self.model.get_parameter_names()
        self.assertEqual(len(parameter_names), 6)
        self.assertEqual(
            parameter_names[0], 'myokit.tumour_volume')
        self.assertEqual(
            parameter_names[1], 'Std. log myokit.drug_concentration')
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
            parameter_names[0], 'myokit.tumour_volume')
        self.assertEqual(
            parameter_names[1], 'Std. log myokit.drug_concentration')
        self.assertEqual(parameter_names[2], 'Pooled myokit.kappa')
        self.assertEqual(parameter_names[3], 'Pooled myokit.lambda_1')
        self.assertEqual(parameter_names[4], 'Pooled Sigma base')

        # Test case III: unfix all parameters
        self.model.fix_parameters(name_value_dict={
            'Mean log myokit.drug_concentration': None,
            'Pooled myokit.lambda_0': None,
            'Pooled Sigma rel.': None})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 8)

        parameter_names = self.model.get_parameter_names()
        self.assertEqual(len(parameter_names), 8)
        self.assertEqual(
            parameter_names[0], 'myokit.tumour_volume')
        self.assertEqual(
            parameter_names[1], 'Mean log myokit.drug_concentration')
        self.assertEqual(
            parameter_names[2], 'Std. log myokit.drug_concentration')
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
            names[0], 'myokit.tumour_volume')
        self.assertEqual(names[1], 'Mean log myokit.drug_concentration')
        self.assertEqual(names[2], 'Std. log myokit.drug_concentration')
        self.assertEqual(names[3], 'Pooled myokit.kappa')
        self.assertEqual(names[4], 'Pooled myokit.lambda_0')
        self.assertEqual(names[5], 'Pooled myokit.lambda_1')
        self.assertEqual(names[6], 'Pooled Sigma base')
        self.assertEqual(names[7], 'Pooled Sigma rel.')

        # Test case II: Multi-output problem
        path = ModelLibrary().one_compartment_pk_model()
        model = chi.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel(),
            chi.ConstantAndMultiplicativeGaussianErrorModel()]
        model = chi.PredictiveModel(model, error_models)
        pop_models = self.population_models + [chi.PooledModel()] * 2
        model = chi.PredictivePopulationModel(model, pop_models)

        names = model.get_parameter_names()

        self.assertEqual(len(names), 10)
        self.assertEqual(
            names[0], 'central.drug_amount')
        self.assertEqual(names[1], 'Mean log dose.drug_amount')
        self.assertEqual(names[2], 'Std. log dose.drug_amount')
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
        self.assertIsInstance(mechanistic_model, chi.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], chi.ErrorModel)

        pop_models = submodels['Population models']
        self.assertEqual(len(pop_models), 7)
        self.assertIsInstance(pop_models[0], chi.PopulationModel)
        self.assertIsInstance(pop_models[1], chi.PopulationModel)
        self.assertIsInstance(pop_models[2], chi.PopulationModel)
        self.assertIsInstance(pop_models[3], chi.PopulationModel)
        self.assertIsInstance(pop_models[4], chi.PopulationModel)
        self.assertIsInstance(pop_models[5], chi.PopulationModel)
        self.assertIsInstance(pop_models[6], chi.PopulationModel)

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
        self.assertIsInstance(mechanistic_model, chi.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 1)
        self.assertIsInstance(error_models[0], chi.ErrorModel)

        pop_models = submodels['Population models']
        self.assertEqual(len(pop_models), 7)
        self.assertIsInstance(pop_models[0], chi.PopulationModel)
        self.assertIsInstance(pop_models[1], chi.PopulationModel)
        self.assertIsInstance(pop_models[2], chi.PopulationModel)
        self.assertIsInstance(pop_models[3], chi.PopulationModel)
        self.assertIsInstance(pop_models[4], chi.PopulationModel)
        self.assertIsInstance(pop_models[5], chi.PopulationModel)
        self.assertIsInstance(pop_models[6], chi.PopulationModel)

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

        # Test case I.2: Return as numpy.ndarray
        samples = self.model.sample(
            parameters, times, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        n_samples = 1
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))

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

        # Test case II.2: Return as numpy.ndarray
        samples = self.model.sample(
            parameters, times, n_samples=n_samples, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))

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

        # Test case III.2: PKmodel, where the dosing regimen is not set
        path = ModelLibrary().one_compartment_pk_model()
        mechanistic_model = chi.PharmacokineticModel(path)
        mechanistic_model.set_administration('central', direct=False)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]
        predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)
        model = chi.PredictivePopulationModel(
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
    Tests the chi.PriorPredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(path)

        # Define error models
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        cls.predictive_model = chi.PredictiveModel(
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
        cls.model = chi.PriorPredictiveModel(
            cls.predictive_model, cls.log_prior)

    def test_bad_instantiation(self):
        # Predictive model has wrong type
        predictive_model = 'wrong type'

        with self.assertRaisesRegex(ValueError, 'The provided predictive'):
            chi.PriorPredictiveModel(predictive_model, self.log_prior)

        # Prior has woring type
        log_prior = 'wrong type'

        with self.assertRaisesRegex(ValueError, 'The provided log-prior'):
            chi.PriorPredictiveModel(self.predictive_model, log_prior)

        # Dimension of predictive model and log-prior don't match
        log_prior = pints.UniformLogPrior(0, 1)  # dim 1, but 7 params

        with self.assertRaisesRegex(ValueError, 'The dimension of the'):
            chi.PriorPredictiveModel(self.predictive_model, log_prior)

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

        # Test case III.2: PK model, regimen not set
        path = ModelLibrary().one_compartment_pk_model()
        mechanistic_model = chi.PharmacokineticModel(path)
        mechanistic_model.set_administration('central')
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]
        predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)
        log_prior = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(1, 2),
            pints.UniformLogPrior(2, 3),
            pints.UniformLogPrior(3, 4),
            pints.UniformLogPrior(4, 5))
        model = chi.PriorPredictiveModel(predictive_model, log_prior)

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

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 2)


class TestStackedPredictiveModel(unittest.TestCase):
    """
    Tests the chi.StackedPredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Test model I: Individual predictive model
        # Create predictive model
        path = ModelLibrary().tumour_growth_inhibition_model_koch()
        mechanistic_model = chi.PharmacodynamicModel(path)
        error_models = [chi.GaussianErrorModel()]
        pred_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Create a posterior samples
        n_chains = 2
        n_draws = 3
        n_ids = 1
        samples = np.ones(shape=(n_chains, n_draws, n_ids))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 1']})
        posterior_samples = xr.Dataset({
            param: samples for param
            in pred_model.get_parameter_names()})

        # Create posterior predictive model
        cls.model_1 = chi.PosteriorPredictiveModel(
            pred_model, posterior_samples)

        # Test model II: Erlotinib PKPD model
        # Create predictive model
        path = ModelLibrary().erlotinib_tumour_growth_inhibition_model()
        mechanistic_model = chi.PharmacokineticModel(path)
        mechanistic_model.set_administration('central', direct=True)
        mechanistic_model.set_outputs(['myokit.tumour_volume'])
        error_models = [chi.GaussianErrorModel()]
        pred_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Create a posterior samples
        n_chains = 2
        n_draws = 3
        n_ids = 1
        samples = np.ones(shape=(n_chains, n_draws, n_ids))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws)),
                'individual': ['ID 1']})
        posterior_samples = xr.Dataset({
            param: samples for param
            in pred_model.get_parameter_names()})

        # Create posterior predictive model
        cls.model_2 = chi.PosteriorPredictiveModel(
            pred_model, posterior_samples)

        # Define stacked model
        cls.weights = [2, 1]
        cls.stacked_model = chi.StackedPredictiveModel(
            predictive_models=[cls.model_1, cls.model_2],
            weights=cls.weights)

    def test_bad_instantiation(self):
        # Models have the wrong type
        models = ['wrong', 'type']
        with self.assertRaisesRegex(TypeError, 'The predictive models'):
            chi.StackedPredictiveModel(
                models, self.weights)

        # The models have a different number of outputs
        path = ModelLibrary().erlotinib_tumour_growth_inhibition_model()
        mechanistic_model = chi.PharmacokineticModel(path)
        mechanistic_model.set_outputs(
            ['central.drug_concentration', 'myokit.tumour_volume'])
        error_models = [chi.GaussianErrorModel()] * 2
        pred_model = chi.PredictiveModel(
            mechanistic_model, error_models)
        samples = np.ones(shape=(2, 3, 1))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(2)),
                'draw': list(range(3)),
                'individual': ['ID 1']})
        posterior_samples = xr.Dataset({
            param: samples for param
            in pred_model.get_parameter_names()})
        model_2 = chi.PosteriorPredictiveModel(
            pred_model, posterior_samples)
        with self.assertRaisesRegex(ValueError, 'All predictive models'):
            chi.StackedPredictiveModel(
                [self.model_1, model_2], self.weights)

        # The models' ouptuts have different names
        mechanistic_model.set_outputs(['central.drug_concentration'])
        error_models = [chi.GaussianErrorModel()]
        pred_model = chi.PredictiveModel(
            mechanistic_model, error_models)
        samples = np.ones(shape=(2, 3, 1))
        samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw', 'individual'],
            coords={
                'chain': list(range(2)),
                'draw': list(range(3)),
                'individual': ['ID 1']})
        posterior_samples = xr.Dataset({
            param: samples for param
            in pred_model.get_parameter_names()})
        model_2 = chi.PosteriorPredictiveModel(
            pred_model, posterior_samples)
        with self.assertRaisesRegex(Warning, 'The predictive models appear'):
            chi.StackedPredictiveModel(
                [self.model_1, model_2], self.weights)

        # The number of models and the number of weights do not
        # coincide
        weights = ['too', 'many', 'weights']
        with self.assertRaisesRegex(ValueError, 'The model weights must be'):
            chi.StackedPredictiveModel(
                [self.model_1, self.model_2], weights)

    def test_get_predictive_model(self):
        predictive_models = self.stacked_model.get_predictive_model()

        self.assertEqual(len(predictive_models), 2)
        self.assertIsInstance(
            predictive_models[0], chi.PosteriorPredictiveModel)
        self.assertIsInstance(
            predictive_models[1], chi.PosteriorPredictiveModel)

    def test_get_weights(self):
        weights = self.stacked_model.get_weights()

        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0], 2 / 3)
        self.assertEqual(weights[1], 1 / 3)

    def test_sample(self):
        # Test case I: Just one sample
        times = [1, 2, 3, 4, 5]
        individual = 'ID 1'
        samples = self.stacked_model.sample(times, individual=individual)

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

        # Test case II: More than one sample
        seed = 1
        n_samples = 4
        samples = self.stacked_model.sample(
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

        # Test case III: include dosing regimen
        # Test case III.1: First model is PD model
        samples = self.stacked_model.sample(times, include_regimen=True)

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

        # Test case III.2: PK model, regimen not set
        model = chi.StackedPredictiveModel(
            [self.model_2, self.model_2], weights=self.weights)

        # Sample
        samples = model.sample(times, include_regimen=True)

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

        # Test case III.3: PK model, regimen set
        model.set_dosing_regimen(1, 1, duration=2, period=2, num=2)

        # Sample
        samples = model.sample(times, include_regimen=True)

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
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Sample'].dropna().unique()
        self.assertEqual(len(values), 5)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 2)


if __name__ == '__main__':
    unittest.main()
