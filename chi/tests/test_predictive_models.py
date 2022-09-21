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


class TestAveragedPredictiveModel(unittest.TestCase):
    """
    Tests the chi.AveragedPredictiveModel class.

    Since most methods only call methods from the
    chi.PredictiveModel the methods are tested rather superficially.
    """
    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        mechanistic_model = \
            ModelLibrary().tumour_growth_inhibition_model_koch()

        # Define error models
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Create data driven predictive model
        cls.model = chi.AveragedPredictiveModel(
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
        mechanistic_model = \
            ModelLibrary().tumour_growth_inhibition_model_koch()
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

        # Test model III: PopulationPredictive model with covariates
        covariate_pop_model = chi.CovariatePopulationModel(
            chi.GaussianModel(centered=True),
            chi.LinearCovariateModel(n_cov=2))
        covariate_pop_model.set_population_parameters([(0, 0)])
        pop_model = chi.ComposedPopulationModel([
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            covariate_pop_model])
        cls.pred_pop_model2 = chi.PopulationPredictiveModel(
            cls.pred_model, pop_model)

        # Create a posterior samples
        n_chains = 2
        n_draws = 3
        n_ids = 2
        samples = np.ones(shape=(n_chains, n_draws, n_ids))
        pop_samples = xr.DataArray(
            data=samples[:, :, 0],
            dims=['chain', 'draw'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        names = pop_model.get_parameter_names()
        cls.pop_post_samples2 = xr.Dataset({
            name: pop_samples for name in names})

        cls.pop_model2 = chi.PosteriorPredictiveModel(
            cls.pred_pop_model2, cls.pop_post_samples2)

    def test_bad_instantiation(self):
        # Posterior samples have the wrong type
        posterior_samples = 'Bad type'
        with self.assertRaisesRegex(TypeError, 'The posterior samples'):
            chi.PosteriorPredictiveModel(
                self.pred_model, posterior_samples)

        # The dimensions have the wrong names (3 dimensions)
        posterior_samples = self.posterior_samples.copy()
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
        seed = 100
        times = [1, 2, 3, 4, 5]
        samples = self.model.sample(times, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case II: More than one sample
        n_samples = 4
        samples = self.model.sample(times, n_samples=n_samples, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 20)

        # Test case III: include dosing regimen

        # Test case III.1: PD model
        samples = self.model.sample(times, include_regimen=True, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case III.2: PK model, regimen not set
        mechanistic_model = ModelLibrary().one_compartment_pk_model()
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
        samples = model.sample(times, include_regimen=True, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case III.3: PK model, regimen set
        model.set_dosing_regimen(1, 1, duration=2, period=2, num=2)

        # Sample
        samples = model.sample(times, include_regimen=True, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertTrue(np.isnan(sample_ids[1]))

        biomarkers = samples['Observable'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].dropna().unique()
        self.assertEqual(len(values), 5)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 2)

        # Test IV: Pooled model
        pop_model = chi.PooledModel(n_dim=7)
        pred_pop_model = chi.PopulationPredictiveModel(
            self.pred_model, pop_model)
        n_chains = 2
        n_draws = 3
        samples = np.ones(shape=(n_chains, n_draws))
        pop_samples = xr.DataArray(
            data=samples,
            dims=['chain', 'draw'],
            coords={
                'chain': list(range(n_chains)),
                'draw': list(range(n_draws))})
        names = pop_model.get_parameter_names()
        pop_post_samples = xr.Dataset({
            name: pop_samples for name in names})
        pop_model = chi.PosteriorPredictiveModel(
            pred_pop_model, pop_post_samples)

        samples = pop_model.sample(times, include_regimen=True, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test covariate population model
        # Just one sample
        n_samples = 1
        n_cov = 2
        times = [1, 2, 3, 4, 5]
        covariates = \
            np.arange(n_samples * n_cov).reshape(n_samples, n_cov) + 0.1
        samples = self.pop_model2.sample(
            times, covariates=covariates, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Multiple samples
        n_samples = 5
        n_cov = 2
        times = [1, 2, 3, 4, 5]
        covariates = \
            np.arange(n_samples * n_cov).reshape(n_samples, n_cov) + 0.1
        samples = self.pop_model2.sample(
            times, covariates=covariates, n_samples=n_samples, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), n_samples)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)
        self.assertEqual(sample_ids[4], 5)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 25)

    def test_sample_bad_input(self):
        # Individual does not exist
        _id = 'some ID'
        times = [1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, 'The individual <some ID>'):
            self.model.sample(times, individual=_id)


class TestPredictiveModel(unittest.TestCase):
    """
    Tests the chi.PredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        cls.mechanistic_model = \
            ModelLibrary().tumour_growth_inhibition_model_koch()

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
        model = ModelLibrary().one_compartment_pk_model()
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
        mechanistic_model = ModelLibrary().one_compartment_pk_model()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)
        self.assertAlmostEqual(values[0], 0.970159924388273)
        self.assertAlmostEqual(values[1], -0.3837168004345003)
        self.assertAlmostEqual(values[2], 1.3172158091846213)
        self.assertAlmostEqual(values[3], 1.4896478457110898)
        self.assertAlmostEqual(values[4], -1.4664469447762758)

        # Test case III.2: PKmodel, where the dosing regimen is not set
        mechanistic_model = ModelLibrary().one_compartment_pk_model()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].dropna().unique()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)

        biomarkers = samples['Observable'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].dropna().unique()
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


class TestPopulationPredictiveModel(unittest.TestCase):
    """
    Tests the chi.PopulationPredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Test case I: No covariates
        # Get mechanistic and error model
        mechanistic_model = \
            ModelLibrary().tumour_growth_inhibition_model_koch()
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        cls.predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)

        # Create population model
        cls.population_model = chi.ComposedPopulationModel([
            chi.HeterogeneousModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(centered=False),
            chi.PooledModel(),
            chi.PooledModel()])

        # Create predictive population model
        cls.model = chi.PopulationPredictiveModel(
            cls.predictive_model, cls.population_model)

        # Test case II: With covariates
        # Create population model
        covariate_population_model = chi.CovariatePopulationModel(
            chi.GaussianModel(),
            chi.LinearCovariateModel(n_cov=2))
        covariate_population_model.set_population_parameters([(0, 0)])
        population_model = chi.ComposedPopulationModel([
            chi.HeterogeneousModel(),
            chi.LogNormalModel(),
            chi.PooledModel(n_dim=2),
            covariate_population_model,
            chi.PooledModel(n_dim=2)])

        # Create predictive population model
        cls.model2 = chi.PopulationPredictiveModel(
            cls.predictive_model, population_model)

    def test_bad_instantiation(self):
        # Predictive model has wrong type
        predictive_model = 'wrong type'

        with self.assertRaisesRegex(TypeError, 'The predictive model'):
            chi.PopulationPredictiveModel(
                predictive_model, self.population_model)

        # Population model has wrong type
        pop_model = ['wrong type']

        with self.assertRaisesRegex(TypeError, 'The population model has'):
            chi.PopulationPredictiveModel(
                self.predictive_model, pop_model)

        # Wrong dimension of population model
        pop_model = chi.HeterogeneousModel(n_dim=3)
        with self.assertRaisesRegex(ValueError, 'The dimension of the pop'):
            chi.PopulationPredictiveModel(
                self.predictive_model, pop_model)

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        self.model.fix_parameters(name_value_dict={
            'Log std. Dim. 2': 1,
            'ID 1 Dim. 1': 1})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 7)

        names = self.model.get_parameter_names()
        self.assertEqual(len(names), 7)
        self.assertEqual(names[0], 'Log mean Dim. 2')
        self.assertEqual(names[1], 'Pooled Dim. 3')
        self.assertEqual(names[2], 'Pooled Dim. 4')
        self.assertEqual(names[3], 'Log mean Dim. 5')
        self.assertEqual(names[4], 'Log std. Dim. 5')
        self.assertEqual(names[5], 'Pooled Dim. 6')
        self.assertEqual(names[6], 'Pooled Dim. 7')

        # Test case II: fix overlapping set of parameters
        self.model.fix_parameters(name_value_dict={
            'Log std. Dim. 2': None,
            'Pooled Dim. 7': 0.3})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 7)

        names = self.model.get_parameter_names()
        self.assertEqual(len(names), 7)
        self.assertEqual(names[0], 'Log mean Dim. 2')
        self.assertEqual(names[1], 'Log std. Dim. 2')
        self.assertEqual(names[2], 'Pooled Dim. 3')
        self.assertEqual(names[3], 'Pooled Dim. 4')
        self.assertEqual(names[4], 'Log mean Dim. 5')
        self.assertEqual(names[5], 'Log std. Dim. 5')
        self.assertEqual(names[6], 'Pooled Dim. 6')

        # Test case III: unfix all parameters
        self.model.fix_parameters(name_value_dict={
            'ID 1 Dim. 1': None,
            'Pooled Dim. 7': None})

        n_parameters = self.model.n_parameters()
        self.assertEqual(n_parameters, 9)

        names = self.model.get_parameter_names()
        self.assertEqual(len(names), 9)
        self.assertEqual(names[0], 'ID 1 Dim. 1')
        self.assertEqual(names[1], 'Log mean Dim. 2')
        self.assertEqual(names[2], 'Log std. Dim. 2')
        self.assertEqual(names[3], 'Pooled Dim. 3')
        self.assertEqual(names[4], 'Pooled Dim. 4')
        self.assertEqual(names[5], 'Log mean Dim. 5')
        self.assertEqual(names[6], 'Log std. Dim. 5')
        self.assertEqual(names[7], 'Pooled Dim. 6')
        self.assertEqual(names[8], 'Pooled Dim. 7')

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

        self.assertEqual(len(names), 9)
        self.assertEqual(names[0], 'ID 1 Dim. 1')
        self.assertEqual(names[1], 'Log mean Dim. 2')
        self.assertEqual(names[2], 'Log std. Dim. 2')
        self.assertEqual(names[3], 'Pooled Dim. 3')
        self.assertEqual(names[4], 'Pooled Dim. 4')
        self.assertEqual(names[5], 'Log mean Dim. 5')
        self.assertEqual(names[6], 'Log std. Dim. 5')
        self.assertEqual(names[7], 'Pooled Dim. 6')
        self.assertEqual(names[8], 'Pooled Dim. 7')

        # Test case II: Multi-output problem
        model = ModelLibrary().one_compartment_pk_model()
        model.set_administration('central', direct=False)
        model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel(),
            chi.ConstantAndMultiplicativeGaussianErrorModel()]
        model = chi.PredictiveModel(model, error_models)
        pop_model = chi.ComposedPopulationModel(
            self.population_model.get_population_models()
            + [chi.PooledModel(n_dim=2)])
        model = chi.PopulationPredictiveModel(model, pop_model)

        names = model.get_parameter_names()

        self.assertEqual(len(names), 11)
        self.assertEqual(names[0], 'ID 1 Dim. 1')
        self.assertEqual(names[1], 'Log mean Dim. 2')
        self.assertEqual(names[2], 'Log std. Dim. 2')
        self.assertEqual(names[3], 'Pooled Dim. 3')
        self.assertEqual(names[4], 'Pooled Dim. 4')
        self.assertEqual(names[5], 'Log mean Dim. 5')
        self.assertEqual(names[6], 'Log std. Dim. 5')
        self.assertEqual(names[7], 'Pooled Dim. 6')
        self.assertEqual(names[8], 'Pooled Dim. 7')
        self.assertEqual(names[9], 'Pooled Dim. 1')
        self.assertEqual(names[10], 'Pooled Dim. 2')

    def test_get_set_dosing_regimen(self):
        # This just wraps the method from the PredictiveModel. So shallow
        # tests should suffice.j
        ref_dosing_regimen = self.predictive_model.get_dosing_regimen()
        dosing_regimen = self.model.get_dosing_regimen()

        self.assertIsNone(ref_dosing_regimen)
        self.assertIsNone(dosing_regimen)

    def test_n_parameters(self):
        self.assertEqual(self.model.n_parameters(), 9)

    def test_sample(self):
        # Test case I: Just one sample
        parameters = [1, 1, 1, 1, 1, 1, 1, 0.1, 0.1]
        times = [1, 2, 3, 4, 5]
        seed = 42

        # Test case I.1: Return as pd.DataFrame
        samples = self.model.sample(parameters, times, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 20)

        # Test case II.2: Return as numpy.ndarray
        times = [1, 2, 3, 4, 5]
        samples = self.model.sample(
            parameters, times, n_samples=n_samples, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))

        # Test case III: Return dosing regimen

        # Test case III.1: PDModel, dosing regimen is not returned even
        # if flag is True
        times = [1, 2, 3, 4, 5]
        samples = self.model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case III.2: PKmodel, where the dosing regimen is not set
        mechanistic_model = ModelLibrary().one_compartment_pk_model()
        mechanistic_model.set_administration('central', direct=False)
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]
        predictive_model = chi.PredictiveModel(
            mechanistic_model, error_models)
        model = chi.PopulationPredictiveModel(
            predictive_model, self.population_model)

        # Sample
        parameters = [1, 1, 1, 1, 1, 1, 1, 0.1, 0.1]
        times = [1, 2, 3, 4, 5]
        samples = model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case III.3: PKmodel, dosing regimen is set
        model.set_dosing_regimen(1, 1, period=1, num=2)

        # Sample
        times = [1, 2, 3, 4, 5]
        samples = model.sample(
            parameters, times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].dropna().unique()
        self.assertEqual(len(values), 5)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 0.01)

        # Test case III.4: PKmodel, dosing regimen is set, 2 samples
        # Sample
        times = [1, 2, 3, 4, 5]
        samples = model.sample(
            parameters, times, n_samples=2, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)

        biomarkers = samples['Observable'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].dropna().unique()
        self.assertEqual(len(values), 10)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 0.01)

        # Test case V: Covariate model
        # Test case V.1: automatic covariate map
        times = [1, 2, 3, 4, 5]
        parameters = [1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 2, 3]
        covariates = [1.3, 2.4]
        samples = self.model2.sample(
            parameters, times, seed=seed, covariates=covariates)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 3)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')
        self.assertEqual(biomarkers[1], 'Cov. 1')
        self.assertEqual(biomarkers[2], 'Cov. 2')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 6)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)
        self.assertTrue(np.isnan(times[5]))

        values = samples['Value'].unique()
        self.assertEqual(len(values), 7)

    def test_sample_bad_input(self):
        # Parameters are not of length n_parameters
        parameters = ['wrong', 'length']
        times = [1, 2, 3, 4]

        with self.assertRaisesRegex(ValueError, 'The length of parameters'):
            self.model.sample(parameters, times)

        # Raises error when number of covariates and does not match model
        seed = 100
        times = [1, 2, 3, 4, 5]
        parameters = [1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 2, 3]
        covariates = [1.3, 2.4, 1]
        with self.assertRaisesRegex(ValueError, 'Provided covariates do not'):
            self.model2.sample(
                parameters, times, seed=seed, covariates=covariates)

        # Raises error when the covariates per sample do not match n_samples
        n_samples = 3
        covariates = np.ones(shape=(5, 2))
        with self.assertRaisesRegex(ValueError, 'Provided covariates cannot'):
            self.model2.sample(
                parameters, times, seed=seed, covariates=covariates,
                n_samples=n_samples)


class TestPriorPredictiveModel(unittest.TestCase):
    """
    Tests the chi.PriorPredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        mechanistic_model = \
            ModelLibrary().tumour_growth_inhibition_model_koch()

        # Define error models
        error_models = [chi.ConstantAndMultiplicativeGaussianErrorModel()]

        # Test case I:
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

        # Test case II:
        # Create population model
        cls.population_models = chi.ComposedPopulationModel([
            chi.HeterogeneousModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            chi.LogNormalModel(centered=False),
            chi.PooledModel(),
            chi.PooledModel()])

        # Create predictive population model
        cls.pop_predictive_model = chi.PopulationPredictiveModel(
            cls.predictive_model, cls.population_models)

        # Create prior
        cls.pop_log_prior = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(1, 2),
            pints.UniformLogPrior(2, 3),
            pints.UniformLogPrior(3, 4),
            pints.UniformLogPrior(4, 5),
            pints.UniformLogPrior(5, 6),
            pints.UniformLogPrior(6, 7),
            pints.UniformLogPrior(7, 8),
            pints.UniformLogPrior(8, 9),)

        # Create prior predictive model
        cls.prior_pop_pred_model = chi.PriorPredictiveModel(
            cls.pop_predictive_model, cls.pop_log_prior)

        # Test case III:
        # Create population model with covariates
        covariate_population_model = chi.CovariatePopulationModel(
            chi.GaussianModel(),
            chi.LinearCovariateModel(n_cov=2))
        covariate_population_model.set_population_parameters([(0, 0)])
        population_models = chi.ComposedPopulationModel([
            chi.HeterogeneousModel(),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.PooledModel(),
            covariate_population_model,
            chi.PooledModel(),
            chi.PooledModel()])

        # Create predictive population model
        pop_predictive_model2 = chi.PopulationPredictiveModel(
            cls.predictive_model, population_models)

        # Create prior
        pop_log_prior2 = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(1, 2),
            pints.UniformLogPrior(2, 3),
            pints.UniformLogPrior(3, 4),
            pints.UniformLogPrior(4, 5),
            pints.UniformLogPrior(5, 6),
            pints.UniformLogPrior(6, 7),
            pints.UniformLogPrior(7, 8),
            pints.UniformLogPrior(8, 9),
            pints.UniformLogPrior(8, 9),
            pints.UniformLogPrior(8, 9),)

        # Create prior predictive model
        cls.prior_pop_pred_model2 = chi.PriorPredictiveModel(
            pop_predictive_model2, pop_log_prior2)

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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case II: More than one sample
        n_samples = 4
        samples = self.model.sample(
            times, n_samples=n_samples, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 20)

        # Test case III: include dosing regimen

        # Test case III.1: PD model
        samples = self.model.sample(times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case III.2: PK model, regimen not set
        mechanistic_model = ModelLibrary().one_compartment_pk_model()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case III.3: PK model, regimen set
        model.set_dosing_regimen(1, 1, duration=2, period=2, num=2)

        # Sample
        samples = model.sample(times, seed=seed, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertTrue(np.isnan(sample_ids[1]))

        biomarkers = samples['Observable'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'central.drug_concentration')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].dropna().unique()
        self.assertEqual(len(values), 5)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 2)

        # Test case IV: Population model with covariates
        times = [1, 2, 3, 4, 5]
        seed = 42
        samples = self.prior_pop_pred_model.sample(times, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case V: Population model with covariates
        times = [1, 2, 3, 4, 5]
        seed = 42
        covariates = [1, 2]
        samples = self.prior_pop_pred_model2.sample(
            times, seed=seed, covariates=covariates)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)


class TestPAMPredictiveModel(unittest.TestCase):
    """
    Tests the chi.PAMPredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Test model I: Individual predictive model
        # Create predictive model
        mechanistic_model = \
            ModelLibrary().tumour_growth_inhibition_model_koch()
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
        mechanistic_model = \
            ModelLibrary().erlotinib_tumour_growth_inhibition_model()
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
        cls.stacked_model = chi.PAMPredictiveModel(
            predictive_models=[cls.model_1, cls.model_2],
            weights=cls.weights)

    def test_bad_instantiation(self):
        # Models have the wrong type
        models = ['wrong', 'type']
        with self.assertRaisesRegex(TypeError, 'The predictive models'):
            chi.PAMPredictiveModel(
                models, self.weights)

        # The models have a different number of outputs
        mechanistic_model = \
            ModelLibrary().erlotinib_tumour_growth_inhibition_model()
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
            chi.PAMPredictiveModel(
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
            chi.PAMPredictiveModel(
                [self.model_1, model_2], self.weights)

        # The number of models and the number of weights do not
        # coincide
        weights = ['too', 'many', 'weights']
        with self.assertRaisesRegex(ValueError, 'The model weights must be'):
            chi.PAMPredictiveModel(
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
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
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 4)
        self.assertEqual(sample_ids[0], 1)
        self.assertEqual(sample_ids[1], 2)
        self.assertEqual(sample_ids[2], 3)
        self.assertEqual(sample_ids[3], 4)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 20)

        # Test case III: include dosing regimen
        # Test case III.1: First model is PD model
        samples = self.stacked_model.sample(times, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case III.2: PK model, regimen not set
        model = chi.PAMPredictiveModel(
            [self.model_2, self.model_2], weights=self.weights)

        # Sample
        samples = model.sample(times, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 1)
        self.assertEqual(sample_ids[0], 1)

        biomarkers = samples['Observable'].unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].unique()
        self.assertEqual(len(values), 5)

        # Test case III.3: PK model, regimen set
        model.set_dosing_regimen(1, 1, duration=2, period=2, num=2)

        # Sample
        samples = model.sample(times, include_regimen=True)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual(keys[0], 'ID')
        self.assertEqual(keys[1], 'Time')
        self.assertEqual(keys[2], 'Observable')
        self.assertEqual(keys[3], 'Value')
        self.assertEqual(keys[4], 'Duration')
        self.assertEqual(keys[5], 'Dose')

        sample_ids = samples['ID'].unique()
        self.assertEqual(len(sample_ids), 2)
        self.assertEqual(sample_ids[0], 1)
        self.assertTrue(np.isnan(sample_ids[1]))

        biomarkers = samples['Observable'].dropna().unique()
        self.assertEqual(len(biomarkers), 1)
        self.assertEqual(biomarkers[0], 'myokit.tumour_volume')

        times = samples['Time'].dropna().unique()
        self.assertEqual(len(times), 5)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[1], 2)
        self.assertEqual(times[2], 3)
        self.assertEqual(times[3], 4)
        self.assertEqual(times[4], 5)

        values = samples['Value'].dropna().unique()
        self.assertEqual(len(values), 5)

        doses = samples['Dose'].dropna().unique()
        self.assertEqual(len(doses), 1)
        self.assertAlmostEqual(doses[0], 1)

        durations = samples['Duration'].dropna().unique()
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 2)


if __name__ == '__main__':
    unittest.main()
