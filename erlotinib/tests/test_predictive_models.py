#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import pandas as pd

import erlotinib as erlo


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

        with self.assertRaisesRegex(ValueError, 'The provided mechanistic'):
            erlo.PredictiveModel(mechanistic_model, self.error_models)

        # Error model has wrong type
        error_models = ['wrong type']

        with self.assertRaisesRegex(ValueError, 'All provided error models'):
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

    def test_get_parameter_names(self):
        names = self.model.get_parameter_names()

        self.assertEqual(len(names), 7)
        self.assertEqual(names[0], 'myokit.tumour_volume')
        self.assertEqual(names[1], 'myokit.drug_concentration')
        self.assertEqual(names[2], 'myokit.kappa')
        self.assertEqual(names[3], 'myokit.lambda_0')
        self.assertEqual(names[4], 'myokit.lambda_1')
        self.assertEqual(names[5], 'Sigma base')
        self.assertEqual(names[6], 'Sigma rel.')

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
        self.assertEqual(keys[0], 'Sample ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['Sample ID'].unique()
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
        self.assertAlmostEqual(values[0], 0.20509142802317099)
        self.assertAlmostEqual(values[1], -1.0317002595742795)
        self.assertAlmostEqual(values[2], 0.7319423564130202)
        self.assertAlmostEqual(values[3], 0.9396406406230962)
        self.assertAlmostEqual(values[4], -1.9962278249297907)

        # Test case I.2: Return as numpy.ndarray
        samples = self.model.sample(
            parameters, times, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        n_samples = 1
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))
        self.assertAlmostEqual(samples[0, 0, 0], 0.20509142802317099)
        self.assertAlmostEqual(samples[0, 1, 0], -1.0317002595742795)
        self.assertAlmostEqual(samples[0, 2, 0], 0.7319423564130202)
        self.assertAlmostEqual(samples[0, 3, 0], 0.9396406406230962)
        self.assertAlmostEqual(samples[0, 4, 0], -1.9962278249297907)

        # Test case II: More than one sample
        n_samples = 4

        # Test case .1: Return as pd.DataFrame
        samples = self.model.sample(
            parameters, times, n_samples=n_samples, seed=seed)

        self.assertIsInstance(samples, pd.DataFrame)

        keys = samples.keys()
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys[0], 'Sample ID')
        self.assertEqual(keys[1], 'Biomarker')
        self.assertEqual(keys[2], 'Time')
        self.assertEqual(keys[3], 'Sample')

        sample_ids = samples['Sample ID'].unique()
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
        self.assertAlmostEqual(values[0], 0.2905738427032242)
        self.assertAlmostEqual(values[1], -1.092079880507265)
        self.assertAlmostEqual(values[2], 0.8439839821788092)
        self.assertAlmostEqual(values[3], 0.9287421525421682)
        self.assertAlmostEqual(values[4], -1.9787901230389422)
        self.assertAlmostEqual(values[5], -1.3249971784747716)
        self.assertAlmostEqual(values[6], 0.16233315790595912)
        self.assertAlmostEqual(values[7], -0.29256242144870864)
        self.assertAlmostEqual(values[8], 0.007354986561733636)
        self.assertAlmostEqual(values[9], -0.8278291179778853)
        self.assertAlmostEqual(values[10], 1.0047429234609757)
        self.assertAlmostEqual(values[11], 0.7540055434391832)
        self.assertAlmostEqual(values[12], 0.037856978386887785)
        self.assertAlmostEqual(values[13], 1.0824831205839882)
        self.assertAlmostEqual(values[14], 0.50138865431229217)
        self.assertAlmostEqual(values[15], -0.797198173349118)
        self.assertAlmostEqual(values[16], 0.3627140656417333)
        self.assertAlmostEqual(values[17], -1.0033924846121873)
        self.assertAlmostEqual(values[18], 0.834770862895378)
        self.assertAlmostEqual(values[19], -0.015458749010722986)

        # Test case II.2: Return as numpy.ndarray
        samples = self.model.sample(
            parameters, times, n_samples=n_samples, seed=seed, return_df=False)

        n_outputs = 1
        n_times = 5
        self.assertEqual(samples.shape, (n_outputs, n_times, n_samples))
        self.assertAlmostEqual(samples[0, 0, 0], 0.2905738427032242)
        self.assertAlmostEqual(samples[0, 0, 1], -1.092079880507265)
        self.assertAlmostEqual(samples[0, 0, 2], 0.8439839821788092)
        self.assertAlmostEqual(samples[0, 0, 3], 0.9287421525421682)
        self.assertAlmostEqual(samples[0, 1, 0], -1.9787901230389422)
        self.assertAlmostEqual(samples[0, 1, 1], -1.3249971784747716)
        self.assertAlmostEqual(samples[0, 1, 2], 0.16233315790595912)
        self.assertAlmostEqual(samples[0, 1, 3], -0.29256242144870864)
        self.assertAlmostEqual(samples[0, 2, 0], 0.007354986561733636)
        self.assertAlmostEqual(samples[0, 2, 1], -0.8278291179778853)
        self.assertAlmostEqual(samples[0, 2, 2], 1.0047429234609757)
        self.assertAlmostEqual(samples[0, 2, 3], 0.7540055434391832)
        self.assertAlmostEqual(samples[0, 3, 0], 0.037856978386887785)
        self.assertAlmostEqual(samples[0, 3, 1], 1.0824831205839882)
        self.assertAlmostEqual(samples[0, 3, 2], 0.50138865431229217)
        self.assertAlmostEqual(samples[0, 3, 3], -0.797198173349118)
        self.assertAlmostEqual(samples[0, 4, 0], 0.3627140656417333)
        self.assertAlmostEqual(samples[0, 4, 1], -1.0033924846121873)
        self.assertAlmostEqual(samples[0, 4, 2], 0.834770862895378)
        self.assertAlmostEqual(samples[0, 4, 3], -0.015458749010722986)

    def test_sample_bad_input(self):
        # Parameters are not of length n_parameters
        parameters = ['wrong', 'length']
        times = [1, 2, 3, 4]

        with self.assertRaisesRegex(ValueError, 'The length of parameters'):
            self.model.sample(parameters, times)


if __name__ == '__main__':
    unittest.main()
