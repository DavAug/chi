#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import pandas as pd
import pints

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

    def test_get_n_outputs(self):
        self.assertEqual(self.model.get_n_outputs(), 1)

    def test_get_output_names(self):
        outputs = self.model.get_output_names()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'myokit.tumour_volume')

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


if __name__ == '__main__':
    unittest.main()
