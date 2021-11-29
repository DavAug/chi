#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

import chi


class TestConstantAndMultiplicativeGaussianErrorModel(unittest.TestCase):
    """
    Tests the chi.ConstantAndMultiplicativeGaussianErrorModel class.
    """
    @classmethod
    def setUpClass(cls):
        cls.error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()

    def test_compute_log_likelihood(self):
        # Tests :meth:`compute_log_likelihood` and
        # :meth:`compute_pointwise_ll`

        # Test case I: If X = X^m, the score reduces to
        # -np.log(2pi)/2 - np.log(sigma_tot)

        # Test case I.1:
        parameters = [1, 0.1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(1 + 0.1 * 1)

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case I.2:
        parameters = [1, 0.1]
        model_output = [10] * 10
        observations = [10] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(1 + 0.1 * 10)

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case II: If sigma_tot = 1, the score reduces to
        # -np.log(2pi)/2 - (X-X^m) / 2

        # Test case II.1:
        parameters = [0.9, 0.1]
        model_output = [1] * 6
        observations = [2] * 6
        ref_score = -3 * np.log(2 * np.pi) - 6 * (1 - 2)**2 / 2

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (6,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case II.2:
        parameters = [0.9, 0.1]
        model_output = [1] * 10
        observations = [10] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 10)**2 / 2

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III: -Infinity for not allowed regimes

        # Test case III.1: Zero sigma_base
        parameters = [0, 0.1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III.2: Negative sigma_base
        parameters = [-1, 0.1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III.3: Zero sigma_rel
        parameters = [1, 0]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III.4: Negative sigma_rel
        parameters = [1, -1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

    def test_compute_log_likelihood_bad_input(self):
        # Model output and observations don't match
        parameters = [1, 0.1]
        model_output = ['some', 'length']
        observations = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_log_likelihood(
                parameters, model_output, observations)
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_pointwise_ll(
                parameters, model_output, observations)

    def test_compute_sensitivities(self):
        # Test case I: If X = X^m, the scores reduce to
        # L = -np.log(2pi)/2 - np.log(sigma_tot)
        # dL/dpsi = -sigma_rel/sigma_tot * dx/dpsi
        # dL/dsigma_base = -1/sigma_tot
        # dL/dsigma_rel = -x/sigma_tot

        # Test case I.1:
        parameters = [1, 0.1]
        model_output = [1] * 10
        observations = [1] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(1 + 0.1 * 1)
        ref_dpsi_0 = -10 * 0.1 / (1 + 0.1) * 1
        ref_dpsi_1 = -10 * 0.1 / (1 + 0.1) * 2
        ref_dsigma_base = -10 / (1 + 0.1)
        ref_dsigma_rel = -10 / (1 + 0.1) * 1

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 4)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)
        self.assertAlmostEqual(sens[3], ref_dsigma_rel)

        # Test case I.2:
        parameters = [1, 0.1]
        model_output = [10] * 10
        observations = [10] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(1 + 0.1 * 10)
        ref_dpsi_0 = -10 * 0.1 / (1 + 1) * 1
        ref_dpsi_1 = -10 * 0.1 / (1 + 1) * 2
        ref_dsigma_base = -10 / (1 + 1)
        ref_dsigma_rel = -10 / (1 + 1) * 10

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 4)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)
        self.assertAlmostEqual(sens[3], ref_dsigma_rel)

        # Test case II: If sigma_tot = 1, the scores reduce to
        # L = -np.log(2pi)/2 - (X-X^m) / 2
        # dL/dpsi = (-sigma_rel + error + error^2*sigma_rel) * dx/dpsi
        # dL/dsigma_base = - 1 + error^2
        # dL/dsigma_rel = - x + x*error^2

        # Test case II.1:
        parameters = [0.9, 0.1]
        model_output = [1] * 10
        observations = [2] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 2)**2 / 2
        ref_dpsi_0 = 10 * (-0.1 + 1 + 1 * 0.1) * 1
        ref_dpsi_1 = 10 * (-0.1 + 1 + 1 * 0.1) * 2
        ref_dsigma_base = -10 + 10 * 1
        ref_dsigma_rel = -10 + 10 * 1 * 1

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 4)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)
        self.assertAlmostEqual(sens[3], ref_dsigma_rel)

        # Test case II.2:
        parameters = [0.9, 0.1]
        model_output = [1] * 10
        observations = [10] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 10)**2 / 2
        ref_dpsi_0 = 10 * (-0.1 + 9 + 9**2 * 0.1) * 1
        ref_dpsi_1 = 10 * (-0.1 + 9 + 9**2 * 0.1) * 2
        ref_dsigma_base = -10 + 10 * 9**2
        ref_dsigma_rel = -10 + 10 * 9**2 * 1

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 4)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)
        self.assertAlmostEqual(sens[3], ref_dsigma_rel)

    def test_compute_sensitivities_bad_input(self):
        # Model output and sensitivities don't match
        parameters = [1, 0.1]
        model_output = ['some', 'length']
        observations = ['some', 'length']
        sens = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The first dimension of'):
            self.error_model.compute_sensitivities(
                parameters, model_output, sens, observations)

    def test_get_parameter_names(self):
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0], 'Sigma base')
        self.assertEqual(parameters[1], 'Sigma rel.')

    def test_n_parameters(self):
        self.assertEqual(self.error_model.n_parameters(), 2)

    def test_sample(self):
        # Test I: sample size 1
        seed = 42
        parameters = [3, 2]
        n_times = 3
        model_output = [1] * n_times
        sample = self.error_model.sample(parameters, model_output, seed=seed)

        n_samples = 1
        self.assertEqual(sample.shape, (n_times, n_samples))
        self.assertAlmostEqual(sample[0, 0], 3.7952806720457217)
        self.assertAlmostEqual(sample[1, 0], -6.022022696029159)
        self.assertAlmostEqual(sample[2, 0], 0.6469945736947356)

        # Test II: sample size > 1
        n_samples = 4
        sample = self.error_model.sample(
            parameters, model_output, n_samples=n_samples, seed=seed)

        self.assertEqual(sample.shape, (n_times, n_samples))
        self.assertAlmostEqual(sample[0, 0], 2.046212634385726)
        self.assertAlmostEqual(sample[1, 0], -4.115603997796511)
        self.assertAlmostEqual(sample[2, 0], 0.5798718003966126)
        self.assertAlmostEqual(sample[0, 1], 0.13453009521457915)
        self.assertAlmostEqual(sample[1, 1], -4.824303722244952)
        self.assertAlmostEqual(sample[2, 1], -2.920990871528623)
        self.assertAlmostEqual(sample[0, 2], 4.186372271923463)
        self.assertAlmostEqual(sample[1, 2], 3.140421812116401)
        self.assertAlmostEqual(sample[2, 2], 6.0832766019365465)
        self.assertAlmostEqual(sample[0, 3], 2.1031092234071647)
        self.assertAlmostEqual(sample[1, 3], -0.04857959900325241)
        self.assertAlmostEqual(sample[2, 3], 3.024316842149241)

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]
        model_output = [1] * 10

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.error_model.sample(parameters, model_output)

    def test_set_parameter_names(self):
        # Set parameter names
        names = ['some', 'names']
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0], 'some')
        self.assertEqual(parameters[1], 'names')

        # Reset parameter names
        names = None
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0], 'Sigma base')
        self.assertEqual(parameters[1], 'Sigma rel.')

    def test_set_parameter_names_bad_input(self):
        # Not the right number of names
        names = ['Too', 'many', 'names']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.error_model.set_parameter_names(names)


class TestErrorModel(unittest.TestCase):
    """
    Tests the chi.ErrorModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.error_model = chi.ErrorModel()

    def test_compute_log_likelihood(self):
        parameters = 'some parameters'
        model_output = 'some output'
        observations = 'some observations'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.error_model.compute_log_likelihood(
                parameters, model_output, observations)

    def test_compute_pointwise_ll(self):
        parameters = 'some parameters'
        model_output = 'some output'
        observations = 'some observations'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.error_model.compute_pointwise_ll(
                parameters, model_output, observations)

    def test_compute_sensitivities(self):
        parameters = 'some parameters'
        model_output = 'some output'
        sens = 'some sensitivities'
        observations = 'some observations'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.error_model.compute_sensitivities(
                parameters, model_output, sens, observations)

    def test_get_parameter_names(self):
        self.assertIsNone(self.error_model.get_parameter_names())

    def test_n_parameters(self):
        self.assertIsNone(self.error_model.n_parameters())

    def test_sample(self):
        parameters = 'some parameters'
        model_output = 'some output'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.error_model.sample(parameters, model_output)

    def test_set_parameter_names(self):
        names = 'some names'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.error_model.set_parameter_names(names)


class TestGaussianErrorModel(unittest.TestCase):
    """
    Tests the chi.GaussianErrorModel class.
    """
    @classmethod
    def setUpClass(cls):
        cls.error_model = chi.GaussianErrorModel()

    def test_compute_log_likelihood(self):
        # Tests :meth:`compute_log_likelihood` and
        # :meth:`compute_pointwise_ll`

        # Test case I: If X = X^m, the score reduces to
        # -np.log(2pi)/2 - np.log(sigma)

        # Test case I.1:
        parameters = [0.5]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(0.5)

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case I.2:
        parameters = [0.5]
        model_output = [10] * 6
        observations = [10] * 6
        ref_score = -3 * np.log(2 * np.pi) - 6 * np.log(0.5)

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (6,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case II: If sigma = 1, the score reduces to
        # -np.log(2pi)/2 - (X-X^m) / 2

        # Test case II.1:
        parameters = [1]
        model_output = [1] * 10
        observations = [2] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 2)**2 / 2

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case II.2:
        parameters = [1]
        model_output = [1] * 10
        observations = [10] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 10)**2 / 2

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III: -Infinity for not allowed regimes

        # Test case III.1: Zero sigma
        parameters = [0]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III.2: Negative sigma
        parameters = [-1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

    def test_compute_log_likelihood_bad_input(self):
        # Model output and observations don't match
        parameters = [1]
        model_output = ['some', 'length']
        observations = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_log_likelihood(
                parameters, model_output, observations)
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_pointwise_ll(
                parameters, model_output, observations)

    def test_compute_sensitivities(self):
        # Test case I: If X = X^m, the scores reduce to
        # L = -np.log(2pi)/2 - np.log(sigma)
        # dL/dpsi = 0
        # dL/dsigma_base = -1/sigma

        # Test case I.1:
        parameters = [0.5]
        model_output = [1] * 10
        observations = [1] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(0.5)
        ref_dpsi_0 = 0
        ref_dpsi_1 = 0
        ref_dsigma_base = -10 / (0.5)

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)

        # Test case I.2:
        parameters = [0.5]
        model_output = [10] * 10
        observations = [10] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(0.5)
        ref_dpsi_0 = 0
        ref_dpsi_1 = 0
        ref_dsigma_base = -10 / (0.5)

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)

        # Test case II: If sigma = 1, the scores reduce to
        # L = -np.log(2pi)/2 - (X-X^m) / 2
        # dL/dpsi = error * dx/dpsi
        # dL/dsigma_base = - 1 + error^2

        # Test case II.1:
        parameters = [1]
        model_output = [1] * 10
        observations = [2] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 2)**2 / 2
        ref_dpsi_0 = 10 * (1) * 1
        ref_dpsi_1 = 10 * (1) * 2
        ref_dsigma_base = -10 + 10 * 1

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)

        # Test case II.2:
        parameters = [1]
        model_output = [1] * 10
        observations = [10] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 10)**2 / 2
        ref_dpsi_0 = 10 * (9) * 1
        ref_dpsi_1 = 10 * (9) * 2
        ref_dsigma_base = -10 + 10 * 9**2

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)

    def test_compute_sensitivities_bad_input(self):
        # Model output and sensitivities don't match
        parameters = [1]
        model_output = ['some', 'length']
        observations = ['some', 'length']
        sens = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The first dimension of'):
            self.error_model.compute_sensitivities(
                parameters, model_output, sens, observations)

    def test_get_parameter_names(self):
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'Sigma')

    def test_n_parameters(self):
        self.assertEqual(self.error_model.n_parameters(), 1)

    def test_sample(self):
        # Test I: sample size 1
        seed = 42
        parameters = [0.5]
        n_times = 3
        model_output = [1] * n_times
        sample = self.error_model.sample(parameters, model_output, seed=seed)

        n_samples = 1
        self.assertEqual(sample.shape, (n_times, n_samples))

        # Test II: sample size > 1
        n_samples = 4
        sample = self.error_model.sample(
            parameters, model_output, n_samples=n_samples, seed=seed)

        self.assertEqual(sample.shape, (n_times, n_samples))

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]
        model_output = [1] * 10

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.error_model.sample(parameters, model_output)

    def test_set_parameter_names(self):
        # Set parameter names
        names = ['some name']
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'some name')

        # Reset parameter names
        names = None
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'Sigma')

    def test_set_parameter_names_bad_input(self):
        # Not the right number of names
        names = ['Too', 'many', 'names']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.error_model.set_parameter_names(names)


class TestLogNormalErrorModel(unittest.TestCase):
    """
    Tests the chi.LogNormalErrorModel class.
    """
    @classmethod
    def setUpClass(cls):
        cls.error_model = chi.LogNormalErrorModel()

    def test_compute_log_likelihood(self):
        # Tests :meth:`compute_log_likelihood` and
        # :meth:`compute_pointwise_ll`

        # Test case I: If log X - log X^m = - sigma^2/2,
        # the score reduces to
        # -np.log(2pi)/2 - np.log(sigma) - np.log(X)

        # Test case I.1:
        parameters = [0.5]
        model_output = [1 * np.exp(parameters[0]**2 / 2)] * 10
        observations = [1] * 10
        ref_score = \
            -5 * np.log(2 * np.pi) - 10 * (np.log(0.5) + np.log(1))

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case I.2:
        parameters = [0.5]
        model_output = [10 * np.exp(parameters[0]**2 / 2)] * 6
        observations = [10] * 6
        ref_score = \
            -3 * np.log(2 * np.pi) - 6 * (np.log(0.5) + np.log(10))

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (6,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case II: If sigma = 1, the score reduces to
        # -np.log(2pi)/2 - log X - (log X - log X^m - 1/2)^2 / 2

        # Test case II.1:
        parameters = [1]
        model_output = [1] * 10
        observations = [2] * 10
        ref_score = \
            -5 * np.log(2 * np.pi) - 10 * np.log(2) \
            - 10 * (np.log(1) - np.log(2) - 1 / 2)**2 / 2

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case II.2:
        parameters = [1]
        model_output = [1] * 10
        observations = [10] * 10
        ref_score = \
            -5 * np.log(2 * np.pi) - 10 * np.log(10) \
            - 10 * (np.log(1) - np.log(10) - 1 / 2)**2 / 2

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III: -Infinity for not allowed regimes

        # Test case III.1: Zero sigma
        parameters = [0]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III.2: Negative sigma
        parameters = [-1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III.3: Zero model output
        parameters = [1]
        model_output = [0] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

    def test_compute_log_likelihood_bad_input(self):
        # Model output and observations don't match
        parameters = [1]
        model_output = ['some', 'length']
        observations = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_log_likelihood(
                parameters, model_output, observations)
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_pointwise_ll(
                parameters, model_output, observations)

    def test_compute_sensitivities(self):
        # Test case I: If If log X - log X^m = - sigma^2/2,
        # the scores reduce to
        # L = -np.log(2pi)/2 - np.log(sigma) - np.log(X)
        # dL/dpsi = 0
        # dL/dsigma = -1/sigma

        # Test case I.1:
        parameters = [0.5]
        model_output = [1 * np.exp(parameters[0]**2 / 2)] * 10
        observations = [1] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = \
            -5 * np.log(2 * np.pi) - 10 * (np.log(0.5) + np.log(1))
        ref_dpsi_0 = 0
        ref_dpsi_1 = 0
        ref_dsigma_base = -10 / (0.5)

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)

        # Test case I.2:
        parameters = [0.5]
        model_output = [10 * np.exp(parameters[0]**2 / 2)] * 10
        observations = [10] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = \
            -5 * np.log(2 * np.pi) - 10 * (np.log(0.5) + np.log(10))
        ref_dpsi_0 = 0
        ref_dpsi_1 = 0
        ref_dsigma_base = -10 / (0.5)

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)

        # Test case II: If sigma = 1, the scores reduce to
        # L = -np.log(2pi)/2 - log X - (log X - log X^m + 1 / 2) / 2
        # dL/dpsi = error * dx/dpsi
        # dL/dsigma_base = - 1 + error^2

        # Test case II.1:
        parameters = [1]
        model_output = [1] * 10
        observations = [2] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = \
            - 5 * np.log(2 * np.pi) - 10 * np.log(2) \
            - 10 * (np.log(1) - np.log(2) - 1 / 2)**2 / 2
        ref_dpsi_0 = 10 * (np.log(2) - np.log(1) + 1 / 2) / 1 * 1
        ref_dpsi_1 = 10 * (np.log(2) - np.log(1) + 1 / 2) / 1 * 2
        ref_dsigma_base = \
            -10 + 10 * (np.log(1) - np.log(2) - 1 / 2)**2 \
            - 10 * (np.log(2) - np.log(1) + 1 / 2)

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)

        # Test case II.2:
        parameters = [1]
        model_output = [1] * 10
        observations = [10] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = \
            - 5 * np.log(2 * np.pi) - 10 * np.log(10) \
            - 10 * (np.log(1) - np.log(10) - 1 / 2)**2 / 2
        ref_dpsi_0 = 10 * (np.log(10) - np.log(1) + 1 / 2) / 1 * 1
        ref_dpsi_1 = 10 * (np.log(10) - np.log(1) + 1 / 2) / 1 * 2
        ref_dsigma_base = \
            -10 + 10 * (np.log(1) - np.log(10) - 1 / 2)**2 \
            - 10 * (np.log(10) - np.log(1) + 1 / 2)

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_base)

    def test_compute_sensitivities_bad_input(self):
        # Model output and sensitivities don't match
        parameters = [1]
        model_output = ['some', 'length']
        observations = ['some', 'length']
        sens = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The first dimension of'):
            self.error_model.compute_sensitivities(
                parameters, model_output, sens, observations)

    def test_get_parameter_names(self):
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'Sigma log')

    def test_n_parameters(self):
        self.assertEqual(self.error_model.n_parameters(), 1)

    def test_sample(self):
        # Test I: sample size 1
        seed = 42
        parameters = [0.5]
        n_times = 3
        model_output = [1] * n_times
        sample = self.error_model.sample(parameters, model_output, seed=seed)

        n_samples = 1
        self.assertEqual(sample.shape, (n_times, n_samples))

        # Test II: sample size > 1
        n_samples = 4
        sample = self.error_model.sample(
            parameters, model_output, n_samples=n_samples, seed=seed)

        self.assertEqual(sample.shape, (n_times, n_samples))

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]
        model_output = [1] * 10

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.error_model.sample(parameters, model_output)

    def test_set_parameter_names(self):
        # Set parameter names
        names = ['some name']
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'some name')

        # Reset parameter names
        names = None
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'Sigma log')

    def test_set_parameter_names_bad_input(self):
        # Not the right number of names
        names = ['Too', 'many', 'names']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.error_model.set_parameter_names(names)


class TestMultiplicativeGaussianErrorModel(unittest.TestCase):
    """
    Tests the erlotinib.MultiplicativeGaussianErrorModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.error_model = chi.MultiplicativeGaussianErrorModel()

    def test_compute_log_likelihood(self):
        # Tests :meth:`compute_log_likelihood` and
        # :meth:`compute_pointwise_ll`

        # Test case I: If X = X^m, the score reduces to
        # -np.log(2pi)/2 - np.log(sigma_tot)

        # Test case I.1:
        parameters = [0.1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(0.1 * 1)

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case I.2:
        parameters = [0.1]
        model_output = [10] * 6
        observations = [10] * 6
        ref_score = -3 * np.log(2 * np.pi) - 6 * np.log(0.1 * 10)

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (6,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case II: If sigma_tot = 1, the score reduces to
        # -np.log(2pi)/2 - (X-X^m) / 2

        # Test case II.1:
        parameters = [1]
        model_output = [1] * 10
        observations = [2] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 2)**2 / 2

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case II.2:
        parameters = [0.1]
        model_output = [10] * 10
        observations = [100] * 10
        ref_score = -5 * np.log(2 * np.pi) - 10 * (10 - 100)**2 / 2

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III: -Infinity for not allowed regimes

        # Test case III.1: Zero sigma_rel
        parameters = [0]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

        # Test case III.2: Negative sigma_rel
        parameters = [-1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (10,))
        self.assertAlmostEqual(np.sum(pw_score), score)

    def test_compute_log_likelihood_bad_input(self):
        # Model output and observations don't match
        parameters = [0.1]
        model_output = ['some', 'length']
        observations = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_log_likelihood(
                parameters, model_output, observations)
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_pointwise_ll(
                parameters, model_output, observations)

    def test_compute_sensitivities(self):
        # Test case I: If X = X^m, the scores reduce to
        # L = -np.log(2pi)/2 - np.log(sigma_tot)
        # dL/dpsi = -sigma_rel/sigma_tot * dx/dpsi
        # dL/dsigma_base = -1/sigma_tot
        # dL/dsigma_rel = -x/sigma_tot

        # Test case I.1:
        parameters = [0.1]
        model_output = [1] * 10
        observations = [1] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(0.1 * 1)
        ref_dpsi_0 = -10 * 0.1 / (0.1) * 1
        ref_dpsi_1 = -10 * 0.1 / (0.1) * 2
        ref_dsigma_rel = -10 / (0.1) * 1

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_rel)

        # Test case I.2:
        parameters = [0.1]
        model_output = [10] * 10
        observations = [10] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * np.log(0.1 * 10)
        ref_dpsi_0 = -10 * 0.1 / (1) * 1
        ref_dpsi_1 = -10 * 0.1 / (1) * 2
        ref_dsigma_rel = -10 / (1) * 10

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_rel)

        # Test case II: If sigma_tot = 1, the scores reduce to
        # L = -np.log(2pi)/2 - (X-X^m) / 2
        # dL/dpsi = (-sigma_rel + error + error^2*sigma_rel) * dx/dpsi
        # dL/dsigma_base = - 1 + error^2
        # dL/dsigma_rel = - x + x*error^2

        # Test case II.1:
        parameters = [1]
        model_output = [1] * 10
        observations = [2] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 2)**2 / 2
        ref_dpsi_0 = 10 * (-1 + 1 + 1 * 1) * 1
        ref_dpsi_1 = 10 * (-1 + 1 + 1 * 1) * 2
        ref_dsigma_rel = -10 + 10 * 1 * 1

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_rel)

        # Test case II.2:
        parameters = [1]
        model_output = [1] * 10
        observations = [10] * 10
        sens = np.array([[1] * 10, [2] * 10]).T
        ref_score = -5 * np.log(2 * np.pi) - 10 * (1 - 10)**2 / 2
        ref_dpsi_0 = 10 * (-1 + 9 + 9**2 * 1) * 1
        ref_dpsi_1 = 10 * (-1 + 9 + 9**2 * 1) * 2
        ref_dsigma_rel = -10 + 10 * 9**2 * 1

        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, sens, observations)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertAlmostEqual(sens[0], ref_dpsi_0)
        self.assertAlmostEqual(sens[1], ref_dpsi_1)
        self.assertAlmostEqual(sens[2], ref_dsigma_rel)

    def test_compute_sensitivities_bad_input(self):
        # Model output and sensitivities don't match
        parameters = [0.1]
        model_output = ['some', 'length']
        observations = ['some', 'length']
        sens = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The first dimension of'):
            self.error_model.compute_sensitivities(
                parameters, model_output, sens, observations)

    def test_get_parameter_names(self):
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'Sigma rel.')

    def test_n_parameters(self):
        self.assertEqual(self.error_model.n_parameters(), 1)

    def test_sample(self):
        # Test I: sample size 1
        seed = 42
        parameters = [2]
        n_times = 3
        model_output = [1, 10, 100]
        sample = self.error_model.sample(parameters, model_output, seed=seed)

        n_samples = 1
        self.assertEqual(sample.shape, (n_times, n_samples))
        self.assertAlmostEqual(sample[0, 0], 1.6094341595088628)
        self.assertAlmostEqual(sample[1, 0], -10.799682124809912)
        self.assertAlmostEqual(sample[2, 0], 250.09023916129144)

        # Test II: sample size > 1
        n_samples = 4
        sample = self.error_model.sample(
            parameters, model_output, n_samples=n_samples, seed=seed)

        self.assertEqual(sample.shape, (n_times, n_samples))
        self.assertAlmostEqual(sample[0, 0], 1.6094341595088628)
        self.assertAlmostEqual(sample[1, 0], -29.020703773076725)
        self.assertAlmostEqual(sample[2, 0], 96.63976849914224)
        self.assertAlmostEqual(sample[0, 1], -1.079968212480991)
        self.assertAlmostEqual(sample[1, 1], -16.043590137246362)
        self.assertAlmostEqual(sample[2, 1], -70.608785514716)
        self.assertAlmostEqual(sample[0, 2], 2.5009023916129145)
        self.assertAlmostEqual(sample[1, 2], 12.556808063345708)
        self.assertAlmostEqual(sample[2, 2], 275.8795949725657)
        self.assertAlmostEqual(sample[0, 3], 2.8811294327824277)
        self.assertAlmostEqual(sample[1, 3], 3.6751481531283563)
        self.assertAlmostEqual(sample[2, 3], 255.55838708578966)

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]
        model_output = [1] * 10

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.error_model.sample(parameters, model_output)

    def test_set_parameter_names(self):
        # Set parameter names
        names = ['some name']
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'some name')

        # Reset parameter names
        names = None
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 1)
        self.assertEqual(parameters[0], 'Sigma rel.')

    def test_set_parameter_names_bad_input(self):
        # Not the right number of names
        names = ['Too', 'many', 'names']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.error_model.set_parameter_names(names)


class TestReducedErrorModel(unittest.TestCase):
    """
    Tests the erlotinib.ReducedErrorModel class.
    """

    @classmethod
    def setUpClass(cls):
        error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()
        cls.error_model = chi.ReducedErrorModel(error_model)

    def test_bad_instantiation(self):
        model = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The error model'):
            chi.ReducedErrorModel(model)

    def test_compute_log_likelihood(self):
        # Tests :meth:`compute_log_likelihood` and
        # :meth:`compute_pointwise_ll`

        # Test case I: fix some parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': 0.1})

        # Compute log-likelihood
        parameters = [0.2]
        model_output = [1, 2, 3, 4]
        observations = [2, 3, 4, 5]
        pw_score = self.error_model.compute_pointwise_ll(
            parameters, model_output, observations)
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        # Compute ref score with original error model
        parameters = [0.1, 0.2]
        error_model = self.error_model.get_error_model()
        ref_score = error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertEqual(score, ref_score)
        self.assertEqual(pw_score.shape, (4,))
        self.assertAlmostEqual(np.sum(pw_score), ref_score)

        # Unfix model parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': None})

    def test_compute_sensitivities(self):
        # Test case I: fix some parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': 0.1})

        # Compute log-likelihood and sensitivities
        parameters = [0.2]
        model_output = [1, 2, 3, 4]
        observations = [2, 3, 4, 5]
        m_sens = np.array([[1, 2, 3, 4], [1, 1, 1, 1]]).T
        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, m_sens, observations)

        # Compute ref score with original error model
        parameters = [0.1, 0.2]
        error_model = self.error_model.get_error_model()
        ref_score, ref_sens = error_model.compute_sensitivities(
            parameters, model_output, m_sens, observations)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 3)
        self.assertEqual(len(ref_sens), 4)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[3])

        # Unfix model parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': None})

        # Compute log-likelihood and sensitivities
        parameters = [0.1, 0.2]
        score, sens = self.error_model.compute_sensitivities(
            parameters, model_output, m_sens, observations)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 4)
        self.assertEqual(len(ref_sens), 4)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma rel.': 1})

        n_parameters = self.error_model.n_parameters()
        self.assertEqual(n_parameters, 1)

        parameter_names = self.error_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 1)
        self.assertEqual(parameter_names[0], 'Sigma base')

        # Test case II: fix overlapping set of parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': 0.2,
            'Sigma rel.': 0.1})

        n_parameters = self.error_model.n_parameters()
        self.assertEqual(n_parameters, 0)

        parameter_names = self.error_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 0)

        # Test case III: unfix all parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': None,
            'Sigma rel.': None})

        n_parameters = self.error_model.n_parameters()
        self.assertEqual(n_parameters, 2)

        parameter_names = self.error_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 2)
        self.assertEqual(parameter_names[0], 'Sigma base')
        self.assertEqual(parameter_names[1], 'Sigma rel.')

    def test_fix_parameters_bad_input(self):
        name_value_dict = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The name-value dictionary'):
            self.error_model.fix_parameters(name_value_dict)

    def test_get_error_model(self):
        error_model = self.error_model.get_error_model()
        self.assertIsInstance(error_model, chi.ErrorModel)

    def test_n_fixed_parameters(self):
        # Test case I: fix some parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': 0.1})

        self.assertEqual(self.error_model.n_fixed_parameters(), 1)

        # Unfix all parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': None})

        self.assertEqual(self.error_model.n_fixed_parameters(), 0)

    def test_n_parameters(self):
        n_parameters = self.error_model.n_parameters()
        self.assertEqual(n_parameters, 2)

    def test_sample(self):
        # Test case I: fix some parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': 0.1})

        # Sample
        seed = 42
        n_samples = 1
        parameters = [0.2]
        model_output = [1, 2, 3, 4]
        samples = self.error_model.sample(
            parameters, model_output, n_samples, seed)

        # Compute ref score with original error model
        parameters = [0.1, 0.2]
        error_model = self.error_model.get_error_model()
        ref_samples = error_model.sample(
            parameters, model_output, n_samples, seed)

        self.assertEqual(samples.shape, (4, 1))
        self.assertEqual(ref_samples.shape, (4, 1))
        self.assertEqual(samples[0, 0], ref_samples[0, 0])
        self.assertEqual(samples[1, 0], ref_samples[1, 0])
        self.assertEqual(samples[2, 0], ref_samples[2, 0])
        self.assertEqual(samples[3, 0], ref_samples[3, 0])

        # Unfix model parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': None})

    def test_set_get_parameter_names(self):
        # Set some parameter names
        self.error_model.set_parameter_names(['Test 1', 'Test 2'])

        names = self.error_model.get_parameter_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Test 1')
        self.assertEqual(names[1], 'Test 2')

        # Reset to defaults
        self.error_model.set_parameter_names(None)

        names = self.error_model.get_parameter_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Sigma base')
        self.assertEqual(names[1], 'Sigma rel.')

        # Fix parameter and set parameter name
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': 1})
        self.error_model.set_parameter_names(
            ['myokit.tumour_volume Sigma rel.'])

        names = self.error_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'myokit.tumour_volume Sigma rel.')

        # Reset to defaults
        self.error_model.set_parameter_names(None)

        names = self.error_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Sigma rel.')

        # Unfix model parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': None})

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['Wrong length']
        with self.assertRaisesRegex(ValueError, 'Length of names does not'):
            self.error_model.set_parameter_names(names)

        # A parameter exceeds 50 characters
        names = [
            '0123456789-0123456789-0123456789-0123456789-0123456789-012345678',
            'Sigma base']
        with self.assertRaisesRegex(ValueError, 'Parameter names cannot'):
            self.error_model.set_parameter_names(names)


if __name__ == '__main__':
    unittest.main()
