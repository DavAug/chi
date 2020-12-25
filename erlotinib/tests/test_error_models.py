#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

import erlotinib as erlo


class TestConstantAndMultiplicativeGaussianErrorModel(unittest.TestCase):
    """
    Tests the erlo.ConstantAndMultiplicativeGaussianErrorModel class.
    """
    @classmethod
    def setUpClass(cls):
        cls.error_model = erlo.ConstantAndMultiplicativeGaussianErrorModel()

    def test_compute_log_likelihood(self):
        # Test case I: If X = X^m, the score reduces to
        # - np.log(sigma_tot)

        # Test case I.1:
        parameters = [1, 0.1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -10 * np.log(1 + 0.1 * 1)

        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)

        # Test case I.2:
        parameters = [1, 0.1]
        model_output = [10] * 10
        observations = [10] * 10
        ref_score = -10 * np.log(1 + 0.1 * 10)

        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertEqual(score, ref_score)

        # Test case II: If sigma_tot = 1, the score reduces to
        # -(X-X^m) / 2

        # Test case II.1:
        parameters = [0.9, 0.1]
        model_output = [1] * 10
        observations = [2] * 10
        ref_score = -10 * (1 - 2)**2 / 2

        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)

        # Test case II.2:
        parameters = [0.9, 0.1]
        model_output = [1] * 10
        observations = [10] * 10
        ref_score = -10 * (1 - 10)**2 / 2

        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)

    def test_compute_log_likelihood_bad_input(self):
        # Model output and observations don't match
        parameters = [1, 0.1]
        model_output = ['some', 'length']
        observations = ['some', 'other', 'length']
        with self.assertRaisesRegex(ValueError, 'The number of model outputs'):
            self.error_model.compute_log_likelihood(
                parameters, model_output, observations)

    def test_get_parameter_names(self):
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0], 'Sigma base')
        self.assertEqual(parameters[1], 'Sigma rel.')

    def test_n_parameters(self):
        self.assertEqual(self.error_model.n_parameters(), 2)

    # def test_sample(self):
    #     parameters = 'some parameters'
    #     model_output = 'some output'
    #     with self.assertRaisesRegex(NotImplementedError, ''):
    #         self.error_model.sample(parameters, model_output)

    def test_set_parameter_names(self):
        # Set parameter names
        names = ['some', 'names']
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0], 'some')
        self.assertEqual(parameters[1], 'names')

        # Reset parameter names
        names = ['Sigma base', 'Sigma rel.']
        self.error_model.set_parameter_names(names)
        parameters = self.error_model.get_parameter_names()

        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0], 'Sigma base')
        self.assertEqual(parameters[1], 'Sigma rel.')


class TestErrorModel(unittest.TestCase):
    """
    Tests the erlo.ErrorModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.error_model = erlo.ErrorModel()

    def test_compute_log_likelihood(self):
        parameters = 'some parameters'
        model_output = 'some output'
        observations = 'some observations'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.error_model.compute_log_likelihood(
                parameters, model_output, observations)

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


if __name__ == '__main__':
    unittest.main()
