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

        # Test case III: -Infinity for not allowed regimes

        # Test case III.1: Zero sigma_base
        parameters = [0, 0.1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)

        # Test case III.2: Negative sigma_base
        parameters = [-1, 0.1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)

        # Test case III.3: Zero sigma_rel
        parameters = [1, 0]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertAlmostEqual(score, ref_score)

        # Test case III.4: Negative sigma_rel
        parameters = [1, -1]
        model_output = [1] * 10
        observations = [1] * 10
        ref_score = -np.inf

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
        names = ['Sigma base', 'Sigma rel.']
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


class TestReducedErrorModel(unittest.TestCase):
    """
    Tests the erlotinib.ReducedErrorModel class.
    """

    @classmethod
    def setUpClass(cls):
        error_model = erlo.ConstantAndMultiplicativeGaussianErrorModel()
        cls.error_model = erlo.ReducedErrorModel(error_model)

    def test_compute_log_likelihood(self):
        # Test case I: fix some parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': 0.1})

        # Compute log-likelihood
        parameters = [0.2]
        model_output = [1, 2, 3, 4]
        observations = [2, 3, 4, 5]
        score = self.error_model.compute_log_likelihood(
            parameters, model_output, observations)

        # Compute ref score with original error model
        parameters = [0.1, 0.2]
        error_model = self.error_model.get_error_model()
        ref_score = error_model.compute_log_likelihood(
            parameters, model_output, observations)

        self.assertEqual(score, ref_score)

        # Unfix model parameters
        self.error_model.fix_parameters(name_value_dict={
            'Sigma base': None})

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

    def test_get_error_model(self):
        error_model = self.error_model.get_error_model()
        self.assertIsInstance(error_model, erlo.ErrorModel)

    def test_set_get_parameter_names(self):
        # Set some parameter names
        self.error_model.set_parameter_names({
            'Sigma base': 'Test'})

        names = self.error_model.get_parameter_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Test')
        self.assertEqual(names[1], 'Sigma rel.')

        # Revert to defaults
        self.error_model.set_parameter_names({
            'Test': 'Sigma base'})

        names = self.error_model.get_parameter_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Sigma base')
        self.assertEqual(names[1], 'Sigma rel.')


if __name__ == '__main__':
    unittest.main()
