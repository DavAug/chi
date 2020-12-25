#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import erlotinib as erlo


class TestConstantAndMultiplicativeGaussianErrorModel(unittest.TestCase):
    """
    Tests the erlo.ConstantAndMultiplicativeGaussianErrorModel class.
    """
    @classmethod
    def setUpClass(cls):
        cls.error_model = erlo.ConstantAndMultiplicativeGaussianErrorModel()

    def test_compute_log_likelihood(self):
        # TODO:
        pass

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

    # def test_set_parameter_names(self):
    #     names = 'some names'
    #     with self.assertRaisesRegex(NotImplementedError, ''):
    #         self.error_model.set_parameter_names(names)


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
