#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

import erlotinib as erlo


class TestCovariateModel(unittest.TestCase):
    """
    Tests the erlotinib.CovariateModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.cov_model = erlo.CovariateModel()

    def test_check_compatibility(self):
        pop_model = 'some model'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.check_compatibility(pop_model)

    def test_compute_individual_parameters(self):
        parameters = 'some parameters'
        eta = 'some fluctuations'
        covariates = 'some covariates'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.compute_individual_parameters(
                parameters, eta, covariates
            )

    def test_compute_individual_sensitivities(self):
        parameters = 'some parameters'
        eta = 'some fluctuations'
        covariates = 'some covariates'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.compute_individual_sensitivities(
                parameters, eta, covariates
            )

    def test_compute_population_parameters(self):
        parameters = 'some parameters'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.compute_population_parameters(
                parameters)

    def test_compute_population_sensitivities(self):
        parameters = 'some parameters'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.compute_population_sensitivities(
                parameters)

    def test_get_parameter_names(self):
        names = self.cov_model.get_parameter_names()
        self.assertIsNone(names)

    def test_n_paramaters(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.n_parameters()

    def test_set_parameter_names(self):
        names = 'some names'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.set_parameter_names(names)


if __name__ == '__main__':
    unittest.main()
