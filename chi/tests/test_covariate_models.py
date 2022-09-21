#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

import chi


class TestCovariateModel(unittest.TestCase):
    """
    Tests the chi.CovariateModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.cov_model = chi.CovariateModel()

    def test_bad_init(self):
        # No covariates
        with self.assertRaisesRegex(ValueError, 'The number of covariates'):
            chi.CovariateModel(n_cov=0)

        # Number of covariate names
        names = ['too', 'many']
        with self.assertRaisesRegex(ValueError, 'The number of covariate '):
            chi.CovariateModel(cov_names=names)

    def test_compute_population_parameters(self):
        parameters = 'some parameters'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.compute_population_parameters(
                parameters)

    def test_compute_sensitivities(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.cov_model.compute_sensitivities(
                'some', 'mat', 'ching', 'input')

    def test_get_covariate_names(self):
        names = self.cov_model.get_covariate_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Cov. 1')

        cov_model = chi.CovariateModel(cov_names=['some name'])
        names = cov_model.get_covariate_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'some name')

    def test_get_parameter_names(self):
        names = self.cov_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Param. 1 Cov. 1')

    def test_n_covariates(self):
        n = self.cov_model.n_covariates()
        self.assertEqual(n, 1)

    def test_n_parameters(self):
        n = self.cov_model.n_parameters()
        self.assertEqual(n, 1)

    def test_set_covariate_names(self):
        names = ['some name']
        self.cov_model.set_covariate_names(names)
        names = self.cov_model.get_covariate_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'some name')

        # Reset
        self.cov_model.set_covariate_names()
        names = self.cov_model.get_covariate_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Cov. 1')

        # bad input
        names = ['too', 'many']
        with self.assertRaisesRegex(ValueError, 'Length of names does'):
            self.cov_model.set_covariate_names(names)

    def test_set_parameter_names(self):
        names = ['some name']
        self.cov_model.set_parameter_names(names)
        names = self.cov_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'some name Cov. 1')

        # Reset
        self.cov_model.set_parameter_names()
        names = self.cov_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Param. 1 Cov. 1')

        # bad input
        names = ['too', 'many']
        with self.assertRaisesRegex(ValueError, 'Length of names does'):
            self.cov_model.set_parameter_names(names)

    def test_set_population_parameters(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.cov_model.set_population_parameters('some input')


class TestLinearCovariateModel(unittest.TestCase):
    """
    Tests the LinearCovariateModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.cov_model = chi.LinearCovariateModel()

    def test_compute_population_parameters(self):
        # 1 covariate, 1 dimension, 1 pop param
        n_cov = 1
        n_ids = 3
        n_pop = 1
        parameters = np.ones(n_cov)
        covariates = np.arange(n_ids).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, 1))
        vartheta = self.cov_model.compute_population_parameters(
            parameters, pop_params, covariates)
        self.assertEqual(vartheta.shape, (n_ids, 1, 1))
        self.assertEqual(vartheta[0, 0, 0], 1)
        self.assertEqual(vartheta[1, 0, 0], 2)
        self.assertEqual(vartheta[2, 0, 0], 3)

        # 1 covariate, 1 dimension, 2 pop param
        n_cov = 1
        n_ids = 3
        n_pop = 2
        parameters = np.ones(n_cov) * 2
        covariates = np.arange(n_ids).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, 1)) * 0.1
        vartheta = self.cov_model.compute_population_parameters(
            parameters, pop_params, covariates)
        self.assertEqual(vartheta.shape, (n_ids, n_pop, 1))
        self.assertEqual(vartheta[0, 0, 0], 0.1)
        self.assertEqual(vartheta[1, 0, 0], 2.1)
        self.assertEqual(vartheta[2, 0, 0], 4.1)
        self.assertEqual(vartheta[0, 1, 0], 0.1)
        self.assertEqual(vartheta[1, 1, 0], 0.1)
        self.assertEqual(vartheta[2, 1, 0], 0.1)

        # select both parameters
        self.cov_model.set_population_parameters(indices=[
            [0, 0], [1, 0]])
        parameters = np.ones(n_cov * n_pop) * 2
        covariates = np.arange(n_ids).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, 1)) * 0.1
        vartheta = self.cov_model.compute_population_parameters(
            parameters, pop_params, covariates)
        self.assertEqual(vartheta.shape, (n_ids, n_pop, 1))
        self.assertEqual(vartheta[0, 0, 0], 0.1)
        self.assertEqual(vartheta[1, 0, 0], 2.1)
        self.assertEqual(vartheta[2, 0, 0], 4.1)
        self.assertEqual(vartheta[0, 1, 0], 0.1)
        self.assertEqual(vartheta[1, 1, 0], 2.1)
        self.assertEqual(vartheta[2, 1, 0], 4.1)
        self.cov_model.set_population_parameters(indices=[
            [0, 0]])

        # 2 covariates, 1 dimension, 1 pop param
        n_cov = 2
        n_ids = 3
        n_pop = 1
        cov_model = chi.LinearCovariateModel(n_cov=2)
        parameters = np.ones(n_cov)
        covariates = np.arange(n_ids * n_cov).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, 1)) * 0.1
        vartheta = cov_model.compute_population_parameters(
            parameters, pop_params, covariates)
        self.assertEqual(vartheta.shape, (n_ids, n_pop, 1))
        self.assertEqual(vartheta[0, 0, 0], 1.1)
        self.assertEqual(vartheta[1, 0, 0], 5.1)
        self.assertEqual(vartheta[2, 0, 0], 9.1)

        # 1 covariates, 2 dimension, 1 pop param
        n_cov = 1
        n_ids = 3
        n_pop = 1
        n_dim = 2
        parameters = np.ones(n_cov)
        covariates = np.arange(n_ids * n_cov).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, n_dim)) * 0.1
        vartheta = self.cov_model.compute_population_parameters(
            parameters, pop_params, covariates)
        self.assertEqual(vartheta.shape, (n_ids, n_pop, n_dim))
        self.assertEqual(vartheta[0, 0, 0], 0.1)
        self.assertEqual(vartheta[1, 0, 0], 1.1)
        self.assertEqual(vartheta[2, 0, 0], 2.1)
        self.assertEqual(vartheta[0, 0, 1], 0.1)
        self.assertEqual(vartheta[1, 0, 1], 0.1)
        self.assertEqual(vartheta[2, 0, 1], 0.1)

        # select both parameters
        parameters = np.ones(n_cov * 2)
        self.cov_model.set_population_parameters(indices=[
            [0, 0], [0, 1]])
        vartheta = self.cov_model.compute_population_parameters(
            parameters, pop_params, covariates)
        self.assertEqual(vartheta.shape, (n_ids, n_pop, 2))
        self.assertEqual(vartheta[0, 0, 0], 0.1)
        self.assertEqual(vartheta[1, 0, 0], 1.1)
        self.assertEqual(vartheta[2, 0, 0], 2.1)
        self.assertEqual(vartheta[0, 0, 1], 0.1)
        self.assertEqual(vartheta[1, 0, 1], 1.1)
        self.assertEqual(vartheta[2, 0, 1], 2.1)
        self.cov_model.set_population_parameters(indices=[
            [0, 0]])

    def test_compute_sensitivities(self):
        # 1 covariate, 1 dimension, 1 pop param
        n_cov = 1
        n_ids = 3
        n_pop = 1
        n_dim = 1
        parameters = np.ones(n_cov)
        covariates = np.arange(n_ids).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, n_dim))
        dlogp_dvartheta = np.ones((n_ids, n_pop, n_dim))
        dpop, dparams = self.cov_model.compute_sensitivities(
            parameters, pop_params, covariates, dlogp_dvartheta)
        self.assertEqual(dpop.shape, (n_pop * n_dim,))
        self.assertEqual(dparams.shape, (n_cov,))

        # 1 covariate, 1 dimension, 2 pop param
        n_cov = 1
        n_ids = 3
        n_pop = 2
        parameters = np.ones(n_cov) * 2
        covariates = np.arange(n_ids).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, 1)) * 0.1
        dlogp_dvartheta = np.ones((n_ids, n_pop, n_dim))
        dpop, dparams = self.cov_model.compute_sensitivities(
            parameters, pop_params, covariates, dlogp_dvartheta)
        self.assertEqual(dpop.shape, (n_pop * n_dim,))
        self.assertEqual(dparams.shape, (n_cov,))

        # select both parameters
        self.cov_model.set_population_parameters(indices=[
            [0, 0], [1, 0]])
        parameters = np.ones(n_cov * n_pop) * 2
        covariates = np.arange(n_ids).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, 1)) * 0.1
        dpop, dparams = self.cov_model.compute_sensitivities(
            parameters, pop_params, covariates, dlogp_dvartheta)
        self.assertEqual(dpop.shape, (n_pop * n_dim,))
        self.assertEqual(dparams.shape, (n_cov * 2,))
        self.cov_model.set_population_parameters(indices=[
            [0, 0]])

        # 2 covariates, 1 dimension, 1 pop param
        n_cov = 2
        n_ids = 3
        n_pop = 1
        cov_model = chi.LinearCovariateModel(n_cov=2)
        parameters = np.ones(n_cov)
        covariates = np.arange(n_ids * n_cov).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, 1)) * 0.1
        dlogp_dvartheta = np.ones((n_ids, n_pop, n_dim))
        dpop, dparams = cov_model.compute_sensitivities(
            parameters, pop_params, covariates, dlogp_dvartheta)
        self.assertEqual(dpop.shape, (n_pop * n_dim,))
        self.assertEqual(dparams.shape, (n_cov,))

        # 1 covariates, 2 dimension, 1 pop param
        n_cov = 1
        n_ids = 3
        n_pop = 1
        n_dim = 2
        parameters = np.ones(n_cov)
        covariates = np.arange(n_ids * n_cov).reshape(n_ids, n_cov)
        pop_params = np.ones((n_pop, 2)) * 0.1
        dlogp_dvartheta = np.ones((n_ids, n_pop, n_dim))
        dpop, dparams = self.cov_model.compute_sensitivities(
            parameters, pop_params, covariates, dlogp_dvartheta)
        self.assertEqual(dpop.shape, (n_pop * n_dim,))
        self.assertEqual(dparams.shape, (n_cov,))


if __name__ == '__main__':
    unittest.main()
