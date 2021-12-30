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

    def test_get_covariate_names(self):
        names = self.cov_model.get_covariate_names()
        self.assertIsNone(names)

    def test_get_parameter_names(self):
        names = self.cov_model.get_parameter_names()
        self.assertIsNone(names)

    def test_n_covariates(self):
        n = self.cov_model.n_covariates()
        self.assertIsNone(n)

    def test_n_parameters(self):
        n = self.cov_model.n_parameters()
        self.assertIsNone(n)

    def test_set_covariate_names(self):
        names = 'some names'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.set_covariate_names(names)

    def test_set_parameter_names(self):
        names = 'some names'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.cov_model.set_parameter_names(names)


class TestLogNormalLinearCovariateModel(unittest.TestCase):
    """
    Tests the chi.LogNormalLinearCovariateModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.cov_model = chi.LogNormalLinearCovariateModel(n_covariates=0)
        cls.cov_model2 = chi.LogNormalLinearCovariateModel(n_covariates=2)

    def test_check_compatibility_fail(self):
        # Check that warning is raised with a population model
        # that is not a GaussianModel
        pop_model = chi.LogNormalModel()
        with self.assertWarns(UserWarning):
            self.cov_model.check_compatibility(pop_model)

    @unittest.expectedFailure
    def test_check_compatibility_pass(self):
        # Check that warning is not raised with a GaussianModel
        pop_model = chi.GaussianModel()
        with self.assertWarns(UserWarning):
            self.cov_model.check_compatibility(pop_model)

    def test_compute_individual_parameters(self):
        n_ids = 5

        # Test case I: sigma almost 0
        # Then psi = np.exp(mu)

        # Test case I.1
        parameters = [1, 1E-10]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis
        psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp([parameters[0]] * n_ids)

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        # Test case I.2
        shifts = [-1]
        parameters = [1, 1E-10] + shifts
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = np.ones(shape=(n_ids, 1))

        # Compute psis
        psis = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(
            np.array([parameters[0]] * n_ids) + covariates @ np.array(shifts))

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        # Test case I.3
        parameters = [1, 1E-10]
        eta = np.linspace(0.5, 10, n_ids)
        covariates = 'some covariates'

        # Compute psis
        psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp([parameters[0]] * n_ids)

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        # Test case I.4
        shifts = [5]
        parameters = [1, 1E-15] + shifts
        eta = np.linspace(0.5, 10, n_ids)
        covariates = np.ones(shape=(n_ids, 1)) * 2

        # Compute psis
        psis = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(
            np.array([parameters[0]] * n_ids) + covariates @ np.array(shifts))

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        # Test case II: mu = 0, sigma != 0
        # Then psi = np.exp(mu)

        # Test case II.1
        parameters = [0, 1]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis
        psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(parameters[1] * eta)

        self.assertEqual(len(psis), n_ids)
        self.assertEqual(psis[0], ref_psis[0])
        self.assertEqual(psis[1], ref_psis[1])
        self.assertEqual(psis[2], ref_psis[2])
        self.assertEqual(psis[3], ref_psis[3])
        self.assertEqual(psis[4], ref_psis[4])

        # Test case II.2
        shifts = [-1]
        parameters = [0, 1] + shifts
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = np.ones(shape=(n_ids, 1))

        # Compute psis
        psis = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(
            parameters[1] * eta + covariates @ np.array(shifts))

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        # Test case II.3
        parameters = [0, 0.1]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis
        psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(parameters[1] * eta)

        self.assertEqual(len(psis), n_ids)
        self.assertEqual(psis[0], ref_psis[0])
        self.assertEqual(psis[1], ref_psis[1])
        self.assertEqual(psis[2], ref_psis[2])
        self.assertEqual(psis[3], ref_psis[3])
        self.assertEqual(psis[4], ref_psis[4])

        # Test case II.4
        shifts = [5]
        parameters = [0, 0.1] + shifts
        eta = np.linspace(0.5, 10, n_ids)
        covariates = np.ones(shape=(n_ids, 1)) * 2

        # Compute psis
        psis = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(
            parameters[1] * eta + covariates @ np.array(shifts))

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        # Test case III: mu != 0, sigma != 0
        # Then psi = np.exp(mu)

        # Test case III.1
        parameters = [-1, 1]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis
        psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(parameters[0] + parameters[1] * eta)

        self.assertEqual(len(psis), n_ids)
        self.assertEqual(psis[0], ref_psis[0])
        self.assertEqual(psis[1], ref_psis[1])
        self.assertEqual(psis[2], ref_psis[2])
        self.assertEqual(psis[3], ref_psis[3])
        self.assertEqual(psis[4], ref_psis[4])

        # Test case III.2
        shifts = [-1]
        parameters = [-1, 1] + shifts
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = np.ones(shape=(n_ids, 1))

        # Compute psis
        psis = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        # Test case III.3
        parameters = [2, 0.1]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis
        psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(parameters[0] + parameters[1] * eta)

        self.assertEqual(len(psis), n_ids)
        self.assertEqual(psis[0], ref_psis[0])
        self.assertEqual(psis[1], ref_psis[1])
        self.assertEqual(psis[2], ref_psis[2])
        self.assertEqual(psis[3], ref_psis[3])
        self.assertEqual(psis[4], ref_psis[4])

        # Test case III.4
        shifts = [5]
        parameters = [2, 0.1] + shifts
        eta = np.linspace(0.5, 10, n_ids)
        covariates = np.ones(shape=(n_ids, 1)) * 2

        # Compute psis
        psis = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        ref_psis = np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        # Test case IV: sigma = 0
        parameters = [2, 0]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis
        psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)

        self.assertEqual(len(psis), n_ids)
        self.assertTrue(np.isnan(psis[0]))
        self.assertTrue(np.isnan(psis[1]))
        self.assertTrue(np.isnan(psis[2]))
        self.assertTrue(np.isnan(psis[3]))
        self.assertTrue(np.isnan(psis[4]))

    def test_compute_individual_sensitivities(self):
        n_ids = 5

        # Test case I: mu != 0, sigma != 0
        # Then psi = np.exp(mu)

        # Test case I.1
        parameters = [-1, 1]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis and sensitivities
        psis, sens = self.cov_model.compute_individual_sensitivities(
            parameters, eta, covariates)
        ref_psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)
        ref_dmu = np.exp(parameters[0] + parameters[1] * eta)
        ref_dsigma = eta * np.exp(parameters[0] + parameters[1] * eta)
        ref_detas = parameters[1] * np.exp(parameters[0] + parameters[1] * eta)

        self.assertEqual(len(psis), n_ids)
        self.assertEqual(psis[0], ref_psis[0])
        self.assertEqual(psis[1], ref_psis[1])
        self.assertEqual(psis[2], ref_psis[2])
        self.assertEqual(psis[3], ref_psis[3])
        self.assertEqual(psis[4], ref_psis[4])

        self.assertEqual(sens.shape, (3, n_ids))
        self.assertEqual(sens[1, 0], ref_dmu[0])
        self.assertEqual(sens[1, 1], ref_dmu[1])
        self.assertEqual(sens[1, 2], ref_dmu[2])
        self.assertEqual(sens[1, 3], ref_dmu[3])
        self.assertEqual(sens[1, 4], ref_dmu[4])
        self.assertEqual(sens[2, 0], ref_dsigma[0])
        self.assertEqual(sens[2, 1], ref_dsigma[1])
        self.assertEqual(sens[2, 2], ref_dsigma[2])
        self.assertEqual(sens[2, 3], ref_dsigma[3])
        self.assertEqual(sens[2, 4], ref_dsigma[4])
        self.assertEqual(sens[0, 0], ref_detas[0])
        self.assertEqual(sens[0, 1], ref_detas[1])
        self.assertEqual(sens[0, 2], ref_detas[2])
        self.assertEqual(sens[0, 3], ref_detas[3])
        self.assertEqual(sens[0, 4], ref_detas[4])

        # Test case I.2
        shifts = [1, 0]
        parameters = [-1, 1] + shifts
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = np.ones(shape=(n_ids, 2))

        # Compute psis and sensitivities
        psis, sens = self.cov_model2.compute_individual_sensitivities(
            parameters, eta, covariates)
        ref_psis = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        ref_dmu = np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )
        ref_dsigma = eta * np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )
        ref_detas = parameters[1] * np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )
        ref_dshift0 = covariates[:, 0] * np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )
        ref_dshift1 = covariates[:, 1] * np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )

        self.assertEqual(len(psis), n_ids)
        self.assertEqual(psis[0], ref_psis[0])
        self.assertEqual(psis[1], ref_psis[1])
        self.assertEqual(psis[2], ref_psis[2])
        self.assertEqual(psis[3], ref_psis[3])
        self.assertEqual(psis[4], ref_psis[4])

        self.assertEqual(sens.shape, (5, n_ids))
        self.assertEqual(sens[0, 0], ref_detas[0])
        self.assertEqual(sens[0, 1], ref_detas[1])
        self.assertEqual(sens[0, 2], ref_detas[2])
        self.assertEqual(sens[0, 3], ref_detas[3])
        self.assertEqual(sens[0, 4], ref_detas[4])
        self.assertEqual(sens[1, 0], ref_dmu[0])
        self.assertEqual(sens[1, 1], ref_dmu[1])
        self.assertEqual(sens[1, 2], ref_dmu[2])
        self.assertEqual(sens[1, 3], ref_dmu[3])
        self.assertEqual(sens[1, 4], ref_dmu[4])
        self.assertEqual(sens[2, 0], ref_dsigma[0])
        self.assertEqual(sens[2, 1], ref_dsigma[1])
        self.assertEqual(sens[2, 2], ref_dsigma[2])
        self.assertEqual(sens[2, 3], ref_dsigma[3])
        self.assertEqual(sens[2, 4], ref_dsigma[4])
        self.assertEqual(sens[3, 0], ref_dshift0[0])
        self.assertEqual(sens[3, 1], ref_dshift0[1])
        self.assertEqual(sens[3, 2], ref_dshift0[2])
        self.assertEqual(sens[3, 3], ref_dshift0[3])
        self.assertEqual(sens[3, 4], ref_dshift0[4])
        self.assertEqual(sens[4, 0], ref_dshift1[0])
        self.assertEqual(sens[4, 1], ref_dshift1[1])
        self.assertEqual(sens[4, 2], ref_dshift1[2])
        self.assertEqual(sens[4, 3], ref_dshift1[3])
        self.assertEqual(sens[4, 4], ref_dshift1[4])

        # Test case I.3
        parameters = [2, 0.1]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis and sensitivities
        psis, sens = self.cov_model.compute_individual_sensitivities(
            parameters, eta, covariates)
        ref_psis = self.cov_model.compute_individual_parameters(
            parameters, eta, covariates)
        ref_dmu = np.exp(parameters[0] + parameters[1] * eta)
        ref_dsigma = eta * np.exp(parameters[0] + parameters[1] * eta)
        ref_detas = parameters[1] * np.exp(parameters[0] + parameters[1] * eta)

        self.assertEqual(len(psis), n_ids)
        self.assertEqual(psis[0], ref_psis[0])
        self.assertEqual(psis[1], ref_psis[1])
        self.assertEqual(psis[2], ref_psis[2])
        self.assertEqual(psis[3], ref_psis[3])
        self.assertEqual(psis[4], ref_psis[4])

        self.assertEqual(sens.shape, (3, n_ids))
        self.assertEqual(sens[1, 0], ref_dmu[0])
        self.assertEqual(sens[1, 1], ref_dmu[1])
        self.assertEqual(sens[1, 2], ref_dmu[2])
        self.assertEqual(sens[1, 3], ref_dmu[3])
        self.assertEqual(sens[1, 4], ref_dmu[4])
        self.assertEqual(sens[2, 0], ref_dsigma[0])
        self.assertEqual(sens[2, 1], ref_dsigma[1])
        self.assertEqual(sens[2, 2], ref_dsigma[2])
        self.assertEqual(sens[2, 3], ref_dsigma[3])
        self.assertEqual(sens[2, 4], ref_dsigma[4])
        self.assertEqual(sens[0, 0], ref_detas[0])
        self.assertEqual(sens[0, 1], ref_detas[1])
        self.assertEqual(sens[0, 2], ref_detas[2])
        self.assertEqual(sens[0, 3], ref_detas[3])
        self.assertEqual(sens[0, 4], ref_detas[4])

        # Test case I.4
        shifts = [1, -2]
        parameters = [2, 0.1] + shifts
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = np.reshape(np.arange(n_ids * 2), newshape=(n_ids, 2))

        # Compute psis and sensitivities
        psis, sens = self.cov_model2.compute_individual_sensitivities(
            parameters, eta, covariates)
        ref_psis = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        ref_dmu = np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )
        ref_dsigma = eta * np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )
        ref_detas = parameters[1] * np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )
        ref_dshift0 = covariates[:, 0] * np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )
        ref_dshift1 = covariates[:, 1] * np.exp(
            parameters[0] + parameters[1] * eta +
            covariates @ np.array(shifts)
        )

        self.assertEqual(len(psis), n_ids)
        self.assertAlmostEqual(psis[0], ref_psis[0])
        self.assertAlmostEqual(psis[1], ref_psis[1])
        self.assertAlmostEqual(psis[2], ref_psis[2])
        self.assertAlmostEqual(psis[3], ref_psis[3])
        self.assertAlmostEqual(psis[4], ref_psis[4])

        self.assertEqual(sens.shape, (5, n_ids))
        self.assertAlmostEqual(sens[0, 0], ref_detas[0])
        self.assertAlmostEqual(sens[0, 1], ref_detas[1])
        self.assertAlmostEqual(sens[0, 2], ref_detas[2])
        self.assertAlmostEqual(sens[0, 3], ref_detas[3])
        self.assertAlmostEqual(sens[0, 4], ref_detas[4])
        self.assertAlmostEqual(sens[1, 0], ref_dmu[0])
        self.assertAlmostEqual(sens[1, 1], ref_dmu[1])
        self.assertAlmostEqual(sens[1, 2], ref_dmu[2])
        self.assertAlmostEqual(sens[1, 3], ref_dmu[3])
        self.assertAlmostEqual(sens[1, 4], ref_dmu[4])
        self.assertAlmostEqual(sens[2, 0], ref_dsigma[0])
        self.assertAlmostEqual(sens[2, 1], ref_dsigma[1])
        self.assertAlmostEqual(sens[2, 2], ref_dsigma[2])
        self.assertAlmostEqual(sens[2, 3], ref_dsigma[3])
        self.assertAlmostEqual(sens[2, 4], ref_dsigma[4])
        self.assertAlmostEqual(sens[3, 0], ref_dshift0[0])
        self.assertAlmostEqual(sens[3, 1], ref_dshift0[1])
        self.assertAlmostEqual(sens[3, 2], ref_dshift0[2])
        self.assertAlmostEqual(sens[3, 3], ref_dshift0[3])
        self.assertAlmostEqual(sens[3, 4], ref_dshift0[4])
        self.assertAlmostEqual(sens[4, 0], ref_dshift1[0])
        self.assertAlmostEqual(sens[4, 1], ref_dshift1[1])
        self.assertAlmostEqual(sens[4, 2], ref_dshift1[2])
        self.assertAlmostEqual(sens[4, 3], ref_dshift1[3])
        self.assertAlmostEqual(sens[4, 4], ref_dshift1[4])

        # Test case II: sigma=0
        parameters = [2, 0]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis and sensitivities
        psis, sens = self.cov_model.compute_individual_sensitivities(
            parameters, eta, covariates)

        self.assertEqual(len(psis), n_ids)
        self.assertTrue(np.isnan(psis[0]))
        self.assertTrue(np.isnan(psis[1]))
        self.assertTrue(np.isnan(psis[2]))
        self.assertTrue(np.isnan(psis[3]))
        self.assertTrue(np.isnan(psis[4]))

        self.assertEqual(sens.shape, (3, n_ids))
        self.assertTrue(np.isnan(sens[1, 0]))
        self.assertTrue(np.isnan(sens[1, 1]))
        self.assertTrue(np.isnan(sens[1, 2]))
        self.assertTrue(np.isnan(sens[1, 3]))
        self.assertTrue(np.isnan(sens[1, 4]))
        self.assertTrue(np.isnan(sens[2, 0]))
        self.assertTrue(np.isnan(sens[2, 1]))
        self.assertTrue(np.isnan(sens[2, 2]))
        self.assertTrue(np.isnan(sens[2, 3]))
        self.assertTrue(np.isnan(sens[2, 4]))
        self.assertTrue(np.isnan(sens[0, 0]))
        self.assertTrue(np.isnan(sens[0, 1]))
        self.assertTrue(np.isnan(sens[0, 2]))
        self.assertTrue(np.isnan(sens[0, 3]))
        self.assertTrue(np.isnan(sens[0, 4]))

    def test_compute_population_parameters(self):
        parameters = 'some parameters'
        params = self.cov_model.compute_population_parameters(
            parameters)

        self.assertEqual(len(params), 2)
        self.assertEqual(params[0], 0)
        self.assertEqual(params[1], 1)

    def test_compute_population_sensitivities(self):
        parameters = ['some', 'param', 'eters']
        params, sens = self.cov_model.compute_population_sensitivities(
            parameters)

        self.assertEqual(len(params), 2)
        self.assertEqual(params[0], 0)
        self.assertEqual(params[1], 1)

        self.assertEqual(sens.shape, (3, 2))
        self.assertEqual(sens[0, 0], 0)
        self.assertEqual(sens[1, 0], 0)
        self.assertEqual(sens[2, 0], 0)
        self.assertEqual(sens[0, 1], 0)
        self.assertEqual(sens[1, 1], 0)
        self.assertEqual(sens[2, 1], 0)

    def test_get_covariate_names(self):
        names = self.cov_model.get_covariate_names()
        self.assertEqual(len(names), 0)

    def test_get_parameter_names(self):
        names = self.cov_model.get_parameter_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Base mean log')
        self.assertEqual(names[1], 'Std. log')

    def test_n_covariates(self):
        self.assertEqual(self.cov_model.n_covariates(), 0)

    def test_n_paramaters(self):
        self.assertEqual(self.cov_model.n_parameters(), 2)

    def test_set_covariate_names(self):
        # Test some names and do not update parameter names
        names = ['test', 'names']
        self.cov_model2.set_covariate_names(names)

        n = self.cov_model2.get_covariate_names()
        self.assertEqual(n[0], names[0])
        self.assertEqual(n[1], names[1])

        n = self.cov_model2.get_parameter_names()
        self.assertEqual(n[0], 'Base mean log')
        self.assertEqual(n[1], 'Std. log')
        self.assertEqual(n[2], 'Shift Covariate 1')
        self.assertEqual(n[3], 'Shift Covariate 2')

        # Also update parameter names
        names = ['test', 'names']
        self.cov_model2.set_covariate_names(names, True)

        n = self.cov_model2.get_covariate_names()
        self.assertEqual(n[0], names[0])
        self.assertEqual(n[1], names[1])

        n = self.cov_model2.get_parameter_names()
        self.assertEqual(n[0], 'Base mean log')
        self.assertEqual(n[1], 'Std. log')
        self.assertEqual(n[2], 'Shift test')
        self.assertEqual(n[3], 'Shift names')

        # Set back to default name
        self.cov_model2.set_covariate_names(None, True)
        names = self.cov_model2.get_covariate_names()

        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Covariate 1')
        self.assertEqual(names[1], 'Covariate 2')

    def test_set_covariate_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'one', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.cov_model.set_covariate_names(names)

    def test_set_parameter_names(self):
        # Test some name
        names = ['test', 'names']
        self.cov_model.set_parameter_names(names)

        n = self.cov_model.get_parameter_names()
        self.assertEqual(n[0], names[0])
        self.assertEqual(n[1], names[1])

        # Set back to default name
        self.cov_model.set_parameter_names(None)
        names = self.cov_model.get_parameter_names()

        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Base mean log')
        self.assertEqual(names[1], 'Std. log')

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'one', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.cov_model.set_parameter_names(names)


if __name__ == '__main__':
    unittest.main()
