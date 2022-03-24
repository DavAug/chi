#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
from scipy.stats import norm, truncnorm

import chi


class TestCovariatePopulationModel(unittest.TestCase):
    """
    Tests the chi.CovariatePopulationModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Test case I
        cls.pop_model = chi.GaussianModel()
        cls.cov_model = chi.LogNormalLinearCovariateModel()
        cls.cpop_model = chi.CovariatePopulationModel(
            cls.pop_model, cls.cov_model)

        # Test case II
        cls.cov_model2 = chi.LogNormalLinearCovariateModel(n_covariates=2)
        cls.cpop_model2 = chi.CovariatePopulationModel(
            cls.pop_model, cls.cov_model2)

    def test_bad_instantiation(self):
        # Population model is not a SimplePopulationModel
        pop_model = 'bad type'

        with self.assertRaisesRegex(TypeError, 'The population model'):
            chi.CovariatePopulationModel(
                pop_model,
                chi.LogNormalLinearCovariateModel())

        # Covariate model is not a CovariateModel
        cov_model = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The covariate model'):
            chi.CovariatePopulationModel(
                chi.GaussianModel(),
                cov_model)

    def test_compute_individual_parameters(self):
        # Test case I: Model that is independent of covariates
        # Test case I.1
        parameters = [1, 1]
        eta = [0.2, -0.3, 1, 5]

        ref_psi = self.cov_model.compute_individual_parameters(parameters, eta)
        psi = self.cpop_model.compute_individual_parameters(parameters, eta)
        self.assertEqual(psi[0], ref_psi[0])
        self.assertEqual(psi[1], ref_psi[1])
        self.assertEqual(psi[2], ref_psi[2])
        self.assertEqual(psi[3], ref_psi[3])

        # Test case I.2
        parameters = [0.3, 1E-10]
        eta = [0.2, -0.3, 1, 5]

        psi = self.cpop_model.compute_individual_parameters(parameters, eta)
        self.assertAlmostEqual(psi[0], np.exp(0.3))
        self.assertAlmostEqual(psi[1], np.exp(0.3))
        self.assertAlmostEqual(psi[2], np.exp(0.3))
        self.assertAlmostEqual(psi[3], np.exp(0.3))

        # Test case II: Model that dependends on covariates
        # Test case II.1
        parameters = [1, 1, -1, 1]
        eta = [0.2, -0.3, 1, 5]
        covariates = np.ones(shape=(4, 2))

        ref_psi = self.cov_model2.compute_individual_parameters(
            parameters, eta, covariates)
        psi = self.cpop_model2.compute_individual_parameters(
            parameters, eta, covariates)
        self.assertEqual(psi[0], ref_psi[0])
        self.assertEqual(psi[1], ref_psi[1])
        self.assertEqual(psi[2], ref_psi[2])
        self.assertEqual(psi[3], ref_psi[3])

        # Test case II.2
        parameters = [0.3, 1E-20, 100, -100]
        eta = [0.2, -0.3, 1, 5]
        covariates = np.reshape(np.arange(8), newshape=(4, 2))

        psi = self.cpop_model2.compute_individual_parameters(
            parameters, eta, covariates)
        self.assertAlmostEqual(psi[0], np.exp(0.3 + 100 * 0 - 100 * 1))
        self.assertAlmostEqual(psi[1], np.exp(0.3 + 100 * 2 - 100 * 3))
        self.assertAlmostEqual(psi[2], np.exp(0.3 + 100 * 4 - 100 * 5))
        self.assertAlmostEqual(psi[3], np.exp(0.3 + 100 * 6 - 100 * 7))

    def test_compute_individual_sensitivities(self):
        n_ids = 5

        # Test case I: mu != 0, sigma != 0
        # Then psi = np.exp(mu)

        # Test case I.1
        parameters = [-1, 1]
        eta = np.linspace(0.5, 1.5, n_ids)
        covariates = 'some covariates'

        # Compute psis and sensitivities
        psis, sens = self.cpop_model.compute_individual_sensitivities(
            parameters, eta, covariates)
        ref_psis, ref_sens = self.cov_model.compute_individual_sensitivities(
            parameters, eta, covariates)

        self.assertEqual(len(psis), n_ids)
        self.assertEqual(psis[0], ref_psis[0])
        self.assertEqual(psis[1], ref_psis[1])
        self.assertEqual(psis[2], ref_psis[2])
        self.assertEqual(psis[3], ref_psis[3])
        self.assertEqual(psis[4], ref_psis[4])

        self.assertEqual(sens.shape, (3, n_ids))
        self.assertEqual(sens[0, 0], ref_sens[0, 0])
        self.assertEqual(sens[0, 1], ref_sens[0, 1])
        self.assertEqual(sens[0, 2], ref_sens[0, 2])
        self.assertEqual(sens[0, 3], ref_sens[0, 3])
        self.assertEqual(sens[0, 4], ref_sens[0, 4])
        self.assertEqual(sens[1, 0], ref_sens[1, 0])
        self.assertEqual(sens[1, 1], ref_sens[1, 1])
        self.assertEqual(sens[1, 2], ref_sens[1, 2])
        self.assertEqual(sens[1, 3], ref_sens[1, 3])
        self.assertEqual(sens[1, 4], ref_sens[1, 4])
        self.assertEqual(sens[2, 0], ref_sens[2, 0])
        self.assertEqual(sens[2, 1], ref_sens[2, 1])
        self.assertEqual(sens[2, 2], ref_sens[2, 2])
        self.assertEqual(sens[2, 3], ref_sens[2, 3])
        self.assertEqual(sens[2, 4], ref_sens[2, 4])

    def test_compute_log_likelihood(self):
        n_ids = 10

        # Test case I:
        # Test case I.1:
        etas = [1] * n_ids
        mu_log = 1
        sigma_log = 10

        # Parameters of standard normal (mean=0, std=1)
        ref_score = self.pop_model.compute_log_likelihood([0, 1], etas)

        parameters = [mu_log] + [sigma_log]
        score = self.cpop_model.compute_log_likelihood(parameters, etas)
        self.assertEqual(score, ref_score)

        # Test case I.2:
        etas = [1] * n_ids
        mu_log = 0.1
        sigma_log = 5

        # Parameters of standard normal (mean=0, std=1)
        sigma = 1
        ref_score = -n_ids * (
            np.log(2 * np.pi * sigma**2) / 2 + etas[0]**2 / (2 * sigma**2))

        parameters = [mu_log] + [sigma_log]
        score = self.cpop_model.compute_log_likelihood(parameters, etas)
        self.assertAlmostEqual(score, ref_score)

        # Test case I.3:
        etas = [0.2] * n_ids
        mu_log = 1
        sigma_log = 2

        # Parameters of standard normal (mean=0, std=1)
        ref_score = -n_ids * (
            np.log(2 * np.pi * 1**2) / 2 + etas[0]**2 / (2 * 1**2))

        parameters = [mu_log] + [sigma_log]
        score = self.cpop_model.compute_log_likelihood(parameters, etas)
        self.assertAlmostEqual(score, ref_score)

    def test_compute_pointwise_ll(self):
        # Hard to test exactly, but at least test some edge cases where
        # loglikelihood is straightforward to compute analytically

        n_ids = 10

        # Test case I:
        # Test case I.1:
        etas = [1] * n_ids
        mu_log = 1
        sigma_log = 10

        # Parameters of standard normal (mean=0, std=1)
        ref_score = -n_ids * (
            np.log(2 * np.pi * 1**2) / 2 + etas[0]**2 / (2 * 1**2))

        parameters = [mu_log] + [sigma_log]
        scores = self.cpop_model.compute_pointwise_ll(parameters, etas)
        self.assertEqual(len(scores), 10)
        self.assertAlmostEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 10))

        # Test case I.2:
        etas = [1] * n_ids
        mu_log = 0.1
        sigma_log = 5

        # Parameters of standard normal (mean=0, std=1)
        sigma = 1
        ref_score = -n_ids * (
            np.log(2 * np.pi * sigma**2) / 2 + etas[0]**2 / (2 * sigma**2))

        parameters = [mu_log] + [sigma_log]
        scores = self.cpop_model.compute_pointwise_ll(parameters, etas)
        self.assertEqual(len(scores), 10)
        self.assertAlmostEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 10))

        # Test case I.3:
        etas = [0.2] * n_ids
        mu_log = 1
        sigma_log = 2

        # Parameters of standard normal (mean=0, std=1)
        ref_score = -n_ids * (
            np.log(2 * np.pi * 1**2) / 2 + etas[0]**2 / (2 * 1**2))

        parameters = [mu_log] + [sigma_log]
        scores = self.cpop_model.compute_pointwise_ll(parameters, etas)
        self.assertEqual(len(scores), 10)
        self.assertAlmostEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 10))

    def test_compute_sensitivities(self):
        n_ids = 10

        # Test case I: Non-centered Log-Normal model
        # Sensitivities reduce to
        # deta = -eta
        # dmu_log = 0
        # dsigma_log = 0

        # Test case I.1:
        etas = [1] * n_ids
        mu_log = 1
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.cpop_model.compute_log_likelihood(parameters, etas)
        ref_detas = -1 * np.array(etas)
        ref_dmu = 0
        ref_dsigma = 0

        # Compute log-likelihood and sensitivities
        score, sens = self.cpop_model.compute_sensitivities(parameters, etas)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_detas[0])
        self.assertEqual(sens[1], ref_detas[1])
        self.assertEqual(sens[2], ref_detas[2])
        self.assertEqual(sens[3], ref_detas[3])
        self.assertEqual(sens[4], ref_detas[4])
        self.assertEqual(sens[5], ref_detas[5])
        self.assertEqual(sens[6], ref_detas[6])
        self.assertEqual(sens[7], ref_detas[7])
        self.assertEqual(sens[8], ref_detas[8])
        self.assertEqual(sens[9], ref_detas[9])
        self.assertEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case I.2:
        etas = np.arange(n_ids)
        mu_log = 1
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.cpop_model.compute_log_likelihood(parameters, etas)
        ref_detas = -1 * np.array(etas)
        ref_dmu = 0
        ref_dsigma = 0

        # Compute log-likelihood and sensitivities
        score, sens = self.cpop_model.compute_sensitivities(parameters, etas)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_detas[0])
        self.assertEqual(sens[1], ref_detas[1])
        self.assertEqual(sens[2], ref_detas[2])
        self.assertEqual(sens[3], ref_detas[3])
        self.assertEqual(sens[4], ref_detas[4])
        self.assertEqual(sens[5], ref_detas[5])
        self.assertEqual(sens[6], ref_detas[6])
        self.assertEqual(sens[7], ref_detas[7])
        self.assertEqual(sens[8], ref_detas[8])
        self.assertEqual(sens[9], ref_detas[9])
        self.assertEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case I.3:
        etas = np.arange(n_ids)
        mu_log = -1
        sigma_log = 10

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.cpop_model.compute_log_likelihood(parameters, etas)
        ref_detas = -1 * np.array(etas)
        ref_dmu = 0
        ref_dsigma = 0

        # Compute log-likelihood and sensitivities
        score, sens = self.cpop_model.compute_sensitivities(parameters, etas)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_detas[0])
        self.assertEqual(sens[1], ref_detas[1])
        self.assertEqual(sens[2], ref_detas[2])
        self.assertEqual(sens[3], ref_detas[3])
        self.assertEqual(sens[4], ref_detas[4])
        self.assertEqual(sens[5], ref_detas[5])
        self.assertEqual(sens[6], ref_detas[6])
        self.assertEqual(sens[7], ref_detas[7])
        self.assertEqual(sens[8], ref_detas[8])
        self.assertEqual(sens[9], ref_detas[9])
        self.assertEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case II: Linear covariate model
        etas = np.arange(n_ids)
        mu_log = -1
        sigma_log = 10
        shifts = [1, 2]

        # Compute ref scores
        # (Distribution of eta is independent of model parameters, it's always
        # standard Gaussian. Thus sensitivities of likelihood are zero.)
        parameters = [mu_log] + [sigma_log] + shifts
        ref_ll = self.cpop_model2.compute_log_likelihood(
            parameters, etas)
        ref_detas = -1 * np.array(etas)
        ref_dmu = 0
        ref_dsigma = 0
        ref_dshift0 = 0
        ref_dshift1 = 0

        # Compute log-likelihood and sensitivities
        score, sens = self.cpop_model2.compute_sensitivities(
            parameters, etas)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 4)
        self.assertEqual(sens[0], ref_detas[0])
        self.assertEqual(sens[1], ref_detas[1])
        self.assertEqual(sens[2], ref_detas[2])
        self.assertEqual(sens[3], ref_detas[3])
        self.assertEqual(sens[4], ref_detas[4])
        self.assertEqual(sens[5], ref_detas[5])
        self.assertEqual(sens[6], ref_detas[6])
        self.assertEqual(sens[7], ref_detas[7])
        self.assertEqual(sens[8], ref_detas[8])
        self.assertEqual(sens[9], ref_detas[9])
        self.assertEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)
        self.assertAlmostEqual(sens[12], ref_dshift0)
        self.assertAlmostEqual(sens[13], ref_dshift1)

    def test_get_covariate_model(self):
        cov_model = self.cpop_model.get_covariate_model()
        self.assertIsInstance(cov_model, chi.CovariateModel)

    def test_get_covariate_names(self):
        # Test case I:
        names = []
        self.assertEqual(self.cpop_model.get_covariate_names(), names)

        # Test case II:
        names = ['Covariate 1', 'Covariate 2']
        self.assertEqual(self.cpop_model2.get_covariate_names(), names)

    def test_get_parameter_names(self):
        # Test case I:
        names = ['Base mean log', 'Std. log']
        self.assertEqual(self.cpop_model.get_parameter_names(), names)

        # Test case II:
        names = [
            'Base mean log', 'Std. log', 'Shift Covariate 1',
            'Shift Covariate 2']
        self.assertEqual(self.cpop_model2.get_parameter_names(), names)

    def test_n_hierarchical_parameters(self):
        # Test case I:
        n_ids = 10
        n_hierarchical_params = self.cpop_model.n_hierarchical_parameters(
            n_ids)

        self.assertEqual(len(n_hierarchical_params), 2)
        self.assertEqual(n_hierarchical_params[0], n_ids)
        self.assertEqual(n_hierarchical_params[1], 2)

        # Test case II:
        n_ids = 10
        n_hierarchical_params = self.cpop_model2.n_hierarchical_parameters(
            n_ids)

        self.assertEqual(len(n_hierarchical_params), 2)
        self.assertEqual(n_hierarchical_params[0], n_ids)
        self.assertEqual(n_hierarchical_params[1], 4)

    def test_n_covariates(self):
        # Test case I:
        n_cov = self.cpop_model.n_covariates()
        self.assertEqual(n_cov, 0)

        # Test case II:
        n_cov = self.cpop_model2.n_covariates()
        self.assertEqual(n_cov, 2)

    def test_n_parameters(self):
        self.assertEqual(self.cpop_model.n_parameters(), 2)

    def test_transforms_individual_parameters(self):
        self.assertTrue(self.cpop_model.transforms_individual_parameters())

    def test_sample(self):
        # Test I: sample size 1
        # Test case I.1: return eta
        seed = 42
        parameters = [3, 2]
        sample = self.cpop_model.sample(parameters, seed=seed)

        n_samples = 1
        self.assertEqual(sample.shape, (n_samples,))

        # Test case I.2: return psi
        sample = self.cpop_model.sample(parameters, seed=seed, return_psi=True)
        self.assertEqual(sample.shape, (n_samples,))

        # Test II: sample size > 1
        # Test case II.1: return eta
        parameters = [3, 2]
        n_samples = 4
        sample = self.cpop_model.sample(
            parameters, n_samples=n_samples, seed=seed)

        self.assertEqual(
            sample.shape, (n_samples,))

        # Test case II.2: return psi
        sample = self.cpop_model.sample(
            parameters, n_samples=n_samples, seed=seed, return_psi=True)
        self.assertEqual(sample.shape, (n_samples,))

        # Test III: Model with covariates
        # Test case III.1: return eta
        seed = 42
        parameters = [3, 2, 10, 20]
        covariates = [2, 4]
        sample = self.cpop_model2.sample(
            parameters, covariates=covariates, seed=seed, return_psi=False)

        n_samples = 1
        self.assertEqual(sample.shape, (n_samples,))

        # Test case III.2: return psi
        sample = self.cpop_model2.sample(
            parameters, covariates=covariates, seed=seed, return_psi=True)
        self.assertEqual(sample.shape, (n_samples,))

    def test_sample_bad_input(self):
        # Covariates do not match
        parameters = [3, 2, 10, 20]
        covariates = ['this', 'is', 'the', 'wrong', 'length']
        with self.assertRaisesRegex(ValueError, 'Covariates must be of'):
            self.cpop_model2.sample(parameters, covariates=covariates)

    def test_set_covariate_names(self):
        # Test some name
        names = []
        self.cpop_model.set_covariate_names(names)

        # This covariate model has no covariates
        self.assertEqual(
            self.cpop_model.get_covariate_names(), [])

    def test_set_parameter_names(self):
        # Test some name
        names = ['test', 'name']
        self.cpop_model.set_parameter_names(names)

        self.assertEqual(
            self.cpop_model.get_parameter_names(), names)

        # Set back to default name
        self.cpop_model.set_parameter_names(None)
        names = self.cpop_model.get_parameter_names()

        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Base mean log')
        self.assertEqual(names[1], 'Std. log')


class TestGaussianModel(unittest.TestCase):
    """
    Tests the chi.GaussianModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = chi.GaussianModel()

    def test_compute_log_likelihood(self):
        n_ids = 10

        # Test case I: psis = 1, mu = 1, sigma = 1
        # Score reduces to
        # -nids * np.log(2pi) / 2

        # Test case I.1:
        psis = [1] * n_ids
        mu = 1
        sigma = 1
        ref_score = - n_ids * np.log(2 * np.pi) / 2

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case I.2:
        psis = [5] * n_ids
        mu = 5
        sigma = 1
        ref_score = - n_ids * np.log(2 * np.pi) / 2

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case II: psis != mu, sigma = 1.
        # Score reduces to
        # -nids * (np.log(2pi)/2 + (psi - mu)^2/2)

        # Test case II.1:
        psis = [2] * n_ids
        mu = 1
        sigma = 1
        ref_score = \
            - n_ids * np.log(2 * np.pi) / 2 \
            - n_ids * (psis[0] - mu)**2 / 2

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case II.2:
        psis = [2] * n_ids
        mu = 10
        sigma = 1
        ref_score = \
            - n_ids * np.log(2 * np.pi) / 2 \
            - n_ids * (psis[0] - mu)**2 / 2

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # # Test case III: Any parameters

        # Test case III.1
        psis = np.arange(10)
        mu = 1
        sigma = 1
        ref_score = \
            - n_ids * np.log(2 * np.pi) / 2 \
            - n_ids * np.log(sigma) \
            - np.sum((psis - mu)**2) / (2 * sigma ** 2)

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case III.2
        psis = np.arange(10)
        mu = 10
        sigma = 15
        ref_score = \
            - n_ids * np.log(2 * np.pi) / 2 \
            - n_ids * np.log(sigma) \
            - np.sum((psis - mu)**2) / (2 * sigma ** 2)

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case IV: sigma negative or zero

        # Test case IV.1
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # Test case IV.2
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = -1

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # Test case V: multi-dimensional input
        # Test case V.1: matrix parameters.
        pop_model = chi.GaussianModel(n_dim=2)
        psis = np.arange(10)
        mu = 10
        sigma = 15
        ref_score = \
            - n_ids * np.log(2 * np.pi) / 2 \
            - n_ids * np.log(sigma) \
            - np.sum((psis - mu)**2) / (2 * sigma ** 2)

        psis = np.vstack([psis, psis]).T
        parameters = np.array([[mu, mu], [sigma, sigma]])
        score = pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, 2 * ref_score)

        # Test case V.2: flat parameters.
        parameters = np.array([mu, mu, sigma, sigma])
        score = pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, 2 * ref_score)

    def test_compute_pointwise_ll(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.pop_model.compute_pointwise_ll('some', 'input')

        # TODO: Pointwise likelihoods have been removed for now
        # # Test case I.1:
        # psis = np.arange(10)
        # mu = 1
        # sigma = 1
        # ref_scores = \
        #     - np.log(2 * np.pi) / 2 \
        #     - np.log(sigma) \
        #     - (psis - mu)**2 / (2 * sigma ** 2)

        # parameters = [mu, sigma]
        # pw_scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # score = self.pop_model.compute_log_likelihood(parameters, psis)
        # self.assertEqual(len(pw_scores), 10)
        # self.assertAlmostEqual(np.sum(pw_scores), score)
        # self.assertAlmostEqual(pw_scores[0], ref_scores[0])
        # self.assertAlmostEqual(pw_scores[1], ref_scores[1])
        # self.assertAlmostEqual(pw_scores[2], ref_scores[2])
        # self.assertAlmostEqual(pw_scores[3], ref_scores[3])
        # self.assertAlmostEqual(pw_scores[4], ref_scores[4])
        # self.assertAlmostEqual(pw_scores[5], ref_scores[5])
        # self.assertAlmostEqual(pw_scores[6], ref_scores[6])
        # self.assertAlmostEqual(pw_scores[7], ref_scores[7])
        # self.assertAlmostEqual(pw_scores[8], ref_scores[8])
        # self.assertAlmostEqual(pw_scores[9], ref_scores[9])

        # # Test case I.2:
        # psis = np.linspace(3, 5, 10)
        # mu = 2
        # sigma = 4
        # ref_scores = \
        #     - np.log(2 * np.pi) / 2 \
        #     - np.log(sigma) \
        #     - (psis - mu)**2 / (2 * sigma ** 2)

        # parameters = [mu, sigma]
        # pw_scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # score = self.pop_model.compute_log_likelihood(parameters, psis)
        # self.assertEqual(len(pw_scores), 10)
        # self.assertAlmostEqual(np.sum(pw_scores), score)
        # self.assertAlmostEqual(pw_scores[0], ref_scores[0])
        # self.assertAlmostEqual(pw_scores[1], ref_scores[1])
        # self.assertAlmostEqual(pw_scores[2], ref_scores[2])
        # self.assertAlmostEqual(pw_scores[3], ref_scores[3])
        # self.assertAlmostEqual(pw_scores[4], ref_scores[4])
        # self.assertAlmostEqual(pw_scores[5], ref_scores[5])
        # self.assertAlmostEqual(pw_scores[6], ref_scores[6])
        # self.assertAlmostEqual(pw_scores[7], ref_scores[7])
        # self.assertAlmostEqual(pw_scores[8], ref_scores[8])
        # self.assertAlmostEqual(pw_scores[9], ref_scores[9])

        # # Test case IV: sigma negative or zero

        # # Test case IV.1
        # psis = [np.exp(10)] * 3
        # mu = 1
        # sigma = 0

        # parameters = [mu] + [sigma]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(scores[0], -np.inf)
        # self.assertEqual(scores[1], -np.inf)
        # self.assertEqual(scores[2], -np.inf)

        # # Test case IV.2
        # psis = [np.exp(10)] * 3
        # mu = 1
        # sigma = -10

        # parameters = [mu] + [sigma]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(scores[0], -np.inf)
        # self.assertEqual(scores[1], -np.inf)
        # self.assertEqual(scores[2], -np.inf)

    def test_compute_sensitivities(self):
        n_ids = 10

        # Test case I: psis = mu, sigma = 1
        # Sensitivities reduce to
        # dpsi = 0
        # dmu = 0
        # dsigma = -n_ids

        # Test case I.1:
        psis = [1] * n_ids
        mu = 1
        sigma = 1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = 0
        ref_dmu = 0
        ref_dsigma = -n_ids

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case I.2:
        psis = [10] * n_ids
        mu = 10
        sigma = 1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = 0
        ref_dmu = 0
        ref_dsigma = -n_ids

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case II: psis != mu, sigma = 1
        # Sensitivities reduce to
        # dpsi = mu - psi
        # dmu = psi - mu
        # dsigma = nids * ((psi - mu)^2 - 1)

        # Test case II.1:
        psis = np.array([1] * n_ids)
        mu = 10
        sigma = 1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = mu - psis[0]
        ref_dmu = np.sum(psis - mu)
        ref_dsigma = - n_ids + np.sum((psis - mu)**2)

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case II.2:
        psis = np.array([7] * n_ids)
        mu = 5
        sigma = 1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = mu - psis[0]
        ref_dmu = np.sum(psis - mu)
        ref_dsigma = - n_ids + np.sum((psis - mu)**2)

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case III: psis != mu, sigma != 1
        # Sensitivities reduce to
        # dpsi = (mu - psi) / std**2
        # dmu = sum((psi - mu)) / std**2
        # dsigma = -nids / std  + sum((psi - mu)^2) / std**2

        # Test case III.1:
        psis = np.array([1] * n_ids)
        mu = 10
        sigma = 2

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = (mu - psis[0]) / sigma**2
        ref_dmu = np.sum(psis - mu) / sigma**2
        ref_dsigma = - n_ids / sigma + np.sum((psis - mu)**2) / sigma**3

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma, 5)

        # Test case III.2:
        psis = np.array([7] * n_ids)
        mu = 0.5
        sigma = 0.1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = (mu - psis[0]) / sigma**2
        ref_dmu = np.sum(psis - mu) / sigma**2
        ref_dsigma = - n_ids / sigma + np.sum((psis - mu)**2) / sigma**3

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case IV: Compare gradients to numpy.gradient
        epsilon = 0.001
        n_parameters = n_ids + self.pop_model.n_parameters()
        parameters = np.ones(shape=n_parameters)
        ref_sens = []
        for index in range(n_parameters):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.pop_model.compute_log_likelihood(
                        low[n_ids:], low[:n_ids]),
                    self.pop_model.compute_log_likelihood(
                        parameters[n_ids:], parameters[:n_ids]),
                    self.pop_model.compute_log_likelihood(
                        high[n_ids:], high[:n_ids])],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities with hierarchical model
        _, sens = self.pop_model.compute_sensitivities(
            parameters[n_ids:], parameters[:n_ids])

        self.assertEqual(len(sens), 12)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])
        self.assertEqual(sens[9], ref_sens[9])
        self.assertAlmostEqual(sens[10], ref_sens[10], 5)
        self.assertAlmostEqual(sens[11], ref_sens[11], 5)

        # Test case V: sigma_log negative or zero
        # Test case V.1
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)
        self.assertEqual(score, -np.inf)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)

        # Test case V.2
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = -10

        parameters = [mu] + [sigma]
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)
        self.assertEqual(score, -np.inf)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)

        # Test case VI: Multi-dimensional distribuion
        # Test case VI.1: matrix parameters
        psis = np.array([7] * n_ids)
        mu = 0.5
        sigma = 0.1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = (mu - psis[0]) / sigma**2
        ref_dmu = np.sum(psis - mu) / sigma**2
        ref_dsigma = - n_ids / sigma + np.sum((psis - mu)**2) / sigma**3

        # Compute log-likelihood and sensitivities
        pop_model = chi.GaussianModel(n_dim=2)
        psis = np.vstack([psis, psis]).T
        parameters = np.array([[mu, mu], [sigma, sigma]])
        score, sens = pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, 2 * ref_ll)
        self.assertEqual(len(sens), n_ids * 2 + 2 * 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dpsi)
        self.assertAlmostEqual(sens[11], ref_dpsi)
        self.assertAlmostEqual(sens[12], ref_dpsi)
        self.assertAlmostEqual(sens[13], ref_dpsi)
        self.assertAlmostEqual(sens[14], ref_dpsi)
        self.assertAlmostEqual(sens[15], ref_dpsi)
        self.assertAlmostEqual(sens[16], ref_dpsi)
        self.assertAlmostEqual(sens[17], ref_dpsi)
        self.assertAlmostEqual(sens[18], ref_dpsi)
        self.assertAlmostEqual(sens[19], ref_dpsi)
        self.assertAlmostEqual(sens[20], ref_dmu)
        self.assertAlmostEqual(sens[21], ref_dmu)
        self.assertAlmostEqual(sens[22], ref_dsigma)
        self.assertAlmostEqual(sens[23], ref_dsigma)

        # Test case V.2: flattened parameters
        # Compute log-likelihood and sensitivities
        parameters = np.array([mu, mu, sigma, sigma])
        score, sens = pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, 2 * ref_ll)
        self.assertEqual(len(sens), n_ids * 2 + 2 * 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dpsi)
        self.assertAlmostEqual(sens[11], ref_dpsi)
        self.assertAlmostEqual(sens[12], ref_dpsi)
        self.assertAlmostEqual(sens[13], ref_dpsi)
        self.assertAlmostEqual(sens[14], ref_dpsi)
        self.assertAlmostEqual(sens[15], ref_dpsi)
        self.assertAlmostEqual(sens[16], ref_dpsi)
        self.assertAlmostEqual(sens[17], ref_dpsi)
        self.assertAlmostEqual(sens[18], ref_dpsi)
        self.assertAlmostEqual(sens[19], ref_dpsi)
        self.assertAlmostEqual(sens[20], ref_dmu)
        self.assertAlmostEqual(sens[21], ref_dmu)
        self.assertAlmostEqual(sens[22], ref_dsigma)
        self.assertAlmostEqual(sens[23], ref_dsigma)

    def test_get_parameter_names(self):
        # Test case for 1 dim
        names = ['Mean Dim. 1', 'Std. Dim. 1']
        self.assertEqual(self.pop_model.get_parameter_names(), names)

        # Test case for 2 dim
        pop_model = chi.GaussianModel(n_dim=2)
        names = ['Mean Dim. 1', 'Mean Dim. 2', 'Std. Dim. 1', 'Std. Dim. 2']
        self.assertEqual(pop_model.get_parameter_names(), names)

    def test_n_hierarchical_parameters(self):
        n_ids = 10
        n_hierarchical_params = self.pop_model.n_hierarchical_parameters(n_ids)

        self.assertEqual(len(n_hierarchical_params), 2)
        self.assertEqual(n_hierarchical_params[0], n_ids)
        self.assertEqual(n_hierarchical_params[1], 2)

    def test_n_parameters(self):
        self.assertEqual(self.pop_model.n_parameters(), 2)

    def test_sample(self):
        # Test I: sample size 1
        seed = np.random.default_rng(seed=42)
        parameters = [3, 2]
        sample = self.pop_model.sample(parameters, seed=seed)

        n_samples = 1
        self.assertEqual(sample.shape, (n_samples, 1))

        # Test II: sample size > 1
        seed = 1
        parameters = [3, 2]
        n_samples = 4
        sample = self.pop_model.sample(
            parameters, n_samples=n_samples, seed=seed)

        self.assertEqual(
            sample.shape, (n_samples, 1))

        # Test III: multi-dimensional sampling
        seed = 1
        parameters = [3, 3, 2, 5]
        n_samples = 4
        pop_model = chi.GaussianModel(n_dim=2)
        sample = pop_model.sample(
            parameters, n_samples=n_samples, seed=seed)

        self.assertEqual(
            sample.shape, (n_samples, 2))

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.pop_model.sample(parameters)

        # Negative std
        parameters = [1, -1]

        with self.assertRaisesRegex(ValueError, 'A Gaussian distribution'):
            self.pop_model.sample(parameters)

    def test_set_parameter_names(self):
        # Test some name
        names = ['test', 'name']
        self.pop_model.set_parameter_names(names)

        self.assertEqual(
            self.pop_model.get_parameter_names(), names)

        # Set back to default name
        self.pop_model.set_parameter_names(None)
        names = self.pop_model.get_parameter_names()

        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Mean')
        self.assertEqual(names[1], 'Std.')

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'two', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.pop_model.set_parameter_names(names)


class TestHeterogeneousModel(unittest.TestCase):
    """
    Tests the chi.HeterogeneousModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = chi.HeterogeneousModel()

    def test_compute_log_likelihood(self):
        # For efficiency the input is actually not checked, and 0 is returned
        # regardless
        parameters = 'some parameters'
        observations = 'some observations'
        score = self.pop_model.compute_log_likelihood(parameters, observations)
        self.assertEqual(score, 0)

    def test_compute_pointwise_ll(self):
        # Test case I: Only the number of observations determines how many 0s
        # are returned
        # Test case I.1
        parameters = [1]
        observations = [0, 1, 1, 1]
        scores = self.pop_model.compute_pointwise_ll(
            parameters, observations)
        self.assertEqual(len(scores), 4)
        self.assertEqual(scores[0], 0)
        self.assertEqual(scores[1], 0)
        self.assertEqual(scores[2], 0)
        self.assertEqual(scores[3], 0)

        # Test case I.2
        parameters = [1]
        observations = [1, 2, 1, 10, 1]
        scores = self.pop_model.compute_pointwise_ll(
            parameters, observations)
        self.assertEqual(len(scores), 5)
        self.assertEqual(scores[0], 0)
        self.assertEqual(scores[1], 0)
        self.assertEqual(scores[2], 0)
        self.assertEqual(scores[3], 0)
        self.assertEqual(scores[4], 0)

    def test_compute_sensitivities(self):
        # For efficiency the input is actually not checked, and 0 is returned
        # regardless
        parameters = 'some parameters'
        observations = ['some', 'observations']
        score, sens = self.pop_model.compute_sensitivities(
            parameters, observations)
        self.assertEqual(score, 0)
        self.assertEqual(len(sens), 2)
        self.assertEqual(sens[0], 0)
        self.assertEqual(sens[1], 0)

    def test_get_parameter_names(self):
        self.assertIsNone(self.pop_model.get_parameter_names())

    def test_n_hierarchical_parameters(self):
        n_ids = 10
        n_hierachical_params = self.pop_model.n_hierarchical_parameters(n_ids)

        self.assertEqual(len(n_hierachical_params), 2)
        self.assertEqual(n_hierachical_params[0], n_ids)
        self.assertEqual(n_hierachical_params[1], 0)

    def test_n_parameters(self):
        self.assertEqual(self.pop_model.n_parameters(), 0)

    def test_sample(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.sample('some params')

    def test_set_get_parameter_names(self):
        # Check default name
        name = self.pop_model.get_parameter_names()
        self.assertIsNone(name)

        # Set name
        name = ['some name']
        self.pop_model.set_parameter_names(name)
        names = self.pop_model.get_parameter_names()
        self.assertIsNone(names)


class TestLogNormalModel(unittest.TestCase):
    """
    Tests the chi.LogNormalModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = chi.LogNormalModel()

    def test_compute_log_likelihood(self):
        # Hard to test exactly, but at least test some edge cases where
        # loglikelihood is straightforward to compute analytically

        n_ids = 10

        # Test case I: psis = 1, sigma_log = 1
        # Score reduces to
        # -n_ids * np.log(2*pi) / 2 - n_ids * mu_log^2 / 2

        # Test case I.1:
        psis = [1] * n_ids
        mu_log = 1
        sigma_log = 1
        ref_score = -n_ids * (np.log(2 * np.pi) + mu_log**2) / 2

        parameters = [mu_log] + [sigma_log]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case I.2:
        psis = [1] * n_ids
        mu_log = 5
        sigma_log = 1
        ref_score = -n_ids * (np.log(2 * np.pi) + mu_log**2) / 2

        parameters = [mu_log] + [sigma_log]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case II: psis = 1.
        # Score reduces to
        # -n_ids * log(sigma_log) - n_ids * log(2 * pi) / 2
        # - n_ids * mu_log^2 / (2 * sigma_log^2)

        # Test case II.1:
        psis = [1] * n_ids
        mu_log = 1
        sigma_log = 2
        ref_score = \
            -n_ids * (
                np.log(2 * np.pi * sigma_log**2)
                + mu_log**2 / sigma_log**2) / 2

        parameters = [mu_log] + [sigma_log]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case II.2:
        psis = [1] * n_ids
        mu_log = 3
        sigma_log = np.exp(3)
        ref_score = \
            -n_ids * (
                np.log(2 * np.pi * sigma_log**2)
                + mu_log**2 / sigma_log**2) / 2

        parameters = [mu_log] + [sigma_log]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case III: psis all the same, sigma_log = 1.
        # Score reduces to
        # -n_ids * log(psi) - n_ids * np.log(2 * pi) / 2
        # - n_ids * (log(psi) - mu_log)^2 / 2

        # Test case III.1
        psis = [np.exp(4)] * n_ids
        mu_log = 1
        sigma_log = 1
        ref_score = \
            -n_ids * (4 + np.log(2 * np.pi) / 2 + (4 - mu_log)**2 / 2)

        parameters = [mu_log] + [sigma_log]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case III.2
        psis = [np.exp(3)] * n_ids
        mu_log = 3
        sigma_log = 1
        ref_score = -n_ids * (3 + np.log(2 * np.pi) / 2)

        parameters = [mu_log] + [sigma_log]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case IV: sigma_log negative or zero

        # Test case IV.1
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # Test case IV.2
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = -10

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

    def test_compute_pointwise_ll(self):
        # TODO:
        with self.assertRaisesRegex(NotImplementedError, None):
            self.pop_model.compute_pointwise_ll('some', 'input')

        # # Hard to test exactly, but at least test some edge cases where
        # # loglikelihood is straightforward to compute analytically

        # n_ids = 10

        # # Test case I: psis = 1, sigma_log = 1
        # # Score reduces to
        # # -n_ids * np.log(2*pi) / 2 - n_ids * mu_log^2 / 2

        # # Test case I.1:
        # psis = [1] * n_ids
        # mu_log = 1
        # sigma_log = 1
        # ref_score = -n_ids * (np.log(2 * np.pi) + mu_log**2) / 2

        # parameters = [mu_log] + [sigma_log]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(len(scores), 10)
        # self.assertAlmostEqual(np.sum(scores), ref_score)
        # self.assertTrue(np.allclose(scores, ref_score / 10))

        # # Test case I.2:
        # n_ids = 6
        # psis = [1] * n_ids
        # mu_log = 5
        # sigma_log = 1
        # ref_score = -n_ids * (np.log(2 * np.pi) + mu_log**2) / 2

        # parameters = [mu_log] + [sigma_log]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(len(scores), 6)
        # self.assertAlmostEqual(np.sum(scores), ref_score)
        # self.assertTrue(np.allclose(scores, ref_score / 6))

        # # Test case II: psis = 1.
        # # Score reduces to
        # # -n_ids * log(sigma_log) - n_ids * log(2 * pi) / 2
        # # - n_ids * mu_log^2 / (2 * sigma_log^2)

        # # Test case II.1:
        # n_ids = 10
        # psis = [1] * n_ids
        # mu_log = 1
        # sigma_log = np.exp(2)
        # ref_score = \
        #     -n_ids * (
        #         np.log(2 * np.pi * sigma_log**2)
        #         + mu_log**2 / sigma_log**2) / 2

        # parameters = [mu_log] + [sigma_log]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(len(scores), 10)
        # self.assertAlmostEqual(np.sum(scores), ref_score)
        # self.assertTrue(np.allclose(scores, ref_score / 10))

        # # Test case II.2:
        # psis = [1] * n_ids
        # mu_log = 3
        # sigma_log = np.exp(3)
        # ref_score = \
        #     -n_ids * (
        #         np.log(2 * np.pi * sigma_log**2)
        #         + mu_log**2 / sigma_log**2) / 2

        # parameters = [mu_log] + [sigma_log]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(len(scores), 10)
        # self.assertAlmostEqual(np.sum(scores), ref_score)
        # self.assertTrue(np.allclose(scores, ref_score / 10))

        # # Test case III: Different psis
        # psis = [1, 2]
        # mu = 1
        # sigma = 1

        # parameters = [mu] + [sigma]
        # ref_score = self.pop_model.compute_log_likelihood(parameters, psis)
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(len(scores), 2)
        # self.assertAlmostEqual(np.sum(scores), ref_score)
        # self.assertNotEqual(scores[0], scores[1])

        # # Test case III: psis all the same, sigma_log = 1.
        # # Score reduces to
        # # -n_ids * log(psi) - n_ids * np.log(2 * pi) / 2
        # # - n_ids * (log(psi) - mu_log)^2 / 2

        # # Test case III.1
        # psis = [np.exp(4)] * n_ids
        # mu_log = 1
        # sigma_log = 1
        # ref_score = \
        #     -n_ids * (4 + np.log(2 * np.pi) / 2 + (4 - mu_log)**2 / 2)

        # parameters = [mu_log] + [sigma_log]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(len(scores), 10)
        # self.assertAlmostEqual(np.sum(scores), ref_score)
        # self.assertTrue(np.allclose(scores, ref_score / 10))

        # # Test case III.2
        # psis = [np.exp(3)] * n_ids
        # mu_log = 3
        # sigma_log = 1
        # ref_score = -n_ids * (3 + np.log(2 * np.pi) / 2)

        # parameters = [mu_log] + [sigma_log]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(len(scores), 10)
        # self.assertAlmostEqual(np.sum(scores), ref_score)
        # self.assertTrue(np.allclose(scores, ref_score / 10))

        # # Test case IV: mu_log or sigma_log negative or zero

        # # Test case IV.1
        # psis = [np.exp(10)] * n_ids
        # mu = 1
        # sigma = 0

        # parameters = [mu] + [sigma]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(scores[0], -np.inf)
        # self.assertEqual(scores[1], -np.inf)
        # self.assertEqual(scores[2], -np.inf)

        # # Test case IV.2
        # psis = [np.exp(10)] * n_ids
        # mu = 1
        # sigma = -10

        # parameters = [mu] + [sigma]
        # scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        # self.assertEqual(scores[0], -np.inf)
        # self.assertEqual(scores[1], -np.inf)
        # self.assertEqual(scores[2], -np.inf)

    def test_compute_sensitivities(self):
        # Hard to test exactly, but at least test some edge cases where
        # loglikelihood is straightforward to compute analytically

        n_ids = 10

        # Test case I: psis = 1, sigma_log = 1
        # Sensitivities reduce to
        # dpsi = -1 + mu_log
        # dmu = - mu_log * nids
        # dsigma = -(1 + mu_log^2) * nids

        # Test case I.1:
        psis = [1] * n_ids
        mu_log = 1
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = -1 + mu_log
        ref_dmu = -mu_log * n_ids
        ref_dsigma = (mu_log**2 - 1) * n_ids

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_dpsi)
        self.assertEqual(sens[1], ref_dpsi)
        self.assertEqual(sens[2], ref_dpsi)
        self.assertEqual(sens[3], ref_dpsi)
        self.assertEqual(sens[4], ref_dpsi)
        self.assertEqual(sens[5], ref_dpsi)
        self.assertEqual(sens[6], ref_dpsi)
        self.assertEqual(sens[7], ref_dpsi)
        self.assertEqual(sens[8], ref_dpsi)
        self.assertEqual(sens[9], ref_dpsi)
        self.assertEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case I.2:
        psis = [1] * n_ids
        mu_log = 5
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = -1 + mu_log
        ref_dmu = -mu_log * n_ids
        ref_dsigma = (mu_log**2 - 1) * n_ids

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_dpsi)
        self.assertEqual(sens[1], ref_dpsi)
        self.assertEqual(sens[2], ref_dpsi)
        self.assertEqual(sens[3], ref_dpsi)
        self.assertEqual(sens[4], ref_dpsi)
        self.assertEqual(sens[5], ref_dpsi)
        self.assertEqual(sens[6], ref_dpsi)
        self.assertEqual(sens[7], ref_dpsi)
        self.assertEqual(sens[8], ref_dpsi)
        self.assertEqual(sens[9], ref_dpsi)
        self.assertEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case II: psis = 1.
        # Sensitivities reduce to
        # dpsi = -1 + mu_log / var_log
        # dmu = - mu_log / var_log * nids
        # dsigma = (mu_log^2 / var_log - 1) / std_log * nids

        # Test case II.1:
        psis = [1] * n_ids
        mu_log = 1
        sigma_log = np.exp(2)

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = -1 + mu_log / sigma_log**2
        ref_dmu = -mu_log / sigma_log**2 * n_ids
        ref_dsigma = (mu_log**2 / sigma_log**2 - 1) / sigma_log * n_ids

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_dpsi)
        self.assertEqual(sens[1], ref_dpsi)
        self.assertEqual(sens[2], ref_dpsi)
        self.assertEqual(sens[3], ref_dpsi)
        self.assertEqual(sens[4], ref_dpsi)
        self.assertEqual(sens[5], ref_dpsi)
        self.assertEqual(sens[6], ref_dpsi)
        self.assertEqual(sens[7], ref_dpsi)
        self.assertEqual(sens[8], ref_dpsi)
        self.assertEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case II.2:
        psis = [1] * n_ids
        mu_log = 3
        sigma_log = np.exp(3)

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = -1 + mu_log / sigma_log**2
        ref_dmu = -mu_log / sigma_log**2 * n_ids
        ref_dsigma = (mu_log**2 / sigma_log**2 - 1) / sigma_log * n_ids

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_dpsi)
        self.assertEqual(sens[1], ref_dpsi)
        self.assertEqual(sens[2], ref_dpsi)
        self.assertEqual(sens[3], ref_dpsi)
        self.assertEqual(sens[4], ref_dpsi)
        self.assertEqual(sens[5], ref_dpsi)
        self.assertEqual(sens[6], ref_dpsi)
        self.assertEqual(sens[7], ref_dpsi)
        self.assertEqual(sens[8], ref_dpsi)
        self.assertEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case III: psis all the same, sigma_log = 1.
        # Score reduces to
        # dpsi = (-1 + mu_log - log psi) / psi
        # dmu = (log psi - mu_log) * nids
        # dsigma = ((log psi - mu_log)^2 - 1) * nids

        # Test case III.1
        psi = [np.exp(4)] * n_ids
        mu_log = 1
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psi)
        ref_dpsi = (-1 + mu_log - np.log(psi[0])) / psi[0]
        ref_dmu = (np.log(psi[0]) - mu_log) * n_ids
        ref_dsigma = ((np.log(psi[0]) - mu_log)**2 - 1) * n_ids

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psi)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_dpsi)
        self.assertEqual(sens[1], ref_dpsi)
        self.assertEqual(sens[2], ref_dpsi)
        self.assertEqual(sens[3], ref_dpsi)
        self.assertEqual(sens[4], ref_dpsi)
        self.assertEqual(sens[5], ref_dpsi)
        self.assertEqual(sens[6], ref_dpsi)
        self.assertEqual(sens[7], ref_dpsi)
        self.assertEqual(sens[8], ref_dpsi)
        self.assertEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case III.2
        psi = [np.exp(3)] * n_ids
        mu_log = 3
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psi)
        ref_dpsi = (-1 + mu_log - np.log(psi[0])) / psi[0]
        ref_dmu = (np.log(psi[0]) - mu_log) * n_ids
        ref_dsigma = ((np.log(psi[0]) - mu_log)**2 - 1) * n_ids

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psi)

        self.assertEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertEqual(sens[0], ref_dpsi)
        self.assertEqual(sens[1], ref_dpsi)
        self.assertEqual(sens[2], ref_dpsi)
        self.assertEqual(sens[3], ref_dpsi)
        self.assertEqual(sens[4], ref_dpsi)
        self.assertEqual(sens[5], ref_dpsi)
        self.assertEqual(sens[6], ref_dpsi)
        self.assertEqual(sens[7], ref_dpsi)
        self.assertEqual(sens[8], ref_dpsi)
        self.assertEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case IV: Compare gradients to numpy.gradient
        epsilon = 0.00001
        n_parameters = n_ids + self.pop_model.n_parameters()
        parameters = np.full(shape=n_parameters, fill_value=0.3)
        ref_sens = []
        for index in range(n_parameters):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.pop_model.compute_log_likelihood(
                        low[n_ids:], low[:n_ids]),
                    self.pop_model.compute_log_likelihood(
                        parameters[n_ids:], parameters[:n_ids]),
                    self.pop_model.compute_log_likelihood(
                        high[n_ids:], high[:n_ids])],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities with hierarchical model
        _, sens = self.pop_model.compute_sensitivities(
            parameters[n_ids:], parameters[:n_ids])

        self.assertEqual(len(sens), 12)
        self.assertAlmostEqual(sens[0], ref_sens[0])
        self.assertAlmostEqual(sens[1], ref_sens[1])
        self.assertAlmostEqual(sens[2], ref_sens[2])
        self.assertAlmostEqual(sens[3], ref_sens[3])
        self.assertAlmostEqual(sens[4], ref_sens[4])
        self.assertAlmostEqual(sens[5], ref_sens[5])
        self.assertAlmostEqual(sens[6], ref_sens[6])
        self.assertAlmostEqual(sens[7], ref_sens[7])
        self.assertAlmostEqual(sens[8], ref_sens[8])
        self.assertAlmostEqual(sens[9], ref_sens[9])
        self.assertAlmostEqual(sens[10], ref_sens[10], 5)
        self.assertAlmostEqual(sens[11], ref_sens[11], 5)

        # Test case V: mu_log or sigma_log negative or zero

        # Test case V.1
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)
        self.assertEqual(score, -np.inf)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)

        # Test case V.2
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = -10

        parameters = [mu] + [sigma]
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)
        self.assertEqual(score, -np.inf)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)

        # Test case VI. Multi-dimensional
        psi = [np.exp(3)] * n_ids
        mu_log = 3
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psi)
        ref_dpsi = (-1 + mu_log - np.log(psi[0])) / psi[0]
        ref_dmu = (np.log(psi[0]) - mu_log) * n_ids
        ref_dsigma = ((np.log(psi[0]) - mu_log)**2 - 1) * n_ids

        # Compute log-likelihood and sensitivities
        pop_model = chi.LogNormalModel(n_dim=2)
        psi = np.vstack((np.array(psi), np.array(psi))).T
        parameters = [mu_log, mu_log, sigma_log, sigma_log]
        score, sens = pop_model.compute_sensitivities(parameters, psi)

        self.assertAlmostEqual(score, 2 * ref_ll)
        self.assertEqual(len(sens), (n_ids + 2) * 2)
        self.assertEqual(sens[0], ref_dpsi)
        self.assertEqual(sens[1], ref_dpsi)
        self.assertEqual(sens[2], ref_dpsi)
        self.assertEqual(sens[3], ref_dpsi)
        self.assertEqual(sens[4], ref_dpsi)
        self.assertEqual(sens[5], ref_dpsi)
        self.assertEqual(sens[6], ref_dpsi)
        self.assertEqual(sens[7], ref_dpsi)
        self.assertEqual(sens[8], ref_dpsi)
        self.assertEqual(sens[9], ref_dpsi)
        self.assertEqual(sens[10], ref_dpsi)
        self.assertEqual(sens[11], ref_dpsi)
        self.assertEqual(sens[12], ref_dpsi)
        self.assertEqual(sens[13], ref_dpsi)
        self.assertEqual(sens[14], ref_dpsi)
        self.assertEqual(sens[15], ref_dpsi)
        self.assertEqual(sens[16], ref_dpsi)
        self.assertEqual(sens[17], ref_dpsi)
        self.assertEqual(sens[18], ref_dpsi)
        self.assertEqual(sens[19], ref_dpsi)
        self.assertAlmostEqual(sens[20], ref_dmu)
        self.assertAlmostEqual(sens[21], ref_dmu)
        self.assertAlmostEqual(sens[22], ref_dsigma)
        self.assertAlmostEqual(sens[23], ref_dsigma)

    def test_get_mean_and_std(self):
        # Test case I: std_log = 0
        # Then:
        # mean = exp(mean_log)
        # std = 0

        # Test case I.1:
        mean_log = 1
        std_log = 0
        parameters = [mean_log, std_log]
        mean, std = self.pop_model.get_mean_and_std(parameters)

        self.assertEqual(np.exp(mean_log), mean)
        self.assertEqual(std_log, std)

        # Test case I.2:
        mean_log = -3
        std_log = 0
        parameters = [mean_log, std_log]
        mean, std = self.pop_model.get_mean_and_std(parameters)

        self.assertEqual(np.exp(mean_log), mean)
        self.assertEqual(std_log, std)

        # Test case II: mean_log = 0
        # Then:
        # mean = exp(std_log**2/2)
        # std = sqrt(exp(std_log**2)*(exp(std_log**2) - 1))

        # Test case I.1:
        mean_log = 0
        std_log = 1

        # Compute references
        mean_ref = np.exp(std_log**2 / 2)
        std_ref = np.sqrt(
            np.exp(std_log**2)*(np.exp(std_log**2) - 1))

        parameters = [mean_log, std_log]
        mean, std = self.pop_model.get_mean_and_std(parameters)

        self.assertEqual(mean, mean_ref)
        self.assertEqual(std, std_ref)

        # Test case I.2:
        mean_log = 0
        std_log = 2

        # Compute references
        mean_ref = np.exp(std_log**2 / 2)
        std_ref = np.sqrt(
            np.exp(std_log**2)*(np.exp(std_log**2) - 1))

        parameters = [mean_log, std_log]
        mean, std = self.pop_model.get_mean_and_std(parameters)

        self.assertEqual(mean, mean_ref)
        self.assertEqual(std, std_ref)

        # Test case II: Negative standard deviation
        mean_log = 0
        std_log = -1
        parameters = [mean_log, std_log]
        with self.assertRaisesRegex(ValueError, 'The standard deviation'):
            self.pop_model.get_mean_and_std(parameters)

        # Test case III: Multi-dimensional
        pop_model = chi.LogNormalModel(n_dim=3)
        m = pop_model.get_mean_and_std([1, 1, 1, 1, 1, 1])
        self.assertEqual(m.shape, (2, 3))

    def test_get_parameter_names(self):
        names = ['Log mean Dim. 1', 'Log std. Dim. 1']

        self.assertEqual(self.pop_model.get_parameter_names(), names)

    def test_n_hierarchical_parameters(self):
        n_ids = 10
        n_hierarchical_params = self.pop_model.n_hierarchical_parameters(n_ids)

        self.assertEqual(len(n_hierarchical_params), 2)
        self.assertEqual(n_hierarchical_params[0], n_ids)
        self.assertEqual(n_hierarchical_params[1], 2)

    def test_n_parameters(self):
        self.assertEqual(self.pop_model.n_parameters(), 2)

    def test_sample(self):
        # Test I: sample size 1
        seed = 42
        parameters = [3, 2]
        sample = self.pop_model.sample(parameters, seed=seed)

        n_samples = 1
        self.assertEqual(sample.shape, (n_samples, 1))

        # Test II: sample size > 1
        parameters = [3, 2]
        n_samples = 4
        sample = self.pop_model.sample(
            parameters, n_samples=n_samples, seed=seed)

        self.assertEqual(
            sample.shape, (n_samples, 1))

        # Test case III: Multi-dimensional
        parameters = [3, 2, 3, 2]
        n_samples = 4
        pop_model = chi.LogNormalModel(n_dim=2)
        sample = pop_model.sample(
            parameters, n_samples=n_samples, seed=seed)

        self.assertEqual(
            sample.shape, (n_samples, 2))

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.pop_model.sample(parameters)

        # Negative std
        parameters = [1, -1]

        with self.assertRaisesRegex(ValueError, 'A log-normal distribution'):
            self.pop_model.sample(parameters)

    def test_set_parameter_names(self):
        # Test some name
        names = ['test', 'name']
        self.pop_model.set_parameter_names(names)

        self.assertEqual(
            self.pop_model.get_parameter_names(), names)

        # Set back to default name
        self.pop_model.set_parameter_names(None)
        names = self.pop_model.get_parameter_names()

        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Log mean Dim. 1')
        self.assertEqual(names[1], 'Log std. Dim. 1')

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'two', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.pop_model.set_parameter_names(names)


class TestPooledModel(unittest.TestCase):
    """
    Tests the chi.PooledModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = chi.PooledModel()

    def test_compute_log_likelihood(self):
        # Test case I: observation differ from parameter
        # Test case I.1
        parameters = [1]
        observations = [0, 1, 1, 1]
        score = self.pop_model.compute_log_likelihood(parameters, observations)
        self.assertEqual(score, -np.inf)

        # Test case I.1
        parameters = [1]
        observations = [1, 1, 1, 10]
        score = self.pop_model.compute_log_likelihood(parameters, observations)
        self.assertEqual(score, -np.inf)

        # Test case II: all values agree with parameter
        parameters = [1]
        observations = [1, 1, 1, 1]
        score = self.pop_model.compute_log_likelihood(parameters, observations)
        self.assertEqual(score, 0)

    def test_compute_pointwise_ll(self):
        # Test case I: observation differ from parameter
        # Test case I.1
        parameters = [1]
        observations = [0, 1, 1, 1]
        scores = self.pop_model.compute_pointwise_ll(
            parameters, observations)
        self.assertEqual(len(scores), 4)
        self.assertEqual(scores[0], -np.inf)
        self.assertEqual(scores[1], 0)
        self.assertEqual(scores[2], 0)
        self.assertEqual(scores[3], 0)

        # Test case I.2
        parameters = [1]
        observations = [1, 2, 1, 10, 1]
        scores = self.pop_model.compute_pointwise_ll(
            parameters, observations)
        self.assertEqual(len(scores), 5)
        self.assertEqual(scores[0], 0)
        self.assertEqual(scores[1], -np.inf)
        self.assertEqual(scores[2], 0)
        self.assertEqual(scores[3], -np.inf)
        self.assertEqual(scores[4], 0)

        # Test case II: all values agree with parameter
        parameters = [1]
        observations = [1, 1, 1]
        scores = self.pop_model.compute_pointwise_ll(
            parameters, observations)
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[0], 0)
        self.assertEqual(scores[1], 0)
        self.assertEqual(scores[2], 0)

    def test_compute_sensitivities(self):
        # Test case I: observation differ from parameter
        # Test case I.1
        parameters = [1]
        observations = [0, 1, 1, 1]
        score, sens = self.pop_model.compute_sensitivities(
            parameters, observations)
        self.assertEqual(score, -np.inf)
        self.assertEqual(len(sens), 5)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)
        self.assertEqual(sens[3], np.inf)
        self.assertEqual(sens[4], np.inf)

        # Test case I.1
        parameters = [1]
        observations = [1, 1, 1, 10]
        score, sens = self.pop_model.compute_sensitivities(
            parameters, observations)
        self.assertEqual(score, -np.inf)
        self.assertEqual(len(sens), 5)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)
        self.assertEqual(sens[3], np.inf)
        self.assertEqual(sens[4], np.inf)

        # Test case II: all values agree with parameter
        parameters = [1]
        observations = [1, 1, 1, 1]
        score, sens = self.pop_model.compute_sensitivities(
            parameters, observations)
        self.assertEqual(score, 0)
        self.assertEqual(len(sens), 5)
        self.assertEqual(sens[0], 0)
        self.assertEqual(sens[1], 0)
        self.assertEqual(sens[2], 0)
        self.assertEqual(sens[3], 0)
        self.assertEqual(sens[4], 0)

    def test_get_parameter_names(self):
        names = ['Pooled']

        self.assertEqual(self.pop_model.get_parameter_names(), names)

    def test_n_hierarchical_parameters(self):
        n_ids = 10
        n_hierarchical_params = self.pop_model.n_hierarchical_parameters(n_ids)

        self.assertEqual(len(n_hierarchical_params), 2)
        self.assertEqual(n_hierarchical_params[0], 0)
        self.assertEqual(n_hierarchical_params[1], 1)

    def test_n_parameters(self):
        self.assertEqual(self.pop_model.n_parameters(), 1)

    def test_sample(self):
        # Test one sample size 1
        parameters = [3]
        sample = self.pop_model.sample(parameters)

        n_samples = 1
        self.assertEqual(sample.shape, (n_samples,))
        self.assertEqual(sample[0], parameters[0])

        # Test one sample size > 1
        parameters = [3]
        n_samples = 4
        sample = self.pop_model.sample(parameters, n_samples=n_samples)

        self.assertEqual(
            sample.shape, (n_samples,))
        self.assertEqual(sample[0], parameters[0])
        self.assertEqual(sample[1], parameters[0])
        self.assertEqual(sample[2], parameters[0])
        self.assertEqual(sample[3], parameters[0])

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.pop_model.sample(parameters)

    def test_set_parameter_names(self):
        # Test some name
        names = ['test name']
        self.pop_model.set_parameter_names(names)

        self.assertEqual(
            self.pop_model.get_parameter_names(), names)

        # Set back to default name
        self.pop_model.set_parameter_names(None)
        names = self.pop_model.get_parameter_names()

        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Pooled')

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'one', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.pop_model.set_parameter_names(names)


class TestPopulationModel(unittest.TestCase):
    """
    Tests the chi.PopulationModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = chi.PopulationModel()

    def test_compute_log_likelihood(self):
        parameters = 'some parameters'
        observations = 'some observations'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.compute_log_likelihood(parameters, observations)

    def test_compute_pointwise_ll(self):
        parameters = 'some parameters'
        observations = 'some observations'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.compute_pointwise_ll(parameters, observations)

    def test_compute_sensitivities(self):
        parameters = 'some parameters'
        observations = 'some observations'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.compute_sensitivities(parameters, observations)

    def test_get_parameter_names(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.get_parameter_names()

    def test_n_dim(self):
        pop_model = chi.PopulationModel(n_dim=1, dim_names=None)
        self.assertEqual(pop_model.n_dim(), 1)
        self.assertEqual(pop_model.get_dim_names(), ['Dim. 1'])
        pop_model.set_dim_names(['Some name'])
        self.assertEqual(pop_model.get_dim_names(), ['Some name'])
        pop_model.set_dim_names(None)
        self.assertEqual(pop_model.get_dim_names(), ['Dim. 1'])

        pop_model = chi.PopulationModel(n_dim=2, dim_names=['Some', 'name'])
        self.assertEqual(pop_model.n_dim(), 2)
        names = pop_model.get_dim_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Some')
        self.assertEqual(names[1], 'name')
        pop_model.set_dim_names(None)
        names = pop_model.get_dim_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Dim. 1')
        self.assertEqual(names[1], 'Dim. 2')

    def test_bad_dim(self):
        with self.assertRaisesRegex(ValueError, 'The dimension of the pop'):
            chi.PopulationModel(n_dim=0)

        # too few names
        with self.assertRaisesRegex(ValueError, 'The number of dimension'):
            chi.PopulationModel(n_dim=2, dim_names=['name'])

        pop_model = chi.PopulationModel(n_dim=2)
        with self.assertRaisesRegex(ValueError, 'Length of names does'):
            pop_model.set_dim_names(['name'])

    def test_n_hierarchical_parameters(self):
        n_ids = 'some ids'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.n_hierarchical_parameters(n_ids)

    def test_n_parameters(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.n_parameters()

    def test_transforms_individual_parameters(self):
        self.assertFalse(self.pop_model.transforms_individual_parameters())

    def test_sample(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.sample('some values')

    def test_set_parameter_names(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.set_parameter_names('some name')


class TestReducedPopulationModel(unittest.TestCase):
    """
    Tests the chi.ReducedPopulationModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Test case I: Non-covariate population model
        pop_model = chi.LogNormalModel()
        cls.pop_model = chi.ReducedPopulationModel(pop_model)

        # Test case II: Covariate population model
        cls.bare_pop_model = chi.CovariatePopulationModel(
            chi.GaussianModel(),
            chi.LogNormalLinearCovariateModel(n_covariates=2)
        )
        cls.cpop_model = chi.ReducedPopulationModel(cls.bare_pop_model)

    def test_bad_instantiation(self):
        model = 'Bad type'
        with self.assertRaisesRegex(TypeError, 'The population model'):
            chi.ReducedPopulationModel(model)

    def test_compute_individual_parameters(self):
        # Test case I: Model does not transform psi
        parameters = [1, 10]
        eta = [0.2, -0.3, 1, 5]
        psi = self.pop_model.compute_individual_parameters(
            parameters, eta)

        self.assertEqual(len(psi), 4)
        self.assertEqual(psi[0], eta[0])
        self.assertEqual(psi[1], eta[1])
        self.assertEqual(psi[2], eta[2])
        self.assertEqual(psi[3], eta[3])

        # Test case II: Model transforms psi
        # Test case II.1: No fixed parameters
        parameters = [1, 1, -1, 1]
        eta = [0.2, -0.3, 1, 5]
        covariates = np.ones(shape=(4, 2))

        ref_psi = self.bare_pop_model.compute_individual_parameters(
            parameters, eta, covariates)
        psi = self.cpop_model.compute_individual_parameters(
            parameters, eta, covariates)
        self.assertEqual(psi[0], ref_psi[0])
        self.assertEqual(psi[1], ref_psi[1])
        self.assertEqual(psi[2], ref_psi[2])
        self.assertEqual(psi[3], ref_psi[3])

        # Test case II.1: Fix some parameters
        self.cpop_model.fix_parameters({
            'Base mean log': 1,
            'Shift Covariate 1': -1
        })
        reduced_parameters = [1, 1]
        eta = [0.2, -0.3, 1, 5]
        covariates = np.ones(shape=(4, 2))

        ref_psi = self.bare_pop_model.compute_individual_parameters(
            parameters, eta, covariates)
        psi = self.cpop_model.compute_individual_parameters(
            reduced_parameters, eta, covariates)
        self.assertEqual(len(psi), 4)
        self.assertEqual(psi[0], ref_psi[0])
        self.assertEqual(psi[1], ref_psi[1])
        self.assertEqual(psi[2], ref_psi[2])
        self.assertEqual(psi[3], ref_psi[3])

        # Unfix parameters
        self.cpop_model.fix_parameters({
            'Base mean log': None,
            'Shift Covariate 1': None
        })

    def test_compute_individual_sensitivities(self):
        # Test case I: Model does not transform psi
        parameters = [1, 10]
        eta = [0.2, -0.3, 1, 5]
        psi, sens = self.pop_model.compute_individual_sensitivities(
            parameters, eta)

        self.assertEqual(len(psi), 4)
        self.assertEqual(psi[0], eta[0])
        self.assertEqual(psi[1], eta[1])
        self.assertEqual(psi[2], eta[2])
        self.assertEqual(psi[3], eta[3])

        self.assertEqual(sens.shape, (3, 4))
        self.assertEqual(sens[0, 0], 1)
        self.assertEqual(sens[0, 1], 1)
        self.assertEqual(sens[0, 2], 1)
        self.assertEqual(sens[0, 3], 1)
        self.assertEqual(sens[1, 0], 0)
        self.assertEqual(sens[1, 1], 0)
        self.assertEqual(sens[1, 2], 0)
        self.assertEqual(sens[1, 3], 0)
        self.assertEqual(sens[2, 0], 0)
        self.assertEqual(sens[2, 1], 0)
        self.assertEqual(sens[2, 2], 0)
        self.assertEqual(sens[2, 3], 0)

        # Test case II: Model transforms psi
        # Test case II.1: No fixed parameters
        parameters = [1, 1, -1, 1]
        eta = [0.2, -0.3, 1, 5]
        covariates = np.ones(shape=(4, 2))

        ref_psi, ref_sens = \
            self.bare_pop_model.compute_individual_sensitivities(
                parameters, eta, covariates)
        psi, sens = self.cpop_model.compute_individual_sensitivities(
            parameters, eta, covariates)

        self.assertEqual(len(psi), 4)
        self.assertEqual(psi[0], ref_psi[0])
        self.assertEqual(psi[1], ref_psi[1])
        self.assertEqual(psi[2], ref_psi[2])
        self.assertEqual(psi[3], ref_psi[3])

        self.assertEqual(sens.shape, (5, 4))
        self.assertEqual(sens[0, 0], ref_sens[0, 0])
        self.assertEqual(sens[0, 1], ref_sens[0, 1])
        self.assertEqual(sens[0, 2], ref_sens[0, 2])
        self.assertEqual(sens[0, 3], ref_sens[0, 3])
        self.assertEqual(sens[1, 0], ref_sens[1, 0])
        self.assertEqual(sens[1, 1], ref_sens[1, 1])
        self.assertEqual(sens[1, 2], ref_sens[1, 2])
        self.assertEqual(sens[1, 3], ref_sens[1, 3])
        self.assertEqual(sens[2, 0], ref_sens[2, 0])
        self.assertEqual(sens[2, 1], ref_sens[2, 1])
        self.assertEqual(sens[2, 2], ref_sens[2, 2])
        self.assertEqual(sens[2, 3], ref_sens[2, 3])
        self.assertEqual(sens[3, 0], ref_sens[3, 0])
        self.assertEqual(sens[3, 1], ref_sens[3, 1])
        self.assertEqual(sens[3, 2], ref_sens[3, 2])
        self.assertEqual(sens[3, 3], ref_sens[3, 3])
        self.assertEqual(sens[4, 0], ref_sens[4, 0])
        self.assertEqual(sens[4, 1], ref_sens[4, 1])
        self.assertEqual(sens[4, 2], ref_sens[4, 2])
        self.assertEqual(sens[4, 3], ref_sens[4, 3])

        # Test case II.2: Fix some parameters
        self.cpop_model.fix_parameters({
            'Base mean log': 1,
            'Shift Covariate 1': -1
        })
        reduced_parameters = [1, 1]
        eta = [0.2, -0.3, 1, 5]
        covariates = np.ones(shape=(4, 2))

        ref_psi, ref_sens = \
            self.bare_pop_model.compute_individual_sensitivities(
                parameters, eta, covariates)
        psi, sens = self.cpop_model.compute_individual_sensitivities(
            reduced_parameters, eta, covariates)

        self.assertEqual(len(psi), 4)
        self.assertEqual(psi[0], ref_psi[0])
        self.assertEqual(psi[1], ref_psi[1])
        self.assertEqual(psi[2], ref_psi[2])
        self.assertEqual(psi[3], ref_psi[3])

        self.assertEqual(sens.shape, (5, 4))
        self.assertEqual(sens[0, 0], ref_sens[0, 0])
        self.assertEqual(sens[0, 1], ref_sens[0, 1])
        self.assertEqual(sens[0, 2], ref_sens[0, 2])
        self.assertEqual(sens[0, 3], ref_sens[0, 3])
        self.assertEqual(sens[1, 0], ref_sens[1, 0])
        self.assertEqual(sens[1, 1], ref_sens[1, 1])
        self.assertEqual(sens[1, 2], ref_sens[1, 2])
        self.assertEqual(sens[1, 3], ref_sens[1, 3])
        self.assertEqual(sens[2, 0], ref_sens[2, 0])
        self.assertEqual(sens[2, 1], ref_sens[2, 1])
        self.assertEqual(sens[2, 2], ref_sens[2, 2])
        self.assertEqual(sens[2, 3], ref_sens[2, 3])
        self.assertEqual(sens[3, 0], ref_sens[3, 0])
        self.assertEqual(sens[3, 1], ref_sens[3, 1])
        self.assertEqual(sens[3, 2], ref_sens[3, 2])
        self.assertEqual(sens[3, 3], ref_sens[3, 3])
        self.assertEqual(sens[4, 0], ref_sens[4, 0])
        self.assertEqual(sens[4, 1], ref_sens[4, 1])
        self.assertEqual(sens[4, 2], ref_sens[4, 2])
        self.assertEqual(sens[4, 3], ref_sens[4, 3])

        # Unfix parameters
        self.cpop_model.fix_parameters({
            'Base mean log': None,
            'Shift Covariate 1': None
        })

    def test_compute_log_likelihood(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': 1})

        # Compute log-likelihood
        parameters = [2]
        observations = [2, 3, 4, 5]
        score = self.pop_model.compute_log_likelihood(
            parameters, observations)

        # Compute ref score with original error model
        parameters = [1, 2]
        error_model = self.pop_model.get_population_model()
        ref_score = error_model.compute_log_likelihood(
            parameters, observations)

        self.assertEqual(score, ref_score)

        # Unfix model parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': None})

    def test_compute_pointwise_ll(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': 1})

        # Compute log-likelihood
        parameters = [2]
        observations = [2, 3, 4, 5]
        scores = self.pop_model.compute_pointwise_ll(
            parameters, observations)

        # Compute ref score with original error model
        parameters = [1, 2]
        error_model = self.pop_model.get_population_model()
        ref_scores = error_model.compute_pointwise_ll(
            parameters, observations)

        self.assertEqual(len(scores), 4)
        self.assertEqual(scores[0], ref_scores[0])
        self.assertEqual(scores[1], ref_scores[1])
        self.assertEqual(scores[2], ref_scores[2])
        self.assertEqual(scores[3], ref_scores[3])

        # Unfix model parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': None})

    def test_compute_sensitivities(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': 1})

        # Compute log-likelihood
        parameters = [2]
        observations = [2, 3, 4, 5]
        score, sens = self.pop_model.compute_sensitivities(
            parameters, observations)

        # Compute ref score with original error model
        parameters = [1, 2]
        error_model = self.pop_model.get_population_model()
        ref_score, ref_sens = error_model.compute_sensitivities(
            parameters, observations)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 5)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[5])

        # Unfix model parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': None})

        # Compute log-likelihood
        score, sens = self.pop_model.compute_sensitivities(
            parameters, observations)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 6)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': 1})

        n_parameters = self.pop_model.n_parameters()
        self.assertEqual(n_parameters, 1)

        parameter_names = self.pop_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 1)
        self.assertEqual(parameter_names[0], 'Std. log')

        # Test case II: fix overlapping set of parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': 0.2,
            'Std. log': 0.1})

        n_parameters = self.pop_model.n_parameters()
        self.assertEqual(n_parameters, 0)

        parameter_names = self.pop_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 0)

        # Test case III: unfix all parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': None,
            'Std. log': None})

        n_parameters = self.pop_model.n_parameters()
        self.assertEqual(n_parameters, 2)

        parameter_names = self.pop_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 2)
        self.assertEqual(parameter_names[0], 'Mean log')
        self.assertEqual(parameter_names[1], 'Std. log')

    def test_fix_parameters_bad_input(self):
        name_value_dict = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The name-value dictionary'):
            self.pop_model.fix_parameters(name_value_dict)

    def test_get_population_model(self):
        pop_model = self.pop_model.get_population_model()
        self.assertIsInstance(pop_model, chi.PopulationModel)

    def test_n_covariates(self):
        # Test case I: Has no covariates
        n = self.pop_model.n_covariates()
        self.assertEqual(n, 0)

        # Test case II: Has covariates
        n = self.cpop_model.n_covariates()
        self.assertEqual(n, 2)

    def test_n_hierarchical_parameters(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Std. log': 0.1})

        n_ids = 10
        n_indiv, n_pop = self.pop_model.n_hierarchical_parameters(n_ids)
        self.assertEqual(n_indiv, 10)
        self.assertEqual(n_pop, 1)

        # Unfix all parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Std. log': None})

        n_ids = 10
        n_indiv, n_pop = self.pop_model.n_hierarchical_parameters(n_ids)
        self.assertEqual(n_indiv, 10)
        self.assertEqual(n_pop, 2)

    def test_n_fixed_parameters(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Std. log': 0.1})

        self.assertEqual(self.pop_model.n_fixed_parameters(), 1)

        # Unfix all parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Std. log': None})

        self.assertEqual(self.pop_model.n_fixed_parameters(), 0)

    def test_n_parameters(self):
        n_parameters = self.pop_model.n_parameters()
        self.assertEqual(n_parameters, 2)

    def test_sample(self):
        # Test case I: No covariates
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': 0.1})

        # Sample
        seed = 42
        n_samples = 4
        parameters = [0.2]
        samples = self.pop_model.sample(parameters, n_samples, seed)

        # Compute ref score with original population model
        parameters = [0.1, 0.2]
        pop_model = self.pop_model.get_population_model()
        ref_samples = pop_model.sample(parameters, n_samples, seed)

        self.assertEqual(samples.shape, (4,))
        self.assertEqual(ref_samples.shape, (4,))
        self.assertEqual(samples[0], ref_samples[0])
        self.assertEqual(samples[1], ref_samples[1])
        self.assertEqual(samples[2], ref_samples[2])
        self.assertEqual(samples[3], ref_samples[3])

        # Unfix model parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': None})

        # Test case II: Covariates
        seed = 42
        n_samples = 4
        parameters = [1, 1, -1, 1]
        covariates = [2, 3]
        samples = self.cpop_model.sample(
            parameters, n_samples, seed, covariates, return_psi=True)
        ref_samples = self.bare_pop_model.sample(
            parameters, n_samples, seed, covariates, return_psi=True)

        self.assertEqual(samples.shape, (4,))
        self.assertEqual(ref_samples.shape, (4,))
        self.assertEqual(samples[0], ref_samples[0])
        self.assertEqual(samples[1], ref_samples[1])
        self.assertEqual(samples[2], ref_samples[2])
        self.assertEqual(samples[3], ref_samples[3])

    def test_set_get_covariate_names(self):
        # Test case I: Has no covariates
        names = self.pop_model.get_covariate_names()
        self.assertEqual(len(names), 0)

        self.pop_model.set_covariate_names(['some', 'names'])
        names = self.pop_model.get_covariate_names()
        self.assertEqual(len(names), 0)

        # Test case II: Has covariates
        names = self.cpop_model.get_covariate_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Covariate 1')
        self.assertEqual(names[1], 'Covariate 2')

        self.cpop_model.set_covariate_names(['some', 'names'])
        names = self.cpop_model.get_covariate_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'some')
        self.assertEqual(names[1], 'names')

        self.cpop_model.set_covariate_names(
            ['Covariate 1', 'Covariate 2'])

    def test_set_get_parameter_names(self):
        # Set some parameter names
        names = ['Test 1', 'Test 2']
        self.pop_model.set_parameter_names(names)

        names = self.pop_model.get_parameter_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Test 1')
        self.assertEqual(names[1], 'Test 2')

        # Reset to defaults
        self.pop_model.set_parameter_names(None)

        names = self.pop_model.get_parameter_names()
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Mean log')
        self.assertEqual(names[1], 'Std. log')

        # Fix parameter and set parameter name
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': 1})
        self.pop_model.set_parameter_names(
            ['Std. log myokit.tumour_volume'])

        names = self.pop_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Std. log myokit.tumour_volume')

        # Reset to defaults
        self.pop_model.set_parameter_names(None)

        names = self.pop_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Std. log')

        # Unfix model parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean log': None})

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['Wrong length']
        with self.assertRaisesRegex(ValueError, 'Length of names does not'):
            self.pop_model.set_parameter_names(names)

        # A parameter exceeds 50 characters
        names = [
            '0123456789-0123456789-0123456789-0123456789-0123456789-012345678',
            'Sigma base']
        with self.assertRaisesRegex(ValueError, 'Parameter names cannot'):
            self.pop_model.set_parameter_names(names)

    def test_transforms_individual_parameters(self):
        # Test case I: No transform
        self.assertFalse(self.pop_model.transforms_individual_parameters())

        # Test case II: Transforms parameters
        self.assertTrue(self.cpop_model.transforms_individual_parameters())


class TestTruncatedGaussianModel(unittest.TestCase):
    """
    Tests the chi.TruncatedGaussianModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = chi.TruncatedGaussianModel()

    def test_compute_log_likelihood(self):
        # Hard to test exactly, but at least test some edge cases where
        # loglikelihood is straightforward to compute analytically

        n_ids = 10

        # Test case I: psis = 1, mu = 1, sigma = 1
        # Score reduces to
        # -nids * (np.log(2pi)/2 + np.log(1 - Phi(-1)))

        # Test case I.1:
        psis = [1] * n_ids
        mu = 1
        sigma = 1
        ref_score1 = - n_ids * (
            np.log(2*np.pi) / 2 + np.log(1 - norm.cdf(-mu/sigma)))
        a = (0 - mu) / sigma
        ref_score2 = np.sum(truncnorm.logpdf(
            psis, a=a, b=np.inf, loc=mu, scale=sigma))

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score1)
        self.assertAlmostEqual(score, ref_score2)

        # Test case I.2:
        psis = [5] * n_ids
        mu = 5
        sigma = 1
        ref_score1 = - n_ids * (
            np.log(2*np.pi) / 2 + np.log(1 - norm.cdf(-mu/sigma)))
        a = (0 - mu) / sigma
        ref_score2 = np.sum(truncnorm.logpdf(
            psis, a=a, b=np.inf, loc=mu, scale=sigma))

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score1)
        self.assertAlmostEqual(score, ref_score2)

        # Test case II: psis != mu, sigma = 1.
        # Score reduces to
        # -nids * (np.log(2pi)/2 + (psi - mu)^2/2 + np.log(1 - Phi(-mu)))

        # Test case II.1:
        psis = [2] * n_ids
        mu = 1
        sigma = 1
        ref_score1 = - n_ids * (
            np.log(2*np.pi) / 2 +
            (psis[0] - mu)**2 / 2 +
            np.log(1 - norm.cdf(-mu/sigma)))
        a = (0 - mu) / sigma
        ref_score2 = np.sum(truncnorm.logpdf(
            psis, a=a, b=np.inf, loc=mu, scale=sigma))

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score1)
        self.assertAlmostEqual(score, ref_score2)

        # Test case II.2:
        psis = [2] * n_ids
        mu = 10
        sigma = 1
        ref_score1 = - n_ids * (
            np.log(2*np.pi) / 2 +
            (psis[0] - mu)**2 / 2 +
            np.log(1 - norm.cdf(-mu/sigma)))
        a = (0 - mu) / sigma
        ref_score2 = np.sum(truncnorm.logpdf(
            psis, a=a, b=np.inf, loc=mu, scale=sigma))

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score1)
        self.assertAlmostEqual(score, ref_score2)

        # Test case III: Any parameters

        # Test case III.1
        psis = np.arange(10)
        mu = 1
        sigma = 1
        a = (0 - mu) / sigma
        ref_score = np.sum(truncnorm.logpdf(
            psis, a=a, b=np.inf, loc=mu, scale=sigma))

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case III.2
        psis = np.arange(10)
        mu = 10
        sigma = 15
        a = (0 - mu) / sigma
        ref_score = np.sum(truncnorm.logpdf(
            psis, a=a, b=np.inf, loc=mu, scale=sigma))

        parameters = [mu, sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertAlmostEqual(score, ref_score)

        # Test case IV: mu and sigma negative or zero

        # Test case IV.1
        psis = [np.exp(10)] * n_ids
        mu = 0
        sigma = 1

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # Test case IV.2
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # Test case IV.3
        psis = [np.exp(10)] * n_ids
        mu = -1
        sigma = 1

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # Test case IV.4
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = -1

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

    def test_compute_pointwise_ll(self):
        # Test case I.1:
        psis = np.arange(10)
        mu = 1
        sigma = 1
        a = (0 - mu) / sigma
        ref_scores = truncnorm.logpdf(
            psis, a=a, b=np.inf, loc=mu, scale=sigma)

        parameters = [mu, sigma]
        pw_scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(len(pw_scores), 10)
        self.assertAlmostEqual(np.sum(pw_scores), score)
        self.assertAlmostEqual(pw_scores[0], ref_scores[0])
        self.assertAlmostEqual(pw_scores[1], ref_scores[1])
        self.assertAlmostEqual(pw_scores[2], ref_scores[2])
        self.assertAlmostEqual(pw_scores[3], ref_scores[3])
        self.assertAlmostEqual(pw_scores[4], ref_scores[4])
        self.assertAlmostEqual(pw_scores[5], ref_scores[5])
        self.assertAlmostEqual(pw_scores[6], ref_scores[6])
        self.assertAlmostEqual(pw_scores[7], ref_scores[7])
        self.assertAlmostEqual(pw_scores[8], ref_scores[8])
        self.assertAlmostEqual(pw_scores[9], ref_scores[9])

        # Test case I.2:
        psis = np.linspace(3, 5, 10)
        mu = 2
        sigma = 4
        a = (0 - mu) / sigma
        ref_scores = truncnorm.logpdf(
            psis, a=a, b=np.inf, loc=mu, scale=sigma)

        parameters = [mu, sigma]
        pw_scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(len(pw_scores), 10)
        self.assertAlmostEqual(np.sum(pw_scores), score)
        self.assertAlmostEqual(pw_scores[0], ref_scores[0])
        self.assertAlmostEqual(pw_scores[1], ref_scores[1])
        self.assertAlmostEqual(pw_scores[2], ref_scores[2])
        self.assertAlmostEqual(pw_scores[3], ref_scores[3])
        self.assertAlmostEqual(pw_scores[4], ref_scores[4])
        self.assertAlmostEqual(pw_scores[5], ref_scores[5])
        self.assertAlmostEqual(pw_scores[6], ref_scores[6])
        self.assertAlmostEqual(pw_scores[7], ref_scores[7])
        self.assertAlmostEqual(pw_scores[8], ref_scores[8])
        self.assertAlmostEqual(pw_scores[9], ref_scores[9])

        # Test case IV: mu_log or sigma_log negative or zero

        # Test case IV.1
        psis = [np.exp(10)] * 3
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(scores[0], -np.inf)
        self.assertEqual(scores[1], -np.inf)
        self.assertEqual(scores[2], -np.inf)

        # Test case IV.2
        psis = [np.exp(10)] * 3
        mu = 1
        sigma = -10

        parameters = [mu] + [sigma]
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(scores[0], -np.inf)
        self.assertEqual(scores[1], -np.inf)
        self.assertEqual(scores[2], -np.inf)

    def test_compute_sensitivities(self):
        n_ids = 10

        # Test case I: psis = mu, sigma = 1
        # Sensitivities reduce to
        # dpsi = 0
        # dmu = - phi(mu) * nids / (1 - Phi(-mu))
        # dsigma = -n_ids + phi(mu) * mu * nids / (1 - Phi(-mu))

        # Test case I.1:
        psis = [1] * n_ids
        mu = 1
        sigma = 1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = 0
        ref_dmu = -norm.pdf(mu) * n_ids / (1 - norm.cdf(-mu))
        ref_dsigma = -n_ids + norm.pdf(mu) * mu * n_ids / (1 - norm.cdf(-mu))

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case I.2:
        psis = [10] * n_ids
        mu = 10
        sigma = 1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = 0
        ref_dmu = -norm.pdf(mu) * n_ids / (1 - norm.cdf(-mu))
        ref_dsigma = -n_ids + norm.pdf(mu) * mu * n_ids / (1 - norm.cdf(-mu))

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case II: psis != mu, sigma = 1
        # Sensitivities reduce to
        # dpsi = mu - psi
        # dmu = psi - mu - phi(mu) * nids / (1 - Phi(-mu))
        # dsigma = (psi - mu)^2 - phi(mu) * mu * nids / (1 - Phi(-mu))

        # Test case II.1:
        psis = np.array([1] * n_ids)
        mu = 10
        sigma = 1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = mu - psis[0]
        ref_dmu = \
            np.sum(psis - mu) \
            - norm.pdf(mu) * n_ids / (1 - norm.cdf(-mu))
        ref_dsigma = \
            - n_ids + np.sum((psis - mu)**2) \
            + norm.pdf(mu) * mu * n_ids / (1 - norm.cdf(-mu))

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case II.2:
        psis = np.array([7] * n_ids)
        mu = 5
        sigma = 1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = mu - psis[0]
        ref_dmu = \
            np.sum(psis - mu) \
            - norm.pdf(mu) * n_ids / (1 - norm.cdf(-mu))
        ref_dsigma = \
            - n_ids + np.sum((psis - mu)**2) \
            + norm.pdf(mu) * mu * n_ids / (1 - norm.cdf(-mu))

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case III: psis != mu, sigma != 1
        # Sensitivities reduce to
        # dpsi = (mu - psi) / sigma^2
        # dmu =
        #   (psi - mu - phi(mu/sigma) * nids / (1 - Phi(-mu/sigma))) / sigma
        # dsigma =
        #   -nids / sigma
        #   + (psi - mu)^2 / sigma^3
        #   + phi(mu) * mu * nids / (1 - Phi(-mu)) / sigma^2

        # Test case III.1:
        psis = np.array([1] * n_ids)
        mu = 10
        sigma = 2

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = (mu - psis[0]) / sigma**2
        ref_dmu = (
            np.sum(psis - mu) / sigma
            - norm.pdf(mu/sigma) * n_ids / (1 - norm.cdf(-mu/sigma))
            ) / sigma
        ref_dsigma = (
            -n_ids + np.sum((psis - mu)**2) / sigma**2
            + norm.pdf(mu/sigma) * mu / sigma * n_ids /
            (1 - norm.cdf(-mu/sigma))
        ) / sigma

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma, 5)

        # Test case III.2:
        psis = np.array([7] * n_ids)
        mu = 0.5
        sigma = 0.1

        # Compute ref scores
        parameters = [mu, sigma]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = (mu - psis[0]) / sigma**2
        ref_dmu = (
            np.sum(psis - mu) / sigma
            - norm.pdf(mu/sigma) * n_ids / (1 - norm.cdf(-mu/sigma))
            ) / sigma
        ref_dsigma = (
            -n_ids + np.sum((psis - mu)**2) / sigma**2
            + norm.pdf(mu/sigma) * mu / sigma * n_ids /
            (1 - norm.cdf(-mu/sigma))
        ) / sigma

        # Compute log-likelihood and sensitivities
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)

        self.assertAlmostEqual(score, ref_ll)
        self.assertEqual(len(sens), n_ids + 2)
        self.assertAlmostEqual(sens[0], ref_dpsi)
        self.assertAlmostEqual(sens[1], ref_dpsi)
        self.assertAlmostEqual(sens[2], ref_dpsi)
        self.assertAlmostEqual(sens[3], ref_dpsi)
        self.assertAlmostEqual(sens[4], ref_dpsi)
        self.assertAlmostEqual(sens[5], ref_dpsi)
        self.assertAlmostEqual(sens[6], ref_dpsi)
        self.assertAlmostEqual(sens[7], ref_dpsi)
        self.assertAlmostEqual(sens[8], ref_dpsi)
        self.assertAlmostEqual(sens[9], ref_dpsi)
        self.assertAlmostEqual(sens[10], ref_dmu)
        self.assertAlmostEqual(sens[11], ref_dsigma)

        # Test case IV: Compare gradients to numpy.gradient
        epsilon = 0.001
        n_parameters = n_ids + self.pop_model.n_parameters()
        parameters = np.ones(shape=n_parameters)
        ref_sens = []
        for index in range(n_parameters):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.pop_model.compute_log_likelihood(
                        low[n_ids:], low[:n_ids]),
                    self.pop_model.compute_log_likelihood(
                        parameters[n_ids:], parameters[:n_ids]),
                    self.pop_model.compute_log_likelihood(
                        high[n_ids:], high[:n_ids])],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities with hierarchical model
        _, sens = self.pop_model.compute_sensitivities(
            parameters[n_ids:], parameters[:n_ids])

        self.assertEqual(len(sens), 12)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])
        self.assertEqual(sens[9], ref_sens[9])
        self.assertAlmostEqual(sens[10], ref_sens[10], 5)
        self.assertAlmostEqual(sens[11], ref_sens[11], 5)

        # Test case V: mu_log or sigma_log negative or zero
        # Test case V.1
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)
        self.assertEqual(score, -np.inf)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)

        # Test case V.2
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = -10

        parameters = [mu] + [sigma]
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)
        self.assertEqual(score, -np.inf)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)

    def test_get_mean_and_std(self):
        # Test case I: sigma approx 0
        # Then:
        # mean approx mu
        # std approx 0

        # Test case I.1:
        mu = 1
        sigma = 0.00001
        parameters = [mu, sigma]
        mean, std = self.pop_model.get_mean_and_std(parameters)

        self.assertAlmostEqual(mean, mu)
        self.assertAlmostEqual(std, sigma)

        mu = 3
        sigma = 0.00001
        parameters = [mu, sigma]
        mean, std = self.pop_model.get_mean_and_std(parameters)

        self.assertAlmostEqual(mean, mu)
        self.assertAlmostEqual(std, sigma)

        # Test case II: mu = 0
        # Then:
        # mean = sigma * phi(0) * 2
        # std = sigma * sqrt(1 + (phi(0) * 2)**2)

        # Test case II.1:
        mu = 0
        sigma = 1

        # Compute references
        mean_ref = sigma * norm.pdf(0) * 2
        std_ref = sigma * np.sqrt(
            1 - (norm.pdf(0) * 2)**2)

        parameters = [mu, sigma]
        mean, std = self.pop_model.get_mean_and_std(parameters)

        self.assertEqual(mean, mean_ref)
        self.assertEqual(std, std_ref)

        # Test case II.2:
        mu = 0
        sigma = 10

        # Compute references
        mean_ref = sigma * norm.pdf(0) * 2
        std_ref = sigma * np.sqrt(
            1 - (norm.pdf(0) * 2)**2)

        parameters = [mu, sigma]
        mean, std = self.pop_model.get_mean_and_std(parameters)

        self.assertEqual(mean, mean_ref)
        self.assertEqual(std, std_ref)

        # Test case III: Negative mu and sigma
        mu = -1
        sigma = 1
        parameters = [mu, sigma]
        with self.assertRaisesRegex(ValueError, 'The parameters mu'):
            self.pop_model.get_mean_and_std(parameters)

        mu = 1
        sigma = -1
        parameters = [mu, sigma]
        with self.assertRaisesRegex(ValueError, 'The parameters mu'):
            self.pop_model.get_mean_and_std(parameters)

    def test_get_parameter_names(self):
        names = ['Mu', 'Sigma']

        self.assertEqual(self.pop_model.get_parameter_names(), names)

    def test_n_hierarchical_parameters(self):
        n_ids = 10
        n_hierarchical_params = self.pop_model.n_hierarchical_parameters(n_ids)

        self.assertEqual(len(n_hierarchical_params), 2)
        self.assertEqual(n_hierarchical_params[0], n_ids)
        self.assertEqual(n_hierarchical_params[1], 2)

    def test_n_parameters(self):
        self.assertEqual(self.pop_model.n_parameters(), 2)

    def test_sample(self):
        # Test I: sample size 1
        seed = np.random.default_rng(seed=42)
        parameters = [3, 2]
        sample = self.pop_model.sample(parameters, seed=seed)

        n_samples = 1
        self.assertEqual(sample.shape, (n_samples,))

        # Test II: sample size > 1
        seed = 1
        parameters = [3, 2]
        n_samples = 4
        sample = self.pop_model.sample(
            parameters, n_samples=n_samples, seed=seed)

        self.assertEqual(
            sample.shape, (n_samples,))

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.pop_model.sample(parameters)

        # Negative std
        parameters = [1, -1]

        with self.assertRaisesRegex(
                ValueError, 'A truncated Gaussian distribution'):
            self.pop_model.sample(parameters)

    def test_set_parameter_names(self):
        # Test some name
        names = ['test', 'name']
        self.pop_model.set_parameter_names(names)

        self.assertEqual(
            self.pop_model.get_parameter_names(), names)

        # Set back to default name
        self.pop_model.set_parameter_names(None)
        names = self.pop_model.get_parameter_names()

        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Mu')
        self.assertEqual(names[1], 'Sigma')

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'two', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.pop_model.set_parameter_names(names)


if __name__ == '__main__':
    unittest.main()
