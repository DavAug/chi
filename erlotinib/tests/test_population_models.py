#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

import erlotinib as erlo


class TestHeterogeneousModel(unittest.TestCase):
    """
    Tests the erlotinib.HeterogeneousModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = erlo.HeterogeneousModel()

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

        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'some name')

        # Set to default
        self.pop_model.set_parameter_names(None)
        name = self.pop_model.get_parameter_names()
        self.assertIsNone(name)

    def test_set_parameter_names_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'Length of names has to be 1'):
            self.pop_model.set_parameter_names('some params')


class TestLogNormalModel(unittest.TestCase):
    """
    Tests the erlotinib.LogNormalModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = erlo.LogNormalModel()

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
        self.assertEqual(score, ref_score)

        # Test case I.2:
        psis = [1] * n_ids
        mu_log = 5
        sigma_log = 1
        ref_score = -n_ids * (np.log(2 * np.pi) + mu_log**2) / 2

        parameters = [mu_log] + [sigma_log]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, ref_score)

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
        self.assertEqual(score, ref_score)

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
        self.assertEqual(score, ref_score)

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
        self.assertEqual(score, ref_score)

        # Test case III.2
        psis = [np.exp(3)] * n_ids
        mu_log = 3
        sigma_log = 1
        ref_score = -n_ids * (3 + np.log(2 * np.pi) / 2)

        parameters = [mu_log] + [sigma_log]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, ref_score)

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
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(len(scores), 10)
        self.assertEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 10))

        # Test case I.2:
        n_ids = 6
        psis = [1] * n_ids
        mu_log = 5
        sigma_log = 1
        ref_score = -n_ids * (np.log(2 * np.pi) + mu_log**2) / 2

        parameters = [mu_log] + [sigma_log]
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(len(scores), 6)
        self.assertEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 6))

        # Test case II: psis = 1.
        # Score reduces to
        # -n_ids * log(sigma_log) - n_ids * log(2 * pi) / 2
        # - n_ids * mu_log^2 / (2 * sigma_log^2)

        # Test case II.1:
        n_ids = 10
        psis = [1] * n_ids
        mu_log = 1
        sigma_log = np.exp(2)
        ref_score = \
            -n_ids * (
                np.log(2 * np.pi * sigma_log**2)
                + mu_log**2 / sigma_log**2) / 2

        parameters = [mu_log] + [sigma_log]
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(len(scores), 10)
        self.assertEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 10))

        # Test case II.2:
        psis = [1] * n_ids
        mu_log = 3
        sigma_log = np.exp(3)
        ref_score = \
            -n_ids * (
                np.log(2 * np.pi * sigma_log**2)
                + mu_log**2 / sigma_log**2) / 2

        parameters = [mu_log] + [sigma_log]
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(len(scores), 10)
        self.assertEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 10))

        # Test case III: Different psis
        psis = [1, 2]
        mu = 1
        sigma = 1

        parameters = [mu] + [sigma]
        ref_score = self.pop_model.compute_log_likelihood(parameters, psis)
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(np.sum(scores), ref_score)
        self.assertNotEqual(scores[0], scores[1])

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
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(len(scores), 10)
        self.assertAlmostEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 10))

        # Test case III.2
        psis = [np.exp(3)] * n_ids
        mu_log = 3
        sigma_log = 1
        ref_score = -n_ids * (3 + np.log(2 * np.pi) / 2)

        parameters = [mu_log] + [sigma_log]
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(len(scores), 10)
        self.assertEqual(np.sum(scores), ref_score)
        self.assertTrue(np.allclose(scores, ref_score / 10))

        # Test case IV: mu_log or sigma_log negative or zero

        # Test case IV.1
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(scores[0], -np.inf)
        self.assertEqual(scores[1], -np.inf)
        self.assertEqual(scores[2], -np.inf)

        # Test case IV.2
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = -10

        parameters = [mu] + [sigma]
        scores = self.pop_model.compute_pointwise_ll(parameters, psis)
        self.assertEqual(scores[0], -np.inf)
        self.assertEqual(scores[1], -np.inf)
        self.assertEqual(scores[2], -np.inf)

    def test_compute_sensitivities(self):
        # Hard to test exactly, but at least test some edge cases where
        # loglikelihood is straightforward to compute analytically

        n_ids = 10

        # Test case I: psis = 1, sigma_log = 1
        # Sensitivities reduce to
        # dpsi = -1 + mu_log
        # dmu = - mu_log * nids
        # dsigma = -(1 + mu_log) * nids

        # Test case I.1:
        psis = [1] * n_ids
        mu_log = 1
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = -1 + mu_log
        ref_dmu = -mu_log * n_ids
        ref_dsigma = -(1 + mu_log) * n_ids

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
        ref_dsigma = -(1 + mu_log) * n_ids

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
        # dsigma = -(1 + mu_log / var_log) / std_log * nids

        # Test case II.1:
        psis = [1] * n_ids
        mu_log = 1
        sigma_log = np.exp(2)

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psis)
        ref_dpsi = -1 + mu_log / sigma_log**2
        ref_dmu = -mu_log / sigma_log**2 * n_ids
        ref_dsigma = -(1 + mu_log / sigma_log**2) / sigma_log * n_ids

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
        ref_dsigma = -(1 + mu_log / sigma_log**2) / sigma_log * n_ids

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
        # dsigma = (log psi - mu_log - 1) * nids

        # Test case III.1
        psi = [np.exp(4)] * n_ids
        mu_log = 1
        sigma_log = 1

        # Compute ref scores
        parameters = [mu_log] + [sigma_log]
        ref_ll = self.pop_model.compute_log_likelihood(parameters, psi)
        ref_dpsi = (-1 + mu_log - np.log(psi[0])) / psi[0]
        ref_dmu = (np.log(psi[0]) - mu_log) * n_ids
        ref_dsigma = (np.log(psi[0]) - mu_log - 1) * n_ids

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
        ref_dsigma = (np.log(psi[0]) - mu_log - 1) * n_ids

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

        # Test case IV: mu_log or sigma_log negative or zero

        # Test case IV.1
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        score, sens = self.pop_model.compute_sensitivities(parameters, psis)
        self.assertEqual(score, -np.inf)
        self.assertEqual(sens[0], np.inf)
        self.assertEqual(sens[1], np.inf)
        self.assertEqual(sens[2], np.inf)

        # Test case IV.2
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

    def test_get_parameter_names(self):
        names = ['Mean log', 'Std. log']

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
        self.assertEqual(sample.shape, (n_samples,))

        # Test II: sample size > 1
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
        self.assertEqual(names[0], 'Mean log')
        self.assertEqual(names[1], 'Std. log')

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'two', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.pop_model.set_parameter_names(names)


class TestPooledModel(unittest.TestCase):
    """
    Tests the erlotinib.PooledModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = erlo.PooledModel()

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
    Tests the erlotinib.PopulationModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.pop_model = erlo.PopulationModel()

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

    def test_n_hierarchical_parameters(self):
        n_ids = 'some ids'
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.n_hierarchical_parameters(n_ids)

    def test_n_parameters(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.n_parameters()

    def test_sample(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.sample('some values')

    def test_set_parameter_names(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.set_parameter_names('some name')


class TestReducedPopulationModel(unittest.TestCase):
    """
    Tests the erlotinib.ReducedPopulationModel class.
    """

    @classmethod
    def setUpClass(cls):
        pop_model = erlo.LogNormalModel()
        cls.pop_model = erlo.ReducedPopulationModel(pop_model)

    def test_bad_instantiation(self):
        model = 'Bad type'
        with self.assertRaisesRegex(TypeError, 'The population model'):
            erlo.ReducedPopulationModel(model)

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
        self.assertIsInstance(pop_model, erlo.PopulationModel)

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
        # Test case I: fix some parameters
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


if __name__ == '__main__':
    unittest.main()
