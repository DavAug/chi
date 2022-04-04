#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

import chi


class TestPopulationFilter(unittest.TestCase):
    """
    Tests the chi.PopulationFilter class.
    """

    @classmethod
    def setUpClass(cls):
        observations = np.empty((2, 3, 4))
        cls.filter = chi.PopulationFilter(observations)

    def test_bad_instantiation(self):
        # Observations are not 3 dimensional
        observations = np.empty((2, 3))
        with self.assertRaisesRegex(ValueError, 'The observations'):
            chi.PopulationFilter(observations)

    def test_compute_log_likelihood(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.filter.compute_log_likelihood('some input')

    def test_compute_sensitivities(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.filter.compute_sensitivities('some input')

    def test_n_observables(self):
        self.assertEqual(self.filter.n_observables(), 3)

    def test_n_times(self):
        self.assertEqual(self.filter.n_times(), 4)

    def test_sort_times(self):
        order = np.array([0, 1, 2, 3])
        self.filter.sort_times(order)

    def test_sort_times_bad_input(self):
        # Wrong length
        order = np.array([0, 1, 2])
        with self.assertRaisesRegex(ValueError, 'Order has to be of'):
            self.filter.sort_times(order)

        # Non-unique entries
        order = np.array([0, 1, 2, 2])
        with self.assertRaisesRegex(ValueError, 'Order has to contain'):
            self.filter.sort_times(order)


class TestGaussianPopulationFilter(unittest.TestCase):
    """
    Tests the chi.GaussianPopulationFilter class.
    """
    @classmethod
    def setUpClass(cls):

        # Test case I: no missing values
        observations = np.ones((2, 3, 4))
        cls.filter1 = chi.GaussianPopulationFilter(observations)

        # Test case II: missing values
        observations = np.array([
            [[1, 2, np.nan, 5], [0.1, 2, 4, 3], [np.nan, 3, 2, np.nan]],
            [[0, 20, 13, -4], [21, 0.2, 8, 4], [0.1, 0.2, 0.3, 0.4]]])
        cls.filter2 = chi.GaussianPopulationFilter(observations)

    def test_compute_log_likelihood(self):
        # Test case I:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter1.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case II:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter2.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

    def test_compute_sensitivities(self):
        # Test case I: Finite difference
        epsilon = 0.00001
        sim_obs = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.5 * np.ones(shape=(1, 3, 4))])
        ref_sens = []
        ref_score = self.filter1.compute_log_likelihood(sim_obs)
        for index in range(len(sim_obs)):
            # Construct parameter grid
            low = sim_obs.copy()
            low[index, 0, 0] -= epsilon
            high = sim_obs.copy()
            high[index, 0, 0] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.filter1.compute_log_likelihood(low),
                    ref_score,
                    self.filter1.compute_log_likelihood(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.filter1.compute_sensitivities(sim_obs)

        self.assertEqual(score, ref_score)
        self.assertEqual(sens.shape, (5, 3, 4))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0])
        self.assertAlmostEqual(sens[1, 0, 0], ref_sens[1])
        self.assertAlmostEqual(sens[2, 0, 0], ref_sens[2])
        self.assertAlmostEqual(sens[3, 0, 0], ref_sens[3])
        self.assertAlmostEqual(sens[4, 0, 0], ref_sens[4])

        # Test case II: Finite difference
        epsilon = 0.00001
        sim_obs = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.5 * np.ones(shape=(1, 3, 4))])
        ref_sens = []
        ref_score = self.filter2.compute_log_likelihood(sim_obs)
        for index in range(len(sim_obs)):
            # Construct parameter grid
            low = sim_obs.copy()
            low[index, 0, 0] -= epsilon
            high = sim_obs.copy()
            high[index, 0, 0] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.filter2.compute_log_likelihood(low),
                    ref_score,
                    self.filter2.compute_log_likelihood(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.filter2.compute_sensitivities(sim_obs)

        self.assertEqual(score, ref_score)
        self.assertEqual(sens.shape, (5, 3, 4))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0])
        self.assertAlmostEqual(sens[1, 0, 0], ref_sens[1])
        self.assertAlmostEqual(sens[2, 0, 0], ref_sens[2])
        self.assertAlmostEqual(sens[3, 0, 0], ref_sens[3])
        self.assertAlmostEqual(sens[4, 0, 0], ref_sens[4])


if __name__ == '__main__':
    unittest.main()
