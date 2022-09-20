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


class TestComposedPopulationFilter(unittest.TestCase):
    """
    Tests the chi.ComposedPopulationFilter class.
    """

    @classmethod
    def setUpClass(cls):
        observations = np.array([
            [[1, 2, np.nan, 5], [0.1, 2, 4, 3], [np.nan, 3, 2, np.nan]],
            [[0, 20, 13, -4], [21, 0.2, 8, 4], [0.1, 0.2, 0.3, 0.4]]])
        cls.filter1 = chi.GaussianFilter(observations)

        observations = np.array([
            [[1, 2], [0.1, 2], [np.nan, 3]],
            [[0.2, 20], [21, 0.2], [0.1, 0.2]],
            [[0.2, 20], [21, 0.2], [0.1, 0.2]]])
        cls.filter2 = chi.LogNormalFilter(observations)

        cls.filter = chi.ComposedPopulationFilter([cls.filter1, cls.filter2])

    def test_bad_instantiation(self):
        # Wrong type
        filters = ['filter 1', 'filter 2']
        with self.assertRaisesRegex(TypeError, 'All filters have'):
            chi.ComposedPopulationFilter(filters)

        # Number of modelled observations doesn't match
        observations = np.ones((2, 5, 4))
        filter3 = chi.GaussianFilter(observations)
        with self.assertRaisesRegex(ValueError, 'All filters need'):
            chi.ComposedPopulationFilter([self.filter1, self.filter2, filter3])

    def test_compute_log_likelihood(self):
        # Test case I: valid input
        sim_obs1 = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.5 * np.ones(shape=(1, 3, 4))])
        sim_obs2 = np.vstack([
            np.ones(shape=(1, 3, 2)),
            2 * np.ones(shape=(1, 3, 2)),
            0.3 * np.ones(shape=(1, 3, 2)),
            0.9 * np.ones(shape=(1, 3, 2)),
            0.5 * np.ones(shape=(1, 3, 2))])
        sim_obs = np.vstack([
            np.ones(shape=(1, 3, 6)),
            2 * np.ones(shape=(1, 3, 6)),
            0.3 * np.ones(shape=(1, 3, 6)),
            0.9 * np.ones(shape=(1, 3, 6)),
            0.5 * np.ones(shape=(1, 3, 6))])
        score = self.filter.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))
        ref_score = \
            self.filter1.compute_log_likelihood(sim_obs1) \
            + self.filter2.compute_log_likelihood(sim_obs2)
        self.assertEqual(score, ref_score)

        # Compare score with sorted times
        sim_obs_unordered = np.vstack([
            np.broadcast_to(
                np.array([1, 1, 1, 1, 2, 2])[np.newaxis, np.newaxis, :],
                (1, 3, 6)),
            np.broadcast_to(
                np.array([3, 3, 3, 3, 4, 4])[np.newaxis, np.newaxis, :],
                (1, 3, 6))])
        order = [0, 4, 1, 5, 2, 3]
        sim_obs_ordered = np.vstack([
            np.broadcast_to(
                np.array([1, 2] * 2 + [1, 1])[np.newaxis, np.newaxis, :],
                (1, 3, 6)),
            np.broadcast_to(
                np.array([3, 4] * 2 + [3, 3])[np.newaxis, np.newaxis, :],
                (1, 3, 6))])
        score1 = self.filter.compute_log_likelihood(sim_obs_unordered)
        self.filter.sort_times(order)
        score2 = self.filter.compute_log_likelihood(sim_obs_ordered)
        self.assertFalse(np.isinf(score1))
        self.assertFalse(np.isinf(score2))
        self.assertEqual(score1, score2)

        # Reset order
        self.filter._time_order = None
        self.filter._time_filter_order = None

    def test_compute_sensitivities(self):
        # Test case I: Finite difference
        epsilon = 0.000001
        n_times = 6
        sim_obs = np.vstack([
            np.broadcast_to(
                np.array([1., 1., 1., 1., 2., 2.])[np.newaxis, np.newaxis, :],
                (1, 3, n_times)),
            np.broadcast_to(
                np.array([3., 3., 3., 3., 4., 4.])[np.newaxis, np.newaxis, :],
                (1, 3, n_times))])
        ref_sens = []
        ref_score = self.filter.compute_log_likelihood(sim_obs)
        for index in range(n_times):
            # Construct parameter grid
            low = sim_obs.copy()
            low[0, 0, index] -= epsilon
            high = sim_obs.copy()
            high[0, 0, index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.filter.compute_log_likelihood(low),
                    ref_score,
                    self.filter.compute_log_likelihood(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.filter.compute_sensitivities(sim_obs)

        self.assertEqual(score, ref_score)
        self.assertEqual(sens.shape, (2, 3, 6))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0])
        self.assertAlmostEqual(sens[0, 0, 1], ref_sens[1])
        self.assertAlmostEqual(sens[0, 0, 2], ref_sens[2])
        self.assertAlmostEqual(sens[0, 0, 3], ref_sens[3])
        self.assertAlmostEqual(sens[0, 0, 4], ref_sens[4])
        self.assertAlmostEqual(sens[0, 0, 5], ref_sens[5])

        # Test case II: Change order of times
        epsilon = 0.000001
        n_times = 6
        ref_sens = []
        ref_score = self.filter.compute_log_likelihood(sim_obs)
        for index in range(n_times):
            # Construct parameter grid
            low = sim_obs.copy()
            low[0, 0, index] -= epsilon
            high = sim_obs.copy()
            high[0, 0, index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.filter.compute_log_likelihood(low),
                    ref_score,
                    self.filter.compute_log_likelihood(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        # (mess up the order of the observations and order them internally)
        sim_obs_unordered = np.vstack([
            np.broadcast_to(
                np.array([1., 2., 1., 2., 1., 1.])[np.newaxis, np.newaxis, :],
                (1, 3, n_times)),
            np.broadcast_to(
                np.array([3., 4., 3., 4., 3., 3.])[np.newaxis, np.newaxis, :],
                (1, 3, n_times))])
        order = [0, 4, 2, 5, 1, 3]
        self.filter.sort_times(order)
        score, sens = self.filter.compute_sensitivities(sim_obs_unordered)

        self.assertEqual(score, ref_score)
        self.assertEqual(sens.shape, (2, 3, 6))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0])
        self.assertAlmostEqual(sens[0, 0, 1], ref_sens[4])
        self.assertAlmostEqual(sens[0, 0, 2], ref_sens[2])
        self.assertAlmostEqual(sens[0, 0, 3], ref_sens[5])
        self.assertAlmostEqual(sens[0, 0, 4], ref_sens[1])
        self.assertAlmostEqual(sens[0, 0, 5], ref_sens[3])

        # Reset order
        self.filter._time_order = None
        self.filter._time_filter_order = None

    def test_n_observables(self):
        self.assertEqual(self.filter.n_observables(), 3)
        self.assertEqual(
            self.filter.n_observables(), self.filter1.n_observables())
        self.assertEqual(
            self.filter.n_observables(), self.filter2.n_observables())

    def test_n_times(self):
        self.assertEqual(self.filter.n_times(), 6)
        n_times = self.filter1.n_times() + self.filter2.n_times()
        self.assertEqual(self.filter.n_times(), n_times)

    def test_sort_times(self):
        order = np.array([5, 4, 3, 2, 1, 0])
        self.filter.sort_times(order)

        order = np.array([0, 1, 2, 3, 4, 5])
        self.filter.sort_times(order)

        # Reset order
        self.filter._time_order = None
        self.filter._time_filter_order = None

    def test_sort_times_bad_input(self):
        # Wrong length
        order = np.array([0, 1, 2])
        with self.assertRaisesRegex(ValueError, 'Order has to be of'):
            self.filter.sort_times(order)

        # Non-unique entries
        order = np.array([0, 1, 2, 2, 1, 1])
        with self.assertRaisesRegex(ValueError, 'Order has to contain'):
            self.filter.sort_times(order)


class TestGaussianFilter(unittest.TestCase):
    """
    Tests the chi.GaussianFilter class.
    """
    @classmethod
    def setUpClass(cls):

        # Test case I: no missing values
        observations = np.ones((2, 3, 4))
        cls.filter1 = chi.GaussianFilter(observations)

        # Test case II: missing values
        observations = np.array([
            [[1, 2, np.nan, 5], [0.1, 2, 4, 3], [np.nan, 3, 2, np.nan]],
            [[0, 20, 13, -4], [21, 0.2, 8, 4], [0.1, 0.2, 0.3, 0.4]]])
        cls.filter2 = chi.GaussianFilter(observations)

    def test_compute_log_likelihood(self):
        # Test case I:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter1.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case II:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter2.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case III: return infinitty if score is masked
        sim_obs = np.ones((10, 3, 4))
        score = self.filter2.compute_log_likelihood(sim_obs)
        self.assertTrue(np.isinf(score))

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

        # Test case III: return infinitty if score is masked
        sim_obs = np.ones((10, 3, 4))
        score, sens = self.filter2.compute_sensitivities(sim_obs)
        self.assertTrue(np.isinf(score))
        self.assertEqual(sens.shape, (10, 3, 4))


class TestGaussianKDEFilter(unittest.TestCase):
    """
    Tests the chi.GaussianKDEFilter class.
    """
    @classmethod
    def setUpClass(cls):

        # Test case I: no missing values, default bandwidth
        observations = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4))])
        cls.filter1 = chi.GaussianKDEFilter(observations)

        # Test case III: missing values, default bandwidth
        observations = np.array([
            [[1, 2, np.nan, 5], [0.1, 2, 4, 3], [np.nan, 3, 2, np.nan]],
            [[0, 20, 13, -4], [21, 0.2, 8, 4], [0.1, 0.2, 0.3, 0.4]],
            [[4, 1, 1, -3], [2, 0.3, 4, 2], [1, 0.7, 2, 1]]])
        cls.filter3 = chi.GaussianKDEFilter(observations)

    def test_compute_log_likelihood(self):
        # Test case I:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter1.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case III:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter3.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case V: infinite score for masked score
        sim_obs = np.full(shape=(10, 3, 4), fill_value=np.nan)
        score = self.filter3.compute_log_likelihood(sim_obs)
        self.assertTrue(np.isinf(score))

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

        # Test case III
        epsilon = 0.00001
        sim_obs = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.5 * np.ones(shape=(1, 3, 4))])
        ref_sens = []
        ref_score = self.filter3.compute_log_likelihood(sim_obs)
        for index in range(len(sim_obs)):
            # Construct parameter grid
            low = sim_obs.copy()
            low[index, 0, 0] -= epsilon
            high = sim_obs.copy()
            high[index, 0, 0] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.filter3.compute_log_likelihood(low),
                    ref_score,
                    self.filter3.compute_log_likelihood(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.filter3.compute_sensitivities(sim_obs)

        self.assertEqual(score, ref_score)
        self.assertEqual(sens.shape, (5, 3, 4))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0])
        self.assertAlmostEqual(sens[1, 0, 0], ref_sens[1])
        self.assertAlmostEqual(sens[2, 0, 0], ref_sens[2])
        self.assertAlmostEqual(sens[3, 0, 0], ref_sens[3])
        self.assertAlmostEqual(sens[4, 0, 0], ref_sens[4])

        # Test case V: infinite score for masked score
        sim_obs = np.full(shape=(10, 3, 4), fill_value=np.nan)
        score, sens = self.filter3.compute_sensitivities(sim_obs)
        self.assertTrue(np.isinf(score))
        self.assertEqual(sens.shape, (10, 3, 4))


class TestGaussianMixtureFilter(unittest.TestCase):
    """
    Tests the chi.GaussianMixtureFilter class.
    """
    @classmethod
    def setUpClass(cls):

        # Test case I: no missing values, default bandwidth
        observations = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4))])
        cls.filter1 = chi.GaussianMixtureFilter(observations)

        # Test case III: missing values, default bandwidth
        observations = np.array([
            [[1, 2, np.nan, 5], [0.1, 2, 4, 3], [np.nan, 3, 2, np.nan]],
            [[0, 20, 13, -4], [21, 0.2, 8, 4], [0.1, 0.2, 0.3, 0.4]],
            [[4, 1, 1, -3], [2, 0.3, 4, 2], [1, 0.7, 2, 1]]])
        cls.filter3 = chi.GaussianMixtureFilter(observations)

    def test_bad_instantiation(self):
        observations = np.array([
            [[1, 2, np.nan, 5], [0.1, 2, 4, 3], [np.nan, 3, 2, np.nan]],
            [[0, 20, 13, -4], [21, 0.2, 8, 4], [0.1, 0.2, 0.3, 0.4]],
            [[4, 1, 1, -3], [2, 0.3, 4, 2], [1, 0.7, 2, 1]]])
        with self.assertRaisesRegex(ValueError, 'Invalid number of kernels.'):
            chi.GaussianMixtureFilter(observations, n_kernels=1)

    def test_compute_log_likelihood(self):
        # Test case I:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter1.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case II: More than 2 kernels
        obs = self.filter1._observations[0, 0, :, :, :]
        f = chi.GaussianMixtureFilter(obs, n_kernels=5)
        score = f.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case III:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter3.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case V: infinite score for masked score
        sim_obs = np.full(shape=(10, 3, 4), fill_value=np.nan)
        score = self.filter3.compute_log_likelihood(sim_obs)
        self.assertTrue(np.isinf(score))

        # Raises error when number of observations is not a multiple of the
        # number of kernels
        sim_obs = np.arange(11 * 3 * 4).reshape(11, 3, 4)
        with self.assertRaisesRegex(ValueError, 'Invalid simulated_obs.'):
            self.filter3.compute_log_likelihood(sim_obs)

    def test_compute_sensitivities(self):
        # Test case I: Finite difference
        epsilon = 0.00001
        sim_obs = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.91 * np.ones(shape=(1, 3, 4)),
            0.92 * np.ones(shape=(1, 3, 4)),
            0.39 * np.ones(shape=(1, 3, 4)),
            0.94 * np.ones(shape=(1, 3, 4)),
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
        self.assertEqual(sens.shape, (10, 3, 4))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0])
        self.assertAlmostEqual(sens[1, 0, 0], ref_sens[1])
        self.assertAlmostEqual(sens[2, 0, 0], ref_sens[2])
        self.assertAlmostEqual(sens[3, 0, 0], ref_sens[3])
        self.assertAlmostEqual(sens[4, 0, 0], ref_sens[4])

        # Test case II: More than 2 kernels
        obs = self.filter1._observations[0, 0, :, :, :]
        f = chi.GaussianMixtureFilter(obs, n_kernels=5)
        epsilon = 0.00001
        sim_obs = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            5 * np.ones(shape=(1, 3, 4)),
            0.1 * np.ones(shape=(1, 3, 4)),
            3.62 * np.ones(shape=(1, 3, 4)),
            0.19 * np.ones(shape=(1, 3, 4)),
            2.94 * np.ones(shape=(1, 3, 4)),
            0.1 * np.ones(shape=(1, 3, 4))])
        ref_sens = []
        ref_score = f.compute_log_likelihood(sim_obs)
        for index in range(len(sim_obs)):
            # Construct parameter grid
            low = sim_obs.copy()
            low[index, 0, 0] -= epsilon
            high = sim_obs.copy()
            high[index, 0, 0] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    f.compute_log_likelihood(low),
                    ref_score,
                    f.compute_log_likelihood(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = f.compute_sensitivities(sim_obs)

        self.assertEqual(score, ref_score)
        self.assertEqual(sens.shape, (10, 3, 4))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0])
        self.assertAlmostEqual(sens[1, 0, 0], ref_sens[1])
        self.assertAlmostEqual(sens[2, 0, 0], ref_sens[2], 5)
        self.assertAlmostEqual(sens[3, 0, 0], ref_sens[3])
        self.assertAlmostEqual(sens[4, 0, 0], ref_sens[4])

        # Test case III
        epsilon = 0.00001
        sim_obs = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.91 * np.ones(shape=(1, 3, 4)),
            0.92 * np.ones(shape=(1, 3, 4)),
            0.39 * np.ones(shape=(1, 3, 4)),
            0.94 * np.ones(shape=(1, 3, 4)),
            0.5 * np.ones(shape=(1, 3, 4))])
        ref_sens = []
        ref_score = self.filter3.compute_log_likelihood(sim_obs)
        for index in range(len(sim_obs)):
            # Construct parameter grid
            low = sim_obs.copy()
            low[index, 0, 0] -= epsilon
            high = sim_obs.copy()
            high[index, 0, 0] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.filter3.compute_log_likelihood(low),
                    ref_score,
                    self.filter3.compute_log_likelihood(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.filter3.compute_sensitivities(sim_obs)

        self.assertEqual(score, ref_score)
        self.assertEqual(sens.shape, (10, 3, 4))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0], 5)
        self.assertAlmostEqual(sens[1, 0, 0], ref_sens[1], 5)
        self.assertAlmostEqual(sens[2, 0, 0], ref_sens[2], 5)
        self.assertAlmostEqual(sens[3, 0, 0], ref_sens[3], 5)
        self.assertAlmostEqual(sens[4, 0, 0], ref_sens[4], 5)

        # Test case V: infinite score for masked score
        sim_obs = np.full(shape=(10, 3, 4), fill_value=np.nan)
        score, sens = self.filter3.compute_sensitivities(sim_obs)
        self.assertTrue(np.isinf(score))
        self.assertEqual(sens.shape, (10, 3, 4))

        # Raises error when number of observations is not a multiple of the
        # number of kernels
        sim_obs = np.arange(11 * 3 * 4).reshape(11, 3, 4)
        with self.assertRaisesRegex(ValueError, 'Invalid simulated_obs.'):
            self.filter3.compute_sensitivities(sim_obs)


class TestLogNormalFilter(unittest.TestCase):
    """
    Tests the chi.LogNormalFilter class.
    """
    @classmethod
    def setUpClass(cls):

        # Test case I: no missing values
        observations = np.ones((2, 3, 4))
        cls.filter1 = chi.LogNormalFilter(observations)

        # Test case II: missing values
        observations = np.array([
            [[1, 2, np.nan, 5], [0.1, 2, 4, 3], [np.nan, 3, 2, np.nan]],
            [[0.2, 20, 13, 4], [21, 0.2, 8, 4], [0.1, 0.2, 0.3, 0.4]]])
        cls.filter2 = chi.LogNormalFilter(observations)

    def test_compute_log_likelihood(self):
        # Test case I:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter1.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case II:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4)
        score = self.filter2.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case III: return infinitty if score is masked
        sim_obs = np.ones((10, 3, 4))
        score = self.filter2.compute_log_likelihood(sim_obs)
        self.assertTrue(np.isinf(score))

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

        # Test case III: return infinitty if score is masked
        sim_obs = np.ones((10, 3, 4))
        score, sens = self.filter2.compute_sensitivities(sim_obs)
        self.assertTrue(np.isinf(score))
        self.assertEqual(sens.shape, (10, 3, 4))


class TestLogNormalKDEFilter(unittest.TestCase):
    """
    Tests the chi.LogNormalKDEFilter class.
    """
    @classmethod
    def setUpClass(cls):
        # Test case I: no missing values, default bandwidth
        observations = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4))])
        cls.filter1 = chi.LogNormalKDEFilter(observations)

        # Test case III: missing values, default bandwidth
        observations = np.array([
            [[1, 2, np.nan, 5], [0.1, 2, 4, 3], [np.nan, 3, 2, np.nan]],
            [[0.1, 20, 13, 4], [21, 0.2, 8, 4], [0.1, 0.2, 0.3, 0.4]],
            [[4, 1, 1, 3], [2, 0.3, 4, 2], [1, 0.7, 2, 1]]])
        cls.filter3 = chi.LogNormalKDEFilter(observations)

    def test_compute_log_likelihood(self):
        # Test case I:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4) + 1
        score = self.filter1.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case III:
        sim_obs = np.arange(10 * 3 * 4).reshape(10, 3, 4) + 1
        score = self.filter3.compute_log_likelihood(sim_obs)
        self.assertFalse(np.isinf(score))

        # Test case V: infinite score for masked score
        sim_obs = np.full(shape=(10, 3, 4), fill_value=np.nan)
        score = self.filter3.compute_log_likelihood(sim_obs)
        self.assertTrue(np.isinf(score))

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

        # Test case III
        epsilon = 0.00001
        sim_obs = np.vstack([
            np.ones(shape=(1, 3, 4)),
            2 * np.ones(shape=(1, 3, 4)),
            0.3 * np.ones(shape=(1, 3, 4)),
            0.9 * np.ones(shape=(1, 3, 4)),
            0.5 * np.ones(shape=(1, 3, 4))])
        ref_sens = []
        ref_score = self.filter3.compute_log_likelihood(sim_obs)
        for index in range(len(sim_obs)):
            # Construct parameter grid
            low = sim_obs.copy()
            low[index, 0, 0] -= epsilon
            high = sim_obs.copy()
            high[index, 0, 0] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.filter3.compute_log_likelihood(low),
                    ref_score,
                    self.filter3.compute_log_likelihood(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.filter3.compute_sensitivities(sim_obs)

        self.assertEqual(score, ref_score)
        self.assertEqual(sens.shape, (5, 3, 4))
        self.assertAlmostEqual(sens[0, 0, 0], ref_sens[0])
        self.assertAlmostEqual(sens[1, 0, 0], ref_sens[1])
        self.assertAlmostEqual(sens[2, 0, 0], ref_sens[2])
        self.assertAlmostEqual(sens[3, 0, 0], ref_sens[3])
        self.assertAlmostEqual(sens[4, 0, 0], ref_sens[4])

        # Test case V: infinite score for masked score
        sim_obs = np.full(shape=(10, 3, 4), fill_value=np.nan)
        score, sens = self.filter3.compute_sensitivities(sim_obs)
        self.assertTrue(np.isinf(score))
        self.assertEqual(sens.shape, (10, 3, 4))


if __name__ == '__main__':
    unittest.main()
