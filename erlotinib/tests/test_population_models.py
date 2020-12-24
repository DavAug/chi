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
        cls.n_ids = 10
        cls.pop_model = erlo.HeterogeneousModel(cls.n_ids)

    def test_call(self):
        # For efficiency the input is actually not checked, and 0 is returned
        # regardless
        self.assertEqual(self.pop_model('some values'), 0)

    def test_get_top_parameter_names(self):
        self.assertIsNone(self.pop_model.get_top_parameter_names())

    def test_n_bottom_parameters(self):
        n_individual_input_params = 10
        self.assertEqual(
            self.pop_model.n_bottom_parameters(),
            n_individual_input_params)

    def test_n_parameters(self):
        n_population_params = 10
        self.assertEqual(self.pop_model.n_parameters(), n_population_params)

    def test_n_top_parameters(self):
        n_population_params = 0
        self.assertEqual(
            self.pop_model.n_top_parameters(), n_population_params)

    def test_sample(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.sample('some params')

    def test_set_top_parameter_names(self):
        with self.assertRaisesRegex(ValueError, 'A heterogeneous population'):
            self.pop_model.set_top_parameter_names('some params')


class TestLogNormalModel(unittest.TestCase):
    """
    Tests the erlotinib.LogNormalModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.n_ids = 10
        cls.pop_model = erlo.LogNormalModel(cls.n_ids)

    def test_call(self):
        # Hard to test exactly, but at least test some edge cases where
        # loglikelihood is straightforward to compute analytically

        # Test case I: psis = 1, sigma_log = 1
        # Score reduces to -n_ids * mu_log^2 / 2

        # Test case I.1:
        psis = [1] * self.n_ids
        mu_log = 1
        var_log = 1
        score = -self.n_ids * mu_log**2 / 2  # mu_log = -5

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertEqual(transformed[1], var_log)

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), score)

        # Test case I.2:
        psis = [1] * self.n_ids
        mu_log = 5
        var_log = 1
        score = -self.n_ids * mu_log**2 / 2  # mu_log = -125

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertEqual(transformed[1], var_log)

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), score)

        # Test case II: psis = 1.
        # Score reduces to
        # -n_ids * log(sigma_log) -n_ids * mu_log^2 / (2 * sigma_log^2)

        # Test case II.1:
        psis = [1] * self.n_ids
        mu_log = 1
        var_log = np.exp(2)
        score = -self.n_ids - self.n_ids * mu_log**2 / (2 * var_log)

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertAlmostEqual(transformed[1], var_log)

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), score)

        # Test case II.2:
        psis = [1] * self.n_ids
        mu_log = 3
        var_log = np.exp(3)
        score = -1.5 * self.n_ids - self.n_ids * mu_log**2 / (2 * var_log)

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertAlmostEqual(transformed[1], var_log)

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), score)

        # Test case III: psis all the same, sigma_log = 1.
        # Score reduces to
        # -n_ids * log(psi) - n_ids * (log(psi) - mu_log)^2 / 2

        # Test case III.1
        psis = [np.exp(4)] * self.n_ids
        mu_log = 1
        var_log = 1
        score = -self.n_ids * 4 - self.n_ids * (4 - mu_log)**2 / 2  # -85

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertAlmostEqual(transformed[1], var_log)

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), score)

        # Test case III.2
        psis = [np.exp(3)] * self.n_ids
        mu_log = 3
        var_log = 1
        score = -self.n_ids * 3  # -100

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertAlmostEqual(transformed[1], var_log)

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), score)

        # Test case IV: mu_log or sigma_log negative or zero

        # Test case IV.1
        psis = [np.exp(10)] * self.n_ids
        mu = 0
        sigma = 1

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), -np.inf)

        # # Test case IV.2
        psis = [np.exp(10)] * self.n_ids
        mu = 1
        sigma = 0

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), -np.inf)

        # Test case IV.3
        psis = [np.exp(10)] * self.n_ids
        mu = -10
        sigma = 1

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), -np.inf)

        # Test case IV.4
        psis = [np.exp(10)] * self.n_ids
        mu = 1
        sigma = -10

        parameters = psis + [mu] + [sigma]
        self.assertEqual(self.pop_model(parameters), -np.inf)

    def test_get_top_parameter_names(self):
        names = ['Mean', 'Std.']

        self.assertEqual(self.pop_model.get_top_parameter_names(), names)

    def test_n_bottom_parameters(self):
        n_individual_input_params = 10
        self.assertEqual(
            self.pop_model.n_bottom_parameters(),
            n_individual_input_params)

    def test_n_parameters(self):
        n_population_params = 12
        self.assertEqual(self.pop_model.n_parameters(), n_population_params)

    def test_n_top_parameters(self):
        n_population_params = 2
        self.assertEqual(
            self.pop_model.n_top_parameters(), n_population_params)

    # def test_sample(self):
    #     # Test I: sample size 1
    #     seed = 42
    #     parameters = [3, 2]
    #     sample = self.pop_model.sample(parameters, seed=seed)

    #     n_samples = 1
    #     self.assertEqual(sample.shape, (n_samples,))
    #     self.assertEqual(sample[0], 36.94514184203785)

    #     # Test II: sample size > 1
    #     parameters = [3, 2]
    #     n_samples = 4
    #     sample = self.pop_model.sample(parameters, n=n_samples, seed=seed)

    #     self.assertEqual(
    #         sample.shape, (n_samples,))
    #     self.assertAlmostEqual(sample[0], 36.94514184203785)
    #     self.assertAlmostEqual(sample[1], 2.509370155320032)
    #     self.assertAlmostEqual(sample[2], 90.09839866680616)
    #     self.assertAlmostEqual(sample[3], 131.77941585966096)

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.pop_model.sample(parameters)

    def test_set_top_parameter_names(self):
        # Test some name
        names = ['test', 'name']
        self.pop_model.set_top_parameter_names(names)

        self.assertEqual(
            self.pop_model.get_top_parameter_names(), names)

        # Set back to default name
        names = ['Mean', 'Std.']
        self.pop_model.set_top_parameter_names(names)

        self.assertEqual(
            self.pop_model.get_top_parameter_names(), names)

    def test_set_top_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'two', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.pop_model.set_top_parameter_names(names)

    def test_transform_parameters(self):
        # Test case I:
        mu = 1
        sigma = 1
        transformed = self.pop_model.transform_parameters(mu, sigma)

        self.assertEqual(len(transformed), 2)
        mu_log, sigma_log = transformed
        self.assertEqual(mu_log, -np.log(2) / 2)
        self.assertEqual(sigma_log, np.log(2))

        # Test case II:
        mu = 2
        sigma = 2
        transformed = self.pop_model.transform_parameters(mu, sigma)

        self.assertEqual(len(transformed), 2)
        mu_log, var_log = transformed
        self.assertAlmostEqual(mu_log, np.log(2) / 2)
        self.assertAlmostEqual(var_log, np.log(2))


class TestPooledModel(unittest.TestCase):
    """
    Tests the erlotinib.PooledModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.n_ids = 10
        cls.pop_model = erlo.PooledModel(cls.n_ids)

    def test_call(self):
        # For efficiency the input is actually not checked, and 0 is returned
        # regardless
        self.assertEqual(self.pop_model('some values'), 0)

    def test_get_top_parameter_names(self):
        names = ['Pooled']

        self.assertEqual(self.pop_model.get_top_parameter_names(), names)

    def test_n_bottom_parameters(self):
        n_individual_input_params = 0
        self.assertEqual(
            self.pop_model.n_bottom_parameters(),
            n_individual_input_params)

    def test_n_parameters(self):
        n_population_params = 1
        self.assertEqual(self.pop_model.n_parameters(), n_population_params)

    def test_n_top_parameters(self):
        n_population_params = 1
        self.assertEqual(
            self.pop_model.n_top_parameters(), n_population_params)

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
        sample = self.pop_model.sample(parameters, n=n_samples)

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

    def test_set_top_parameter_names(self):
        # Test some name
        names = ['test name']
        self.pop_model.set_top_parameter_names(names)

        self.assertEqual(
            self.pop_model.get_top_parameter_names(), names)

        # Set back to default name
        names = ['Pooled']
        self.pop_model.set_top_parameter_names(names)

        self.assertEqual(
            self.pop_model.get_top_parameter_names(), names)

    def test_set_top_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'one', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.pop_model.set_top_parameter_names(names)


class TestPopulationModel(unittest.TestCase):
    """
    Tests the erlotinib.PopulationModel class.
    """

    @classmethod
    def setUpClass(cls):
        cls.n_ids = 10
        cls.pop_model = erlo.PopulationModel(cls.n_ids)

    def test_call(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model('some values')

    def test_get_bottom_parameter_name(self):
        name = 'Bottom param'
        self.assertEqual(
            self.pop_model.get_bottom_parameter_name(), name)

    def test_get_set_ids(self):
        # Set some ids
        pop_model = erlo.PopulationModel(n_ids=3)
        ids = ['1', '2', '3']
        pop_model.set_ids(ids)

        self.assertEqual(pop_model.get_ids(), ids)

    def test_set_ids_bad_input(self):
        pop_model = erlo.PopulationModel(n_ids=3)
        ids = ['wrong', 'number', 'of', 'IDs']

        with self.assertRaisesRegex(ValueError, 'Length of IDs'):
            pop_model.set_ids(ids)

    def test_get_top_parameter_names(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.get_top_parameter_names()

    def test_n_bottom_parameters(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.n_bottom_parameters()

    def test_n_ids(self):
        self.assertEqual(self.pop_model.n_ids(), self.n_ids)

    def test_n_parameters(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.n_parameters()

    def test_n_parameters_per_id(self):
        self.assertEqual(self.pop_model.n_parameters_per_id(), 1)

    def test_n_top_parameters(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.n_top_parameters()

    def test_sample(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.sample('some values')

    def test_set_bottom_parameter_name(self):
        # Check seeting some name
        name = 'test name'
        self.pop_model.set_bottom_parameter_name(name)

        self.assertEqual(
            self.pop_model.get_bottom_parameter_name(), name)

        # Set back to default
        name = 'Bottom param'
        self.pop_model.set_bottom_parameter_name(name)

        self.assertEqual(
            self.pop_model.get_bottom_parameter_name(), name)

    def test_set_top_parameter_names(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.pop_model.set_top_parameter_names('some name')


if __name__ == '__main__':
    unittest.main()
