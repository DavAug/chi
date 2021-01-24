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
        # Score reduces to -n_ids * mu_log^2 / 2

        # Test case I.1:
        psis = [1] * n_ids
        mu_log = 1
        var_log = 1
        ref_score = -n_ids * mu_log**2 / 2  # mu_log = -5

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertEqual(transformed[1], var_log)

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, ref_score)

        # Test case I.2:
        psis = [1] * n_ids
        mu_log = 5
        var_log = 1
        ref_score = -n_ids * mu_log**2 / 2  # mu_log = -125

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertEqual(transformed[1], var_log)

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, ref_score)

        # Test case II: psis = 1.
        # Score reduces to
        # -n_ids * log(sigma_log) -n_ids * mu_log^2 / (2 * sigma_log^2)

        # Test case II.1:
        psis = [1] * n_ids
        mu_log = 1
        var_log = np.exp(2)
        ref_score = -n_ids - n_ids * mu_log**2 / (2 * var_log)

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertAlmostEqual(transformed[1], var_log)

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, ref_score)

        # Test case II.2:
        psis = [1] * n_ids
        mu_log = 3
        var_log = np.exp(3)
        ref_score = -1.5 * n_ids - n_ids * mu_log**2 / (2 * var_log)

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertAlmostEqual(transformed[1], var_log)

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, ref_score)

        # Test case III: psis all the same, sigma_log = 1.
        # Score reduces to
        # -n_ids * log(psi) - n_ids * (log(psi) - mu_log)^2 / 2

        # Test case III.1
        psis = [np.exp(4)] * n_ids
        mu_log = 1
        var_log = 1
        ref_score = -n_ids * 4 - n_ids * (4 - mu_log)**2 / 2  # -85

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertAlmostEqual(transformed[1], var_log)

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, ref_score)

        # Test case III.2
        psis = [np.exp(3)] * n_ids
        mu_log = 3
        var_log = 1
        ref_score = -n_ids * 3  # -100

        # Transform parameters
        mu = np.exp(mu_log + var_log / 2)
        var = mu**2 * (np.exp(var_log) - 1)
        sigma = np.sqrt(var)

        # Make sure that the transform works
        transformed = self.pop_model.transform_parameters(mu, sigma)
        self.assertEqual(transformed[0], mu_log)
        self.assertAlmostEqual(transformed[1], var_log)

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, ref_score)

        # Test case IV: mu_log or sigma_log negative or zero

        # Test case IV.1
        psis = [np.exp(10)] * n_ids
        mu = 0
        sigma = 1

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # # Test case IV.2
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = 0

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # Test case IV.3
        psis = [np.exp(10)] * n_ids
        mu = -10
        sigma = 1

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

        # Test case IV.4
        psis = [np.exp(10)] * n_ids
        mu = 1
        sigma = -10

        parameters = [mu] + [sigma]
        score = self.pop_model.compute_log_likelihood(parameters, psis)
        self.assertEqual(score, -np.inf)

    def test_get_parameter_names(self):
        names = ['Mean', 'Std.']

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
        self.assertEqual(sample[0], 3.0027582879721875)

        # Test II: sample size > 1
        parameters = [3, 2]
        n_samples = 4
        sample = self.pop_model.sample(
            parameters, n_samples=n_samples, seed=seed)

        self.assertEqual(
            sample.shape, (n_samples,))
        self.assertAlmostEqual(sample[0], 3.0027582879721875)
        self.assertAlmostEqual(sample[1], 1.3285661271871976)
        self.assertAlmostEqual(sample[2], 3.9346654828223047)
        self.assertAlmostEqual(sample[3], 4.415456848935877)

    def test_sample_bad_input(self):
        # Too many paramaters
        parameters = [1, 1, 1, 1, 1]

        with self.assertRaisesRegex(ValueError, 'The number of provided'):
            self.pop_model.sample(parameters)

        # Negative mean
        parameters = [-1, 1]

        with self.assertRaisesRegex(ValueError, 'A log-normal distribution'):
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
        self.assertEqual(names[0], 'Mean')
        self.assertEqual(names[1], 'Std.')

    def test_set_parameter_names_bad_input(self):
        # Wrong number of names
        names = ['only', 'two', 'is', 'allowed']
        with self.assertRaisesRegex(ValueError, 'Length of names'):
            self.pop_model.set_parameter_names(names)

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
        cls.pop_model = erlo.PooledModel()

    def test_compute_log_likelihood(self):
        # Test case I: observations empty
        # (exception where 0 is returned for convenience)
        parameters = [1]
        observations = []
        score = self.pop_model.compute_log_likelihood(parameters, observations)
        self.assertEqual(score, 0)

        # Test case II: observation differ from parameter
        # Test case II.1
        parameters = [1]
        observations = [0, 1, 1, 1]
        score = self.pop_model.compute_log_likelihood(parameters, observations)
        self.assertEqual(score, -np.inf)

        # Test case II.1
        parameters = [1]
        observations = [1, 1, 1, 10]
        score = self.pop_model.compute_log_likelihood(parameters, observations)
        self.assertEqual(score, -np.inf)

        # Test case III: all values agree with parameter
        parameters = [1]
        observations = [1, 1, 1, 1]
        score = self.pop_model.compute_log_likelihood(parameters, observations)
        self.assertEqual(score, 0)

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
            'Mean': 1})

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
            'Sigma base': None})

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean': 1})

        n_parameters = self.pop_model.n_parameters()
        self.assertEqual(n_parameters, 1)

        parameter_names = self.pop_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 1)
        self.assertEqual(parameter_names[0], 'Std.')

        # Test case II: fix overlapping set of parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean': 0.2,
            'Std.': 0.1})

        n_parameters = self.pop_model.n_parameters()
        self.assertEqual(n_parameters, 0)

        parameter_names = self.pop_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 0)

        # Test case III: unfix all parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean': None,
            'Std.': None})

        n_parameters = self.pop_model.n_parameters()
        self.assertEqual(n_parameters, 2)

        parameter_names = self.pop_model.get_parameter_names()
        self.assertEqual(len(parameter_names), 2)
        self.assertEqual(parameter_names[0], 'Mean')
        self.assertEqual(parameter_names[1], 'Std.')

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
            'Std.': 0.1})

        n_ids = 10
        n_indiv, n_pop = self.pop_model.n_hierarchical_parameters(n_ids)
        self.assertEqual(n_indiv, 10)
        self.assertEqual(n_pop, 1)

        # Unfix all parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Std.': None})

        n_ids = 10
        n_indiv, n_pop = self.pop_model.n_hierarchical_parameters(n_ids)
        self.assertEqual(n_indiv, 10)
        self.assertEqual(n_pop, 2)

    def test_n_fixed_parameters(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Std.': 0.1})

        self.assertEqual(self.pop_model.n_fixed_parameters(), 1)

        # Unfix all parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Std.': None})

        self.assertEqual(self.pop_model.n_fixed_parameters(), 0)

    def test_n_parameters(self):
        n_parameters = self.pop_model.n_parameters()
        self.assertEqual(n_parameters, 2)

    def test_sample(self):
        # Test case I: fix some parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean': 0.1})

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
            'Mean': None})

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
        self.assertEqual(names[0], 'Mean')
        self.assertEqual(names[1], 'Std.')

        # Fix parameter and set parameter name
        self.pop_model.fix_parameters(name_value_dict={
            'Mean': 1})
        self.pop_model.set_parameter_names(
            ['Std. myokit.tumour_volume'])

        names = self.pop_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Std. myokit.tumour_volume')

        # Reset to defaults
        self.pop_model.set_parameter_names(None)

        names = self.pop_model.get_parameter_names()
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0], 'Std.')

        # Unfix model parameters
        self.pop_model.fix_parameters(name_value_dict={
            'Mean': None})

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
