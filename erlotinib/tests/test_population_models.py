#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

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

        n_params_per_individual = len(parameters)
        n_samples = 1
        self.assertEqual(
            sample.shape, (n_samples, n_params_per_individual))
        self.assertEqual(sample[0, 0], parameters[0])

        # Test one sample size > 1
        parameters = [3]
        n_samples = 4
        sample = self.pop_model.sample(parameters, n=n_samples)

        n_params_per_individual = len(parameters)
        self.assertEqual(
            sample.shape, (n_samples, n_params_per_individual))
        self.assertEqual(sample[0, 0], parameters[0])
        self.assertEqual(sample[1, 0], parameters[0])
        self.assertEqual(sample[2, 0], parameters[0])
        self.assertEqual(sample[3, 0], parameters[0])

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

        with self.assertRaisesRegex(ValueError, 'Length of IDs')
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
