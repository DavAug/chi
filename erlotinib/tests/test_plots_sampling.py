#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pandas as pd

import erlotinib as erlo


class TestParameterEstimatePlot(unittest.TestCase):
    """
    Tests the erlotinib.plots.ParameterEstimatePlot class.
    """

    @classmethod
    def setUpClass(cls):
        # Summary of dataset
        ids = [0, 1, 2]
        n_ids = len(ids)
        params = ['Param 1', 'Param 2']
        n_params = len(params)
        iterations = [1, 2, 3, 4]
        n_iter = len(iterations)
        runs = [1, 2]
        n_runs = len(runs)

        # Create dataset
        ids = [0] * (n_params * n_runs * n_iter) \
            + [1] * (n_params * n_runs * n_iter) \
            + [2] * (n_params * n_runs * n_iter)
        params = params * (n_ids * n_runs * n_iter)
        samples = [1, 2, 1.5, 2.1, 0.9, 1.9, 6, 8] * (n_ids * n_runs)
        iterations = (
            [1] * n_params + [2] * n_params + [3] * n_params + [4] * n_params
            ) * n_runs * n_ids
        runs = (
            [1] * (n_params * n_iter) + [2] * (n_params * n_iter)
            ) * n_ids

        data = pd.DataFrame({
            'ID': ids,
            'Parameter': params,
            'Sample': samples,
            'Iteration': iterations,
            'Run': runs})

        # Add a 'population' parameter to the dataset
        cls.data = data.append(pd.DataFrame({
            'ID': ['Pooled'] * 8,
            'Parameter': 'Param 3',
            'Sample': np.arange(8),
            'Iteration': [1, 2, 3, 4] * 2,
            'Run': [1] * 4 + [2] * 4
        }))

        # Create test figure
        cls.fig = erlo.plots.MarginalPosteriorPlot()

    def test_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            self.fig.add_data, data)

    def test_bad_warm_up_iter_input(self):
        # Check that error is raised when warm up exceeds number of iterations
        with self.assertRaisesRegex(ValueError, 'The number of warm up'):
            self.fig.add_data(data=self.data, warm_up_iter=10000)

    def test_wrong_id_key(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <ID>.',
            self.fig.add_data, data)

    def test_wrong_param_key(self):
        # Rename parameter key
        data = self.data.rename(columns={'Parameter': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Parameter>.',
            self.fig.add_data, data)

    def test_wrong_sample_key(self):
        # Rename sample key
        data = self.data.rename(columns={'Sample': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Sample>.',
            self.fig.add_data, data)

    def test_wrong_iter_key(self):
        # Rename iteration key
        data = self.data.rename(columns={'Iteration': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Iteration>.',
            self.fig.add_data, data)

    def test_wrong_run_key(self):
        # Rename estimate key
        data = self.data.rename(columns={'Run': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Run>.',
            self.fig.add_data, data)

    def test_id_key_mapping(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, id_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, id_key='SOME WRONG KEY')

    def test_param_key_mapping(self):
        # Rename parameter key
        data = self.data.rename(columns={'Parameter': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, param_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, param_key='SOME WRONG KEY')

    def test_sample_key_mapping(self):
        # Rename sample key
        data = self.data.rename(columns={'Sample': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, sample_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, sample_key='SOME WRONG KEY')

    def test_iter_key_mapping(self):
        # Rename iteration key
        data = self.data.rename(columns={'Iteration': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, iter_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, iter_key='SOME WRONG KEY')

    def test_run_key_mapping(self):
        # Rename run key
        data = self.data.rename(columns={'Run': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, run_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, run_key='SOME WRONG KEY')


if __name__ == '__main__':
    unittest.main()
