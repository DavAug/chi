#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pandas as pd

from chi import plots


class TestParameterEstimatePlot(unittest.TestCase):
    """
    Tests the chi.plots.ParameterEstimatePlot class.
    """

    @classmethod
    def setUpClass(cls):
        # Summary of dataset
        ids = [0, 1, 2]
        n_ids = len(ids)
        params = ['Param 1', 'Param 2']
        n_params = len(params)
        runs = [1, 2]
        n_runs = len(runs)

        # Create dataset
        ids = [0] * (n_params * n_runs) + [1] * (n_params * n_runs) + [
            2] * (n_params * n_runs)
        params = params * (n_ids * n_runs)
        estimates = [1] * (n_ids * n_params * n_runs)
        scores = [10] * (n_ids * n_params * n_runs)
        runs = [1] * (n_ids * n_params) + [2] * (n_ids * n_params)

        data = pd.DataFrame({
            'ID': ids,
            'Parameter': params,
            'Estimate': estimates,
            'Score': scores,
            'Run': runs})

        # Add a 'population' parameter to the dataset
        cls.data = data.append(pd.DataFrame({
            'ID': [None] * 8,
            'Parameter': 'Param 3',
            'Sample': np.arange(8),
            'Iteration': [1, 2, 3, 4] * 2,
            'Run': [1] * 4 + [2] * 4
        }))

        # Create test figure
        cls.fig = plots.ParameterEstimatePlot()

    def test_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            self.fig.add_data, data)

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

    def test_wrong_est_key(self):
        # Rename estimate key
        data = self.data.rename(columns={'Estimate': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Estimate>.',
            self.fig.add_data, data)

    def test_wrong_score_key(self):
        # Rename score key
        data = self.data.rename(columns={'Score': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Score>.',
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

    def test_est_key_mapping(self):
        # Rename estimate key
        data = self.data.rename(columns={'Estimate': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, est_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, est_key='SOME WRONG KEY')

    def test_score_key_mapping(self):
        # Rename score key
        data = self.data.rename(columns={'Score': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, score_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, score_key='SOME WRONG KEY')

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
