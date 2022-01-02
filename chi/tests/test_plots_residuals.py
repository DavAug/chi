#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

from chi import plots
from chi.library import DataLibrary


class TestResidualPlot(unittest.TestCase):
    """
    Tests the chi.plots.ResidualPlot class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test datasets
        cls.measurements = DataLibrary().lung_cancer_control_group()
        cls.data = cls.measurements.rename({'Value': 'Value'})

        # Create test figure
        cls.fig = plots.ResidualPlot(cls.measurements)

    def test_bad_instantiation(self):
        # Create data of wrong type
        measurements = np.ones(shape=(10, 4))
        with self.assertRaisesRegex(TypeError, 'Measurements has to be'):
            plots.ResidualPlot(measurements)

        # Wrong ID key
        with self.assertRaisesRegex(ValueError, 'Measurements does not have'):
            plots.ResidualPlot(self.measurements, id_key='Wrong')

        # Wrong time key
        with self.assertRaisesRegex(ValueError, 'Measurements does not have'):
            plots.ResidualPlot(self.measurements, time_key='Wrong')

        # Wrong observable key
        with self.assertRaisesRegex(ValueError, 'Measurements does not have'):
            plots.ResidualPlot(self.measurements, obs_key='Wrong')

        # Wrong measurement key
        with self.assertRaisesRegex(ValueError, 'Measurements does not have'):
            plots.ResidualPlot(self.measurements, value_key='Wrong')

    def test_add_data_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        with self.assertRaisesRegex(TypeError, 'Data has to be'):
            self.fig.add_data(data)

    def test_add_data_wrong_observable(self):
        # observable does not exist in prediction dataframe
        observable = 'Does not exist'

        with self.assertRaisesRegex(ValueError, 'The observable could not be'):
            self.fig.add_data(self.data, observable)

        # observable does not exist in measurement dataframe
        data = self.data.copy()
        data['Observable'] = 'Does not exist'
        observable = 'Does not exist'

        with self.assertRaisesRegex(ValueError, 'The observable <Does not'):
            self.fig.add_data(data, observable)

    def test_add_data_wrong_individual(self):
        individual = 'does not exist'

        self.assertRaisesRegex(
            ValueError, 'The ID <does not exist> does not exist.',
            self.fig.add_data, self.data, individual=individual)

    def test_add_data_wrong_time_key(self):
        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Time>.',
            self.fig.add_data, data)

    def test_add_data_wrong_obs_key(self):
        # Rename observable key
        data = self.data.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Observable>.',
            self.fig.add_data, data)

    def test_add_data_wrong_value_key(self):
        # Rename measurement key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Value>.',
            self.fig.add_data, data)

    def test_add_data_time_key_mapping(self):
        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(
            data, time_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(
                data, time_key='SOME WRONG KEY')

    def test_add_data_obs_key_mapping(self):
        # Rename observable key
        data = self.data.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(
            data, obs_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(
                data, obs_key='SOME WRONG KEY')

    def test_add_data_value_key_mapping(self):
        # Rename measurement key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(
            data, value_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(
                data, value_key='SOME WRONG KEY')

    def test_add_data_show_relative(self):
        self.fig.add_data(self.data, show_relative=True)

    def test_data_wrong_time_points(self):
        # Not all measured time points can be found in the prediction
        # dataframe
        data = self.data.copy()
        data['Time'] = 1

        with self.assertRaisesRegex(
                ValueError, 'The prediction dataframe is not'):
            self.fig.add_data(data)

    def test_add_data_individual(self):
        # Select an individual
        self.fig.add_data(self.data, individual=40)


if __name__ == '__main__':
    unittest.main()
