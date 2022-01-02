#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import plotly.graph_objects as go

from chi import plots
from chi.library import DataLibrary


class TestPDPredictivePlot(unittest.TestCase):
    """
    Tests the chi.plots.PDPredictivePlot class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test datasets
        cls.data = DataLibrary().lung_cancer_control_group()
        cls.prediction = cls.data.copy()

        # Create test figure
        cls.fig = plots.PDPredictivePlot()

    def test_add_data_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        with self.assertRaisesRegex(TypeError, 'Data has to be'):
            self.fig.add_data(data)

    def test_add_data_wrong_observable(self):
        biomarker = 'Does not exist'

        with self.assertRaisesRegex(ValueError, 'The observable could not be'):
            self.fig.add_data(self.data, biomarker)

    def test_add_data_wrong_id_key(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <ID>.',
            self.fig.add_data, data)

    def test_add_data_wrong_time_key(self):
        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Time>.',
            self.fig.add_data, data)

    def test_add_data_wrong_obs_key(self):
        # Rename biomarker key
        data = self.data.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Observable>.',
            self.fig.add_data, data)

    def test_add_data_wrong_value_key(self):
        # Rename Value key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Value>.',
            self.fig.add_data, data)

    def test_add_data_id_key_mapping(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(
            data, id_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data, id_key='SOME WRONG KEY')

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
        # Rename biomarker key
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
        # Rename Value key
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

    def test_add_prediction_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            self.fig.add_prediction, data)

    def test_add_prediction_wrong_observable(self):
        # Specify biomarker that is not in the dataset
        biomarker = 'Does not exist'

        with self.assertRaisesRegex(ValueError, 'The observable could not be'):
            self.fig.add_prediction(self.prediction, biomarker)

    def test_add_prediction_bad_bulk_probs(self):
        # A maximum of 7 bulk probs are allowed
        bulk_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        with self.assertRaisesRegex(ValueError, 'At most 7 different bulk'):
            self.fig.add_prediction(self.prediction, bulk_probs=bulk_probs)

        # Negative probability
        bulk_probs = [-0.1]

        with self.assertRaisesRegex(ValueError, 'The provided bulk prob'):
            self.fig.add_prediction(self.prediction, bulk_probs=bulk_probs)

        # Probability greater 1
        bulk_probs = [1.1]

        with self.assertRaisesRegex(ValueError, 'The provided bulk prob'):
            self.fig.add_prediction(self.prediction, bulk_probs=bulk_probs)

    def test_add_prediction_wrong_time_key(self):
        # Rename time key
        data = self.prediction.rename(
            columns={'Time': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Time>.',
            self.fig.add_prediction, data)

    def test_add_prediction_wrong_obs_key(self):
        # Rename biomarker key
        data = self.prediction.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Observable>.',
            self.fig.add_prediction, data)

    def test_add_prediction_wrong_value_key(self):
        # Rename value key
        data = self.prediction.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Value>.',
            self.fig.add_prediction, data)

    def test_add_prediction_obs_mapping(self):
        # Rename rename biomarker
        data = self.prediction.copy()
        data['Observable'] = 'SOME NON-STANDARD BIOMARKER'

        # Test that it works with correct mapping
        self.fig.add_prediction(
            data=data, observable='SOME NON-STANDARD BIOMARKER')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'The observable could not be found'):
            self.fig.add_prediction(
                data=data, observable='SOME WRONG BIOMARKER')

    def test_add_prediction_no_provided_bulk_prob(self):
        # Test that it works with correct mapping
        self.fig.add_prediction(data=self.prediction, bulk_probs=None)

        trace = self.fig._fig.data[-1]
        self.assertIsInstance(trace, go.Scatter)

    def test_add_prediction_time_key_mapping(self):
        # Rename time key
        data = self.prediction.rename(
            columns={'Time': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_prediction(data=data, time_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_prediction(data=data, time_key='SOME WRONG KEY')

    def test_add_prediction_obs_key_mapping(self):
        # Rename biomarker key
        data = self.prediction.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_prediction(data=data, obs_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_prediction(data=data, obs_key='SOME WRONG KEY')

    def test_add_prediction_value_key_mapping(self):
        # Rename value key
        data = self.prediction.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_prediction(data=data, value_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_prediction(data=data, value_key='SOME WRONG KEY')


class TestPKPredictivePlot(unittest.TestCase):
    """
    Tests the chi.plots.PKPredictivePlot class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test datasets
        cls.data = DataLibrary().lung_cancer_low_erlotinib_dose_group()
        cls.prediction = cls.data.copy()

        # Create test figure
        cls.fig = plots.PKPredictivePlot()

    def test_add_data_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        with self.assertRaisesRegex(TypeError, 'Data has to be'):
            self.fig.add_data(data)

    def test_add_data_wrong_observable(self):
        observable = 'Does not exist'

        with self.assertRaisesRegex(ValueError, 'The observable could not be'):
            self.fig.add_data(self.data, observable)

    def test_add_data_wrong_id_key(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <ID>.',
            self.fig.add_data, data)

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
        # Rename Value key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Value>.',
            self.fig.add_data, data)

    def test_add_data_id_key_mapping(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(
            data, id_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data, id_key='SOME WRONG KEY')

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
        # Rename Value key
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

    def test_add_prediction_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            self.fig.add_prediction, data)

    def test_add_prediction_wrong_observable(self):
        # Specify observable that is not in the dataset
        observable = 'Does not exist'

        with self.assertRaisesRegex(ValueError, 'The observable could not be'):
            self.fig.add_prediction(self.prediction, observable)

    def test_add_prediction_bad_bulk_probs(self):
        # A maximum of 7 bulk probs are allowed
        bulk_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        with self.assertRaisesRegex(ValueError, 'At most 7 different bulk'):
            self.fig.add_prediction(self.prediction, bulk_probs=bulk_probs)

        # Negative probability
        bulk_probs = [-0.1]

        with self.assertRaisesRegex(ValueError, 'The provided bulk prob'):
            self.fig.add_prediction(self.prediction, bulk_probs=bulk_probs)

        # Probability greater 1
        bulk_probs = [1.1]

        with self.assertRaisesRegex(ValueError, 'The provided bulk prob'):
            self.fig.add_prediction(self.prediction, bulk_probs=bulk_probs)

    def test_add_prediction_wrong_time_key(self):
        # Rename time key
        data = self.prediction.rename(
            columns={'Time': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Time>.',
            self.fig.add_prediction, data)

    def test_add_prediction_wrong_obs_key(self):
        # Rename observable key
        data = self.prediction.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Observable>.',
            self.fig.add_prediction, data)

    def test_add_prediction_wrong_value_key(self):
        # Rename value key
        data = self.prediction.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Value>.',
            self.fig.add_prediction, data)

    def test_add_prediction_biom_mapping(self):
        # Rename rename observable
        data = self.prediction.copy()
        data['Observable'] = 'SOME NON-STANDARD observable'

        # Test that it works with correct mapping
        self.fig.add_prediction(
            data=data, observable='SOME NON-STANDARD observable')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'The observable could not be found'):
            self.fig.add_prediction(
                data=data, observable='SOME WRONG observable')

    def test_add_prediction_no_provided_bulk_prob(self):
        # Test that it works with correct mapping
        self.fig.add_prediction(data=self.prediction, bulk_probs=None)

        trace = self.fig._fig.data[-1]
        self.assertIsInstance(trace, go.Scatter)

    def test_add_prediction_time_key_mapping(self):
        # Rename time key
        data = self.prediction.rename(
            columns={'Time': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_prediction(data=data, time_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_prediction(data=data, time_key='SOME WRONG KEY')

    def test_add_prediction_obs_key_mapping(self):
        # Rename observable key
        data = self.prediction.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_prediction(data=data, obs_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_prediction(data=data, obs_key='SOME WRONG KEY')

    def test_add_prediction_value_key_mapping(self):
        # Rename value key
        data = self.prediction.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_prediction(data=data, value_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_prediction(data=data, value_key='SOME WRONG KEY')


class TestPDTimeSeriesPlot(unittest.TestCase):
    """
    Tests the chi.plots.PDTimeSeriesPlot class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test dataset
        cls.data = DataLibrary().lung_cancer_control_group()

        # Create test figure
        cls.fig = plots.PDTimeSeriesPlot()

    def test_add_data_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            self.fig.add_data, data)

    def test_add_data_wrong_observable(self):
        observable = 'Does not exist'

        with self.assertRaisesRegex(ValueError, 'The observable could not be'):
            self.fig.add_data(self.data, observable)

    def test_add_data_wrong_id_key(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <ID>.',
            self.fig.add_data, data)

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
        # Rename Value key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Value>.',
            self.fig.add_data, data)

    def test_add_data_id_key_mapping(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, id_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, id_key='SOME WRONG KEY')

    def test_add_data_time_key_mapping(self):
        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, time_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, time_key='SOME WRONG KEY')

    def test_add_data_obs_key_mapping(self):
        # Rename observable key
        data = self.data.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, obs_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, obs_key='SOME WRONG KEY')

    def test_add_data_value_key_mapping(self):
        # Rename Value key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, value_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, value_key='SOME WRONG KEY')

    def test_add_simulation_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            self.fig.add_simulation, data)

    def test_add_simulation_wrong_time_key(self):
        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Time>.',
            self.fig.add_simulation, data)

    def test_add_simulation_wrong_value_key(self):
        # Rename value key
        data = self.data.rename(columns={'Value': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Value>.',
            self.fig.add_simulation, data)

    def test_add_simulation_time_key_mapping(self):
        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_simulation(data=data, time_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_simulation(data=data, time_key='SOME WRONG KEY')

    def test_add_simulation_value_key_mapping(self):
        # Rename value key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_simulation(data=data, value_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_simulation(data=data, value_key='SOME WRONG KEY')


class TestPKTimeSeriesPlot(unittest.TestCase):
    """
    Tests the chi.plots.PKTimeSeriesPlot class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test dataset
        cls.data = DataLibrary().lung_cancer_low_erlotinib_dose_group()

        # Create test figure
        cls.fig = plots.PKTimeSeriesPlot()

    def test_add_data_wrong_data_type(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        self.assertRaisesRegex(
            TypeError, 'Data has to be pandas.DataFrame.',
            self.fig.add_data, data)

    def test_add_data_wrong_observable(self):
        observable = 'Does not exist'

        with self.assertRaisesRegex(ValueError, 'The observable could not be'):
            self.fig.add_data(self.data, observable)

    def test_add_data_wrong_id_key(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <ID>.',
            self.fig.add_data, data)

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
        # Rename Value key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Value>.',
            self.fig.add_data, data)

    def test_add_data_wrong_dose_key(self):
        # Rename dose key
        data = self.data.rename(columns={'Dose': 'SOME NON-STANDARD KEY'})

        self.assertRaisesRegex(
            ValueError, 'Data does not have the key <Dose>.',
            self.fig.add_data, data)

    def test_add_data_id_key_mapping(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, id_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, id_key='SOME WRONG KEY')

    def test_add_data_time_key_mapping(self):
        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, time_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, time_key='SOME WRONG KEY')

    def test_add_data_obs_key_mapping(self):
        # Rename observable key
        data = self.data.rename(
            columns={'Observable': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, obs_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, obs_key='SOME WRONG KEY')

    def test_add_data_dose_key_mapping(self):
        # Rename dose key
        data = self.data.rename(columns={'Dose': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, dose_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, dose_key='SOME WRONG KEY')

    def test_add_data_value_key_mapping(self):
        # Rename Value key
        data = self.data.rename(
            columns={'Value': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        self.fig.add_data(data=data, value_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            self.fig.add_data(data=data, value_key='SOME WRONG KEY')

    def test_add_simulation(self):
        with self.assertRaisesRegex(NotImplementedError, ''):
            self.fig.add_simulation(self.data)


if __name__ == '__main__':
    unittest.main()
