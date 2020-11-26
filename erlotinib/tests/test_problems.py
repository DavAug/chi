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


class TestProblemModellingController(unittest.TestCase):
    """
    Tests the erlotinib.ProblemModellingController class.
    """
    @classmethod
    def setUpClass(cls):
        # Create test dataset
        ids = [0, 0, 0, 1, 1, 1, 2, 2]
        times = [0, 1, 2, 2, np.nan, 4, 1, 3]
        volumes = [np.nan, 0.3, 0.2, 0.5, 0.1, 0.2, 0.234, np.nan]
        cytokines = [3.4, 0.3, 0.2, 0.5, np.nan, 234, 0.234, 0]
        cls.data = pd.DataFrame({
            'ID': ids,
            'Time': times,
            'Biomarker': volumes,
            'Biomarker 1': volumes,
            'Biomarker 2': cytokines})

        # Create test problem
        cls.problem = erlo.ProblemModellingController(
            cls.data, biom_keys=['Biomarker'])

        # Create test model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.model = erlo.PharmacodynamicModel(path)

    def test_bad_input(self):
        # Create data of wrong type
        data = np.ones(shape=(10, 4))

        with self.assertRaisesRegex(ValueError, 'Data has to be'):
            erlo.ProblemModellingController(data=data)

        # Wrong ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        with self.assertRaisesRegex(ValueError, 'Data does not have'):
            erlo.ProblemModellingController(data=data)

        # Wrong time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        with self.assertRaisesRegex(ValueError, 'Data does not have'):
            erlo.ProblemModellingController(data=data)

        # Wrong Biomarker key
        data = self.data.rename(columns={'Biomarker': 'SOME NON-STANDARD KEY'})

        with self.assertRaisesRegex(ValueError, 'Data does not have'):
            erlo.ProblemModellingController(data=data)

    def test_data_keys(self):
        # Rename ID key
        data = self.data.rename(columns={'ID': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        erlo.ProblemModellingController(
            data=data, id_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            erlo.ProblemModellingController(
                data=data, id_key='SOME WRONG KEY')

        # Rename time key
        data = self.data.rename(columns={'Time': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        erlo.ProblemModellingController(
            data=data, time_key='SOME NON-STANDARD KEY')

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            erlo.ProblemModellingController(
                data=data, time_key='SOME WRONG KEY')

        # Rename biomarker key
        data = self.data.rename(columns={'Biomarker': 'SOME NON-STANDARD KEY'})

        # Test that it works with correct mapping
        erlo.ProblemModellingController(
            data=data, biom_keys=['SOME NON-STANDARD KEY'])

        # Test that it fails with wrong mapping
        with self.assertRaisesRegex(
                ValueError, 'Data does not have the key <SOME WRONG KEY>.'):
            erlo.ProblemModellingController(
                data=data, biom_keys=['SOME WRONG KEY'])

    def test_set_mechanistic_model(self):
        # Set output biomarker mapping automatically
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker 1'])

        problem.set_mechanistic_model(model=self.model)

        self.assertEqual(problem._mechanistic_model, self.model)
        outputs = problem._mechanistic_model.outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'myokit.tumour_volume')
        self.assertEqual(len(problem._biom_keys), 1)
        self.assertEqual(problem._biom_keys[0], 'Biomarker 1')

        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker 1', 'Biomarker 2'])

        # Set output biomarker mapping explicitly
        output_biomarker_map = dict({
            'myokit.tumour_volume': 'Biomarker 2',
            'myokit.drug_concentration': 'Biomarker 1'})

        problem.set_mechanistic_model(self.model, output_biomarker_map)

        self.assertEqual(problem._mechanistic_model, self.model)
        outputs = problem._mechanistic_model.outputs()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], 'myokit.tumour_volume')
        self.assertEqual(outputs[1], 'myokit.drug_concentration')
        self.assertEqual(len(problem._biom_keys), 2)
        self.assertEqual(problem._biom_keys[0], 'Biomarker 2')
        self.assertEqual(problem._biom_keys[1], 'Biomarker 1')

    def test_set_mechanistic_model_bad_input(self):
        # Wrong model type
        model = 'some model'

        with self.assertRaisesRegex(ValueError, 'The model has to be'):
            self.problem.set_mechanistic_model(model=model)

        # Wrong number of model outputs
        problem = erlo.ProblemModellingController(
            self.data, biom_keys=['Biomarker', 'Biomarker 1', 'Biomarker 2'])

        with self.assertRaisesRegex(ValueError, 'The model does not have'):
            problem.set_mechanistic_model(model=self.model)

        # Wrong map type
        output_biomarker_map = 'bad map'

        with self.assertRaisesRegex(ValueError, 'The output-biomarker'):
            self.problem.set_mechanistic_model(
                self.model, output_biomarker_map)

        # Map does not contain biomarkers specfied by the dataset
        output_biomarker_map = dict({'Some variable': 'Some biomarker'})

        with self.assertRaisesRegex(ValueError, 'The provided output'):
            self.problem.set_mechanistic_model(
                self.model, output_biomarker_map)



class TestInverseProblem(unittest.TestCase):
    """
    Tests the erlotinib.InverseProblem class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        cls.times = [1, 2, 3, 4, 5]
        cls.values = [1, 2, 3, 4, 5]

        # Set up inverse problem
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.model = erlo.PharmacodynamicModel(path)
        cls.problem = erlo.InverseProblem(cls.model, cls.times, cls.values)

    def test_bad_model_input(self):
        model = 'bad model'

        with self.assertRaisesRegex(ValueError, 'Model has to be an instance'):
            erlo.InverseProblem(model, self.times, self.values)

    def test_bad_times_input(self):
        times = [-1, 2, 3, 4, 5]
        with self.assertRaisesRegex(ValueError, 'Times cannot be negative.'):
            erlo.InverseProblem(self.model, times, self.values)

        times = [5, 4, 3, 2, 1]
        with self.assertRaisesRegex(ValueError, 'Times must be increasing.'):
            erlo.InverseProblem(self.model, times, self.values)

    def test_bad_values_input(self):
        values = [1, 2, 3, 4, 5, 6, 7]
        with self.assertRaisesRegex(ValueError, 'Values array must have'):
            erlo.InverseProblem(self.model, self.times, values)

        values = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        with self.assertRaisesRegex(ValueError, 'Values array must have'):
            erlo.InverseProblem(self.model, self.times, values)

    def test_evaluate(self):
        parameters = [0.1, 1, 1, 1, 1]
        output = self.problem.evaluate(parameters)

        n_times = 5
        n_outputs = 1
        self.assertEqual(output.shape, (n_times, n_outputs))

    def test_evaluateS1(self):
        parameters = [0.1, 1, 1, 1, 1]
        with self.assertRaises(NotImplementedError):
            self.problem.evaluateS1(parameters)

    def test_n_ouputs(self):
        self.assertEqual(self.problem.n_outputs(), 1)

    def test_n_parameters(self):
        self.assertEqual(self.problem.n_parameters(), 5)

    def test_n_times(self):
        n_times = len(self.times)
        self.assertEqual(self.problem.n_times(), n_times)

    def test_times(self):
        times = self.problem.times()
        n_times = len(times)

        self.assertEqual(n_times, 5)

        self.assertEqual(times[0], self.times[0])
        self.assertEqual(times[1], self.times[1])
        self.assertEqual(times[2], self.times[2])
        self.assertEqual(times[3], self.times[3])
        self.assertEqual(times[4], self.times[4])

    def test_values(self):
        values = self.problem.values()

        n_times = 5
        n_outputs = 1
        self.assertEqual(values.shape, (n_times, n_outputs))

        self.assertEqual(values[0], self.values[0])
        self.assertEqual(values[1], self.values[1])
        self.assertEqual(values[2], self.values[2])
        self.assertEqual(values[3], self.values[3])
        self.assertEqual(values[4], self.values[4])


if __name__ == '__main__':
    unittest.main()
