#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

import erlotinib as erlo


class TestPharmacodynamicModel(unittest.TestCase):
    """
    Tests `pkpd.PharmacodynamicModel`.
    """
    @classmethod
    def setUpClass(cls):
        path = pkpd.ModelLibrary().get_path('Tumour growth without treatment')
        cls.model = pkpd.PharmacodynamicModel(path, is_log_transformed=False)
        cls.model_log_transformed = pkpd.PharmacodynamicModel(path)

    def test_n_outputs(self):
        self.assertEqual(self.model.n_outputs(), 1)
        self.assertEqual(self.model_log_transformed.n_outputs(), 1)

    def test_n_parameters(self):
        self.assertEqual(self.model.n_parameters(), 3)
        self.assertEqual(self.model_log_transformed.n_parameters(), 3)

    def test_outputs(self):
        outputs = self.model.outputs()

        self.assertEqual(outputs, ['myokit.tumour_volume'])

    def test_parameters(self):
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.critical_volume')
        self.assertEqual(parameters[2], 'myokit.lambda')

    def test_set_outputs(self):

        # Set bad output
        self.assertRaisesRegex(
            KeyError, 'The variable <', self.model.set_outputs, ['some.thing'])

        # Set twice the same output
        outputs = ['myokit.tumour_volume', 'myokit.tumour_volume']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 2)
        output = self.model.simulate([0.1, 2, 1], [0, 1])
        self.assertEqual(output.shape, (2, 2))

        # Set to default again
        outputs = ['myokit.tumour_volume']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 1)
        output = self.model.simulate([0.1, 2, 1], [0, 1])
        self.assertEqual(output.shape, (2,))

    def test_simulate(self):

        times = [0, 1, 2, 3]

        # Test model with bare parameters
        parameters = [0.1, 1, 1]
        output = self.model.simulate(parameters, times)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (4,))

        # Test model with log-parameters
        parameters = np.log(parameters)
        log_output = self.model_log_transformed.simulate(parameters, times)
        self.assertIsInstance(log_output, np.ndarray)
        self.assertEqual(log_output.shape, (4,))

        # Compare results to each other
        np.testing.assert_almost_equal(output, log_output)


if __name__ == '__main__':
    unittest.main()
