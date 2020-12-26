#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import erlotinib as erlo


class TestPredictiveModel(unittest.TestCase):
    """
    Tests the erlo.PredictiveModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Get mechanistic model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)

        # Define error models
        error_models = [erlo.ConstantAndMultiplicativeGaussianErrorModel()]

        # Create predictive model
        cls.model = erlo.PredictiveModel(model, error_models)

    def test_get_parameter_names(self):
        names = self.model.get_parameter_names()

        self.assertEqual(len(names), 7)
        self.assertEqual(names[0], 'myokit.tumour_volume')
        self.assertEqual(names[1], 'myokit.drug_concentration')
        self.assertEqual(names[2], 'myokit.kappa')
        self.assertEqual(names[3], 'myokit.lambda_0')
        self.assertEqual(names[4], 'myokit.lambda_1')
        self.assertEqual(names[5], 'Sigma base')
        self.assertEqual(names[6], 'Sigma rel.')

    def test_n_parameters(self):
        self.assertEqual(self.model.n_parameters(), 7)


if __name__ == '__main__':
    unittest.main()