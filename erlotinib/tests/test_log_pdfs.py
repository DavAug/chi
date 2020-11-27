#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pints

import erlotinib as erlo


class TestReducedLogPDF(unittest.TestCase):
    """
    Tests the erlotinib.ReducedLogPDF class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        times = [1, 2, 3, 4, 5]
        values = [1, 2, 3, 4, 5]

        # Set up inverse problem
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem = erlo.InverseProblem(model, times, values)
        cls.log_likelihood = pints.GaussianLogLikelihood(problem)
        cls.mask = [True, False, False, True, False, True]
        cls.values = [11, 12, 13]
        cls.reduced_log_pdf = erlo.ReducedLogPDF(
            cls.log_likelihood, cls.mask, cls.values)

    def test_bad_input(self):
        # Wrong log-pdf
        log_pdf = 'Bad type'

        with self.assertRaisesRegex(ValueError, 'The log-pdf has to'):
            erlo.ReducedLogPDF(log_pdf, self.mask, self.values)

        # Mask is not as long as the number of parameyers
        mask = [True, True]

        with self.assertRaisesRegex(ValueError, 'Length of mask has to'):
            erlo.ReducedLogPDF(self.log_likelihood, mask, self.values)

        # Mask is not boolean
        mask = ['yes', 'no', 'yes', 'yes', 'yes', 'yes']

        with self.assertRaisesRegex(ValueError, 'Mask has to be a'):
            erlo.ReducedLogPDF(self.log_likelihood, mask, self.values)

        # There are not as many input values as fixed parameters
        values = [1]

        with self.assertRaisesRegex(ValueError, 'There have to be'):
            erlo.ReducedLogPDF(self.log_likelihood, self.mask, values)

    def test_call(self):
        parameters = np.array([11, 1, 1, 12, 1, 13])
        reduced_params = parameters[~np.array(self.mask)]

        self.assertEqual(
            self.reduced_log_pdf(reduced_params),
            self.log_likelihood(parameters))

    def test_n_parameters(self):
        before = self.log_likelihood.n_parameters()
        n_fixed = np.sum(self.mask)

        self.assertEqual(
            self.reduced_log_pdf.n_parameters(), before - n_fixed)


if __name__ == '__main__':
    unittest.main()
