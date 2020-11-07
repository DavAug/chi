#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pints
import pints.toy

import erlotinib as erlo


class TestOptimise(unittest.TestCase):
    """
    Tests the pkpd.optimise function.
    """

    @classmethod
    def setUpClass(cls):
        # Define objective function
        model = pints.toy.ConstantModel(1)
        times = np.linspace(1, 10)
        cls.true_params = [2.5]
        values = model.simulate(cls.true_params, times)
        problem = pints.SingleOutputProblem(model, times, values)
        cls.error = pints.SumOfSquaresError(problem)

        # Define initial parameters
        cls.params = [[3]]

        # Define boundaries
        cls.boundaries = pints.RectangularBoundaries(1, 5)

    def test_bad_objective_function(self):
        objective_function = 'bad obj func'

        self.assertRaisesRegex(
            ValueError, 'Objective function has to be',
            erlo.optimise, objective_function, pints.CMAES, self.params)

    def test_bad_optimiser(self):
        optimiser = pints.LogPDF

        self.assertRaisesRegex(
            ValueError, 'Optimiser has to be',
            erlo.optimise, self.error, optimiser, self.params)

    def test_bad_parameters(self):
        params = [[1, 2, 3]]

        self.assertRaisesRegex(
            ValueError, 'Initial parameters has the wrong shape!',
            erlo.optimise, self.error, pints.CMAES, params)

    def test_bad_boundaries(self):
        boundaries = 'boundaries'

        self.assertRaisesRegex(
            ValueError, 'Boundaries have to be',
            erlo.optimise, self.error, pints.CMAES, self.params, 1, boundaries)

    def test_nan_when_fails(self):
        # CMAES return NAN for 1-dim problems
        params, _ = erlo.optimise(
            self.error, pints.CMAES, self.params, 1, self.boundaries)

        self.assertEqual(params.shape, (1, 1))
        np.testing.assert_equal(params[0, 0], np.nan)

    def test_call(self):
        # Test single run
        params, _ = erlo.optimise(
            self.error, pints.XNES, self.params, 1, self.boundaries, 1000)

        self.assertEqual(params.shape, (1, 1))
        self.assertAlmostEqual(params[0, 0], self.true_params[0], 1)

        # Test multiple runs
        initial_params = [[2], [3]]
        params, _ = erlo.optimise(
            self.error, pints.XNES, initial_params, 2, self.boundaries)

        self.assertEqual(params.shape, (2, 1))
        self.assertAlmostEqual(params[0, 0], self.true_params[0], 1)
        self.assertAlmostEqual(params[1, 0], self.true_params[0], 1)


if __name__ == '__main__':
    unittest.main()
