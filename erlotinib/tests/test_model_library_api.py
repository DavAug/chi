#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os
import unittest

import myokit.formats.sbml as sbml

import erlotinib as erlo


class TestModelLibrary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_library = erlo.ModelLibrary()

    def test_existence_tumour_growth_inhibition_model_koch(self):
        path = self.model_library.tumour_growth_inhibition_model_koch()

        self.assertTrue(os.path.exists(path))


class TestTumourGrowthInhibitionPDModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        lib = erlo.ModelLibrary()
        path = lib.tumour_growth_inhibition_model_koch()
        importer = sbml.SBMLImporter()
        cls.model = importer.model(path)

    def test_states(self):
        state_names = sorted(
            [var.qname() for var in self.model.states()])

        n_states = len(state_names)
        self.assertTrue(n_states, 1)

        tumour_volume = 'myokit.tumour_volume'
        self.assertEqual(state_names[0], tumour_volume)

    def test_constant_variables(self):
        const_names = sorted(
            [var.qname() for var in self.model.variables(const=True)])

        n_const = len(const_names)
        self.assertEqual(n_const, 4)

        drug_conc = 'myokit.drug_concentration'
        self.assertEqual(const_names[0], drug_conc)

        kappa = 'myokit.kappa'
        self.assertEqual(const_names[1], kappa)

        lambda_0 = 'myokit.lambda_0'
        self.assertEqual(const_names[2], lambda_0)

        lambda_1 = 'myokit.lambda_1'
        self.assertEqual(const_names[3], lambda_1)


if __name__ == '__main__':
    unittest.main()
