#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os
import unittest

import myokit.formats.sbml as sbml

from chi.library import ModelLibrary


class TestModelLibrary(unittest.TestCase):
    """
    Tests the chi.ModelLibrary class.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_library = ModelLibrary()

    def test_existence_erlotinib_tumour_growth_inhibition_model(self):
        path = self.model_library.erlotinib_tumour_growth_inhibition_model()

        self.assertTrue(os.path.exists(path))

    def test_existence_tumour_growth_inhibition_model_koch(self):
        path = self.model_library.tumour_growth_inhibition_model_koch()

        self.assertTrue(os.path.exists(path))

    def test_existence_tumour_growth_inhibition_model_koch_reparametrised(
            self):
        lib = self.model_library
        path = lib.tumour_growth_inhibition_model_koch_reparametrised()

        self.assertTrue(os.path.exists(path))

    def test_existence_one_compartment_pk_model(self):
        path = self.model_library.one_compartment_pk_model()

        self.assertTrue(os.path.exists(path))


class TestErlotinibTumourGrowthInhibitionModel(unittest.TestCase):
    """
    Tests the chi.modelLibrary.erlotinib_tumour_growth_inhibition_model
    method.
    """

    @classmethod
    def setUpClass(cls):
        lib = ModelLibrary()
        path = lib.erlotinib_tumour_growth_inhibition_model()
        importer = sbml.SBMLImporter()
        cls.model = importer.model(path)

    def test_constant_variables(self):
        const_names = sorted(
            [var.qname() for var in self.model.variables(const=True)])

        n_const = len(const_names)
        self.assertEqual(n_const, 5)
        self.assertEqual(const_names[0], 'central.size')
        self.assertEqual(const_names[1], 'myokit.critical_volume')
        self.assertEqual(const_names[2], 'myokit.elimination_rate')
        self.assertEqual(const_names[3], 'myokit.kappa')
        self.assertEqual(const_names[4], 'myokit.lambda')

    def test_intermediate_variables(self):
        inter_names = sorted(
            [var.qname() for var in self.model.variables(inter=True)])

        n_inter = len(inter_names)
        self.assertEqual(n_inter, 1)

        drug_conc = 'central.drug_concentration'
        self.assertEqual(inter_names[0], drug_conc)

    def test_states(self):
        state_names = sorted(
            [var.qname() for var in self.model.states()])

        n_states = len(state_names)
        self.assertTrue(n_states, 2)
        self.assertEqual(state_names[0], 'central.drug_amount')
        self.assertEqual(state_names[1], 'myokit.tumour_volume')


class TestTumourGrowthInhibitionModelKoch(unittest.TestCase):
    """
    Tests the chi.modelLibrary.tumour_growth_inhibition_model_koch
    method.
    """

    @classmethod
    def setUpClass(cls):
        lib = ModelLibrary()
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


class TestTumourGrowthInhibitionModelKochReparametrised(unittest.TestCase):
    """
    Tests the
    chi.modelLibrary.tumour_growth_inhibition_model_koch_reparametrised
    method.
    """

    @classmethod
    def setUpClass(cls):
        lib = ModelLibrary()
        path = lib.tumour_growth_inhibition_model_koch_reparametrised()
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

        critical_volume = 'myokit.critical_volume'
        self.assertEqual(const_names[0], critical_volume)

        drug_conc = 'myokit.drug_concentration'
        self.assertEqual(const_names[1], drug_conc)

        kappa = 'myokit.kappa'
        self.assertEqual(const_names[2], kappa)

        growth_rate = 'myokit.lambda'
        self.assertEqual(const_names[3], growth_rate)


class TestOneCompartmentPKModel(unittest.TestCase):
    """
    Tests the chi.modelLibrary.one_compartment_pk_model method.
    """

    @classmethod
    def setUpClass(cls):
        lib = ModelLibrary()
        path = lib.one_compartment_pk_model()
        importer = sbml.SBMLImporter()
        cls.model = importer.model(path)

    def test_states(self):
        state_names = sorted(
            [var.qname() for var in self.model.states()])

        n_states = len(state_names)
        self.assertTrue(n_states, 1)

        drug_amount = 'central.drug_amount'
        self.assertEqual(state_names[0], drug_amount)

    def test_constant_variables(self):
        const_names = sorted(
            [var.qname() for var in self.model.variables(const=True)])

        n_const = len(const_names)
        self.assertEqual(n_const, 2)

        volume = 'central.size'
        self.assertEqual(const_names[0], volume)

        elimination_rate = 'myokit.elimination_rate'
        self.assertEqual(const_names[1], elimination_rate)

    def test_intermediate_variables(self):
        inter_names = sorted(
            [var.qname() for var in self.model.variables(inter=True)])

        n_const = len(inter_names)
        self.assertEqual(n_const, 1)

        drug_conc = 'central.drug_concentration'
        self.assertEqual(inter_names[0], drug_conc)


if __name__ == '__main__':
    unittest.main()
