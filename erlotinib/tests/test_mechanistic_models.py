#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np

import erlotinib as erlo


class TestModel(unittest.TestCase):
    """
    Tests `erlotinib.MechanisticModel`.
    """
    @classmethod
    def setUpClass(cls):
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.model = erlo.MechanisticModel(path)

    def test_n_outputs(self):
        self.assertEqual(self.model.n_outputs(), 1)

    def test_n_parameters(self):
        self.assertEqual(self.model.n_parameters(), 5)

    def test_outputs(self):
        outputs = self.model.outputs()

        self.assertEqual(outputs, ['myokit.tumour_volume'])

    def test_parameters(self):
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')

    def test_set_outputs(self):

        # Set bad output
        self.assertRaisesRegex(
            KeyError, 'The variable <', self.model.set_outputs, ['some.thing'])

        # Set twice the same output
        outputs = ['myokit.tumour_volume', 'myokit.tumour_volume']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 2)
        output = self.model.simulate([0.1, 2, 1, 1, 1], [0, 1])
        self.assertEqual(output.shape, (2, 2))

        # Set to default again
        outputs = ['myokit.tumour_volume']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 1)
        output = self.model.simulate([0.1, 2, 1, 1, 1], [0, 1])
        self.assertEqual(output.shape, (1, 2))

    def test_set_parameter_names(self):
        # Set some parameter names
        names = {
            'myokit.tumour_volume': 'TV',
            'myokit.lambda_0': 'Expon. growth rate'}
        self.model.set_parameter_names(names)
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'TV')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'Expon. growth rate')
        self.assertEqual(parameters[4], 'myokit.lambda_1')

        # Reverse parameter names
        names = {
            'TV': 'myokit.tumour_volume',
            'Expon. growth rate': 'myokit.lambda_0'}
        self.model.set_parameter_names(names)
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')

    def test_set_parameter_names_bad_input(self):
        # List input is not ok!
        names = ['TV', 'some name']

        with self.assertRaisesRegex(TypeError, 'Names has to be a dictionary'):
            self.model.set_parameter_names(names)

    def test_simulate(self):

        times = [0, 1, 2, 3]

        # Test model with bare parameters
        parameters = [0.1, 1, 1, 1, 1]
        output = self.model.simulate(parameters, times)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (1, 4))


class TestPharmacodynamicModel(unittest.TestCase):
    """
    Tests `erlotinib.PharmacodynamicModel`.
    """
    @classmethod
    def setUpClass(cls):
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        cls.model = erlo.PharmacodynamicModel(path)

    def test_n_outputs(self):
        self.assertEqual(self.model.n_outputs(), 1)

    def test_n_parameters(self):
        self.assertEqual(self.model.n_parameters(), 5)

    def test_outputs(self):
        outputs = self.model.outputs()

        self.assertEqual(outputs, ['myokit.tumour_volume'])

    def test_parameters(self):
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')

    def test_pk_input(self):
        pk_input = self.model.pk_input()

        self.assertEqual(pk_input, 'myokit.drug_concentration')

    def test_set_outputs(self):

        # Set bad output
        self.assertRaisesRegex(
            KeyError, 'The variable <', self.model.set_outputs, ['some.thing'])

        # Set twice the same output
        outputs = ['myokit.tumour_volume', 'myokit.tumour_volume']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 2)
        output = self.model.simulate([0.1, 2, 1, 1, 1], [0, 1])
        self.assertEqual(output.shape, (2, 2))

        # Set to default again
        outputs = ['myokit.tumour_volume']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 1)
        output = self.model.simulate([0.1, 2, 1, 1, 1], [0, 1])
        self.assertEqual(output.shape, (1, 2))

    def test_set_parameter_names(self):
        # Set some parameter names
        names = {
            'myokit.tumour_volume': 'TV',
            'myokit.lambda_0': 'Expon. growth rate'}
        self.model.set_parameter_names(names)
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'TV')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'Expon. growth rate')
        self.assertEqual(parameters[4], 'myokit.lambda_1')

        # Reverse parameter names
        names = {
            'TV': 'myokit.tumour_volume',
            'Expon. growth rate': 'myokit.lambda_0'}
        self.model.set_parameter_names(names)
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_0')
        self.assertEqual(parameters[4], 'myokit.lambda_1')

    def test_set_parameter_names_bad_input(self):
        # List input is not ok!
        names = ['TV', 'some name']

        with self.assertRaisesRegex(TypeError, 'Names has to be a dictionary'):
            self.model.set_parameter_names(names)

    def test_set_pk_input(self):
        # Set pk input variable
        pk_input = 'myokit.kappa'
        self.model.set_pk_input(pk_input)

        self.assertEqual(self.model.pk_input(), pk_input)

        # Reset pk input
        pk_input = 'myokit.drug_concentration'
        self.model.set_pk_input(pk_input)

        self.assertEqual(self.model.pk_input(), pk_input)

    def test_set_pk_input_bad_input(self):
        # Set pk input variable
        pk_input = 'SOME NON-EXISTENT VARIABLE'

        with self.assertRaisesRegex(ValueError, 'The name does not'):
            self.model.set_pk_input(pk_input)

    def test_simulate(self):

        times = [0, 1, 2, 3]

        # Test model with bare parameters
        parameters = [0.1, 1, 1, 1, 1]
        output = self.model.simulate(parameters, times)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (1, 4))


class TestPharmacokineticModel(unittest.TestCase):
    """
    Tests `erlotinib.PharmacokineticModel`.
    """
    @classmethod
    def setUpClass(cls):
        path = erlo.ModelLibrary().one_compartment_pk_model()
        cls.model = erlo.PharmacokineticModel(path)

    def test_administration(self):
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)

        # Test default
        self.assertIsNone(model.administration())

        # Administer dose directly to central compartment
        model.set_administration(compartment='central')
        admin = model.administration()

        self.assertIsInstance(admin, dict)
        self.assertEqual(len(admin.keys()), 2)
        self.assertEqual(admin['compartment'], 'central')
        self.assertTrue(admin['direct'])

        # Administer dose indirectly to central compartment
        model.set_administration(compartment='central', direct=False)
        admin = model.administration()

        self.assertIsInstance(admin, dict)
        self.assertEqual(len(admin.keys()), 2)
        self.assertEqual(admin['compartment'], 'central')
        self.assertFalse(admin['direct'])

    def test_n_outputs(self):
        self.assertEqual(self.model.n_outputs(), 1)

    def test_n_parameters(self):
        self.assertEqual(self.model.n_parameters(), 3)

    def test_outputs(self):
        outputs = self.model.outputs()

        self.assertEqual(outputs, ['central.drug_concentration'])

    def test_parameters(self):
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'central.drug_amount')
        self.assertEqual(parameters[1], 'central.size')
        self.assertEqual(parameters[2], 'myokit.elimination_rate')

    def test_pd_output(self):
        pd_output = self.model.pd_output()

        self.assertEqual(pd_output, 'central.drug_concentration')

    def test_set_administration(self):
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)

        # Administer dose directly to central compartment
        model.set_administration(compartment='central')
        parameters = model.parameters()

        self.assertEqual(len(parameters), 3)
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(parameters[0], 'central.drug_amount')
        self.assertEqual(parameters[1], 'central.size')
        self.assertEqual(parameters[2], 'myokit.elimination_rate')

        # Administer dose indirectly to central compartment
        model.set_administration(compartment='central', direct=False)
        parameters = model.parameters()

        self.assertEqual(len(parameters), 5)
        self.assertEqual(model.n_parameters(), 5)
        self.assertEqual(parameters[0], 'central.drug_amount')
        self.assertEqual(parameters[1], 'dose.drug_amount')
        self.assertEqual(parameters[2], 'central.size')
        self.assertEqual(parameters[3], 'dose.absorption_rate')
        self.assertEqual(parameters[4], 'myokit.elimination_rate')

    def test_set_administration_bad_input(self):
        # Bad compartment
        with self.assertRaisesRegex(ValueError, 'The model does not'):
            self.model.set_administration(compartment='SOME COMP')

        # Bad amount variable (not existent)
        with self.assertRaisesRegex(ValueError, 'The drug amount variable'):
            self.model.set_administration(
                compartment='central', amount_var='SOME VARIABLE')

        # Bad amount variable (not state)
        with self.assertRaisesRegex(ValueError, 'The variable <'):
            self.model.set_administration(
                compartment='central', amount_var='drug_concentration')

    def test_set_dosing_regimen(self):
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)

        # Administer dose directly to central compartment
        model.set_administration(compartment='central')

        # Test case I: Single bolus dose
        dose = 1
        start = 5
        model.set_dosing_regimen(dose, start)

        events = model.dosing_regimen().events()
        self.assertEqual(len(events), 1)

        num = 0
        period = 0
        duration = 0.01
        event = events[0]
        self.assertEqual(event.level(), dose / duration)
        self.assertEqual(event.start(), start)
        self.assertEqual(event.period(), period)
        self.assertEqual(event.duration(), duration)
        self.assertEqual(event.multiplier(), num)

        # Test case II: Single infusion
        dose = 1
        start = 5
        duration = 1
        model.set_dosing_regimen(dose, start, duration)

        events = model.dosing_regimen().events()
        self.assertEqual(len(events), 1)

        num = 0
        period = 0
        event = events[0]
        self.assertEqual(event.level(), dose / duration)
        self.assertEqual(event.start(), start)
        self.assertEqual(event.period(), period)
        self.assertEqual(event.duration(), duration)
        self.assertEqual(event.multiplier(), num)

        # Test case III: Infinitely many doses
        dose = 1
        start = 5
        period = 1
        model.set_dosing_regimen(dose, start, period=period)

        events = model.dosing_regimen().events()
        self.assertEqual(len(events), 1)

        num = 0
        duration = 0.01
        event = events[0]
        self.assertEqual(event.level(), dose / duration)
        self.assertEqual(event.start(), start)
        self.assertEqual(event.period(), period)
        self.assertEqual(event.duration(), duration)
        self.assertEqual(event.multiplier(), num)

        # Test case IV: Finitely many doses
        dose = 10
        start = 3
        duration = 0.01
        period = 5
        num = 4
        model.set_dosing_regimen(
            dose, start, duration, period, num)

        events = model.dosing_regimen().events()
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(event.level(), dose/duration)
        self.assertEqual(event.start(), start)
        self.assertEqual(event.period(), period)
        self.assertEqual(event.duration(), duration)
        self.assertEqual(event.multiplier(), num)

    def test_set_dosing_regimen_bad_input(self):
        # Not setting an administration prior to setting a dosing regimen
        # should raise an error
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)

        with self.assertRaisesRegex(ValueError, 'The route of administration'):
            model.set_dosing_regimen(dose=10, start=3, period=5)

    def test_set_outputs(self):

        # Set bad output
        self.assertRaisesRegex(
            KeyError, 'The variable <', self.model.set_outputs, ['some.thing'])

        # Set two outputs
        outputs = ['central.drug_amount', 'central.drug_concentration']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 2)
        output = self.model.simulate([0.1, 2, 1, 1, 1], [0, 1])
        self.assertEqual(output.shape, (2, 2))

        # Set to default again
        outputs = ['central.drug_concentration']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 1)
        output = self.model.simulate([0.1, 2, 1, 1, 1], [0, 1])
        self.assertEqual(output.shape, (1, 2))

    def test_set_parameter_names(self):
        # Set some parameter names
        names = {
            'central.drug_amount': 'A',
            'myokit.elimination_rate': 'k_e'}
        self.model.set_parameter_names(names)
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'A')
        self.assertEqual(parameters[1], 'central.size')
        self.assertEqual(parameters[2], 'k_e')

        # Reverse parameter names
        names = {
            'A': 'central.drug_amount',
            'k_e': 'myokit.elimination_rate'}
        self.model.set_parameter_names(names)
        parameters = self.model.parameters()

        self.assertEqual(parameters[0], 'central.drug_amount')
        self.assertEqual(parameters[1], 'central.size')
        self.assertEqual(parameters[2], 'myokit.elimination_rate')

    def test_set_parameter_names_bad_input(self):
        # List input is not ok!
        names = ['TV', 'some name']

        with self.assertRaisesRegex(TypeError, 'Names has to be a dictionary'):
            self.model.set_parameter_names(names)

    def test_set_pd_output(self):
        # Set pd output variable
        pd_output = 'central.drug_amount'
        self.model.set_pd_output(pd_output)

        self.assertEqual(self.model.pd_output(), pd_output)

        # Reset pd output
        pd_output = 'central.drug_concentration'
        self.model.set_pd_output(pd_output)

        self.assertEqual(self.model.pd_output(), pd_output)

    def test_set_pd_output_bad_input(self):
        # Set pd output variable
        pd_output = 'SOME NON-EXISTENT VARIABLE'

        with self.assertRaisesRegex(ValueError, 'The name does not'):
            self.model.set_pd_output(pd_output)

    def test_simulate(self):

        times = [0, 1, 2, 3]

        # Test model with bare parameters
        parameters = [1, 1, 1]
        output = self.model.simulate(parameters, times)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (1, 4))


if __name__ == '__main__':
    unittest.main()
