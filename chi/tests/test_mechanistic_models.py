#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os
import unittest

import myokit
import numpy as np

import chi
from chi.library import ModelLibrary


class ToyMechanisticModel(chi.MechanisticModel):
    """
    Minimal implementation of a chi.MechanisticModel.
    """
    def __init__(self):
        super(ToyMechanisticModel, self).__init__()
        self._has_sensitivities = False
        self._n_outputs = 1
        self._n_parameters = 1

    def enable_sensitivities(self, enabled, parameter_names=None):
        self._has_sensitivities = bool(enabled)

    def has_sensitivities(self):
        return self._has_sensitivities

    def n_outputs(self):
        return self._n_outputs

    def n_parameters(self):
        return self._n_parameters

    def outputs(self):
        return ['Output']

    def parameters(self):
        return ['Parameter']

    def simulate(self, parameters, times):
        n_times = len(times)
        output = parameters[0]
        outputs = np.full(
            shape=(n_times, self._n_outputs), fill_value=output)
        if not self._has_sensitivities:
            return outputs

        sensitivities = np.ones(
            shape=((n_times, self._n_outputs, self._n_parameters)))

        return outputs, sensitivities


class TestMechanisticModel(unittest.TestCase):
    """
    Tests `chi.MechanisticModel`.
    """
    @classmethod
    def setUpClass(cls):
        cls.model = chi.MechanisticModel()

    def test_copy(self):
        model = self.model.copy()
        self.assertIsInstance(model, chi.MechanisticModel)

    def test_enable_sensitivities(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.model.enable_sensitivities(True)

    def test_has_sensitivities(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.model.has_sensitivities()

    def test_n_outputs(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.model.n_outputs()

    def test_n_parameters(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.model.n_parameters()

    def test_outputs(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.model.outputs()

    def test_parameters(self):
        with self.assertRaisesRegex(NotImplementedError, None):
            self.model.parameters()

    def test_simulate(self):
        parameters = 'some params'
        times = 'some times'
        with self.assertRaisesRegex(NotImplementedError, None):
            self.model.simulate(parameters, times)

    def test_supports_dosing(self):
        self.assertFalse(self.model.supports_dosing())


class TestSBMLModel(unittest.TestCase):
    """
    Tests `chi.SBMLModel`.
    """
    @classmethod
    def setUpClass(cls):
        cls.model = ModelLibrary().tumour_growth_inhibition_model_koch()

    def test_model_is_sbml_model(self):
        self.assertIsInstance(self.model, chi.SBMLModel)

    def test_copy(self):
        # Case I: Copy model and check that all public properties coincide
        model = self.model.copy()

        self.assertFalse(model.has_sensitivities())
        self.assertFalse(self.model.has_sensitivities())
        self.assertEqual(model.n_outputs(), 1)
        self.assertEqual(self.model.n_outputs(), 1)
        outputs_c = model.outputs()
        outputs = self.model.outputs()
        self.assertEqual(outputs_c[0], 'myokit.tumour_volume')
        self.assertEqual(outputs[0], 'myokit.tumour_volume')
        self.assertEqual(model.n_parameters(), 5)
        self.assertEqual(self.model.n_parameters(), 5)
        params_c = model.parameters()
        params = self.model.parameters()
        self.assertEqual(params_c[0], 'myokit.tumour_volume')
        self.assertEqual(params_c[1], 'myokit.drug_concentration')
        self.assertEqual(params_c[2], 'myokit.kappa')
        self.assertEqual(params_c[3], 'myokit.lambda_0')
        self.assertEqual(params_c[4], 'myokit.lambda_1')
        self.assertEqual(params[0], 'myokit.tumour_volume')
        self.assertEqual(params[1], 'myokit.drug_concentration')
        self.assertEqual(params[2], 'myokit.kappa')
        self.assertEqual(params[3], 'myokit.lambda_0')
        self.assertEqual(params[4], 'myokit.lambda_1')

        # Change output name
        model.set_output_names({'myokit.tumour_volume': 'test'})
        self.assertEqual(model.n_outputs(), 1)
        self.assertEqual(self.model.n_outputs(), 1)
        outputs_c = model.outputs()
        outputs = self.model.outputs()
        self.assertEqual(outputs_c[0], 'test')
        self.assertEqual(outputs[0], 'myokit.tumour_volume')

        # Set new outputs
        model.set_outputs(
            ['myokit.tumour_volume', 'myokit.tumour_volume'])
        self.assertEqual(model.n_outputs(), 2)
        self.assertEqual(self.model.n_outputs(), 1)
        outputs_c = model.outputs()
        outputs = self.model.outputs()
        self.assertEqual(outputs_c[0], 'test')
        self.assertEqual(outputs_c[1], 'test')
        self.assertEqual(outputs[0], 'myokit.tumour_volume')
        model.set_outputs(['myokit.tumour_volume'])

        # Rename some parameters
        model.set_parameter_names({
            'myokit.kappa': 'new 1',
            'myokit.lambda_0': 'new 2'})
        self.assertEqual(model.n_parameters(), 5)
        self.assertEqual(self.model.n_parameters(), 5)
        params_c = model.parameters()
        params = self.model.parameters()
        self.assertEqual(params_c[0], 'myokit.tumour_volume')
        self.assertEqual(params_c[1], 'myokit.drug_concentration')
        self.assertEqual(params_c[2], 'new 1')
        self.assertEqual(params_c[3], 'new 2')
        self.assertEqual(params_c[4], 'myokit.lambda_1')
        self.assertEqual(params[0], 'myokit.tumour_volume')
        self.assertEqual(params[1], 'myokit.drug_concentration')
        self.assertEqual(params[2], 'myokit.kappa')
        self.assertEqual(params[3], 'myokit.lambda_0')
        self.assertEqual(params[4], 'myokit.lambda_1')

    def test_enable_sensitivities(self):
        # Enable sensitivities
        self.model.enable_sensitivities(True)
        self.assertTrue(self.model.has_sensitivities())

        # Enable sensitivties for a subset of parameters
        parameters = ['myokit.tumour_volume', 'myokit.kappa']
        self.model.enable_sensitivities(True, parameters)
        self.assertTrue(self.model.has_sensitivities())

        # Disable sensitivities
        self.model.enable_sensitivities(False)
        self.assertFalse(self.model.has_sensitivities())

        # Disable sensitvities a second time
        self.model.enable_sensitivities(False)
        self.assertFalse(self.model.has_sensitivities())

    def test_enable_sensitivities_bad_input(self):
        # Specify parameter names that cannot be identified
        parameters = ['do', 'not', 'exist']
        with self.assertRaisesRegex(ValueError, 'None of the'):
            self.model.enable_sensitivities(True, parameters)

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

        # Test case II: Model where lambda_0 is defined as lambda_1 * kappa
        directory = os.path.dirname(__file__)
        model = chi.SBMLModel(
            directory + '/data/model_with_derived_constants.xml')
        parameters = model.parameters()

        self.assertEqual(len(parameters), 4)
        self.assertEqual(model.n_parameters(), 4)
        self.assertEqual(parameters[0], 'myokit.tumour_volume')
        self.assertEqual(parameters[1], 'myokit.drug_concentration')
        self.assertEqual(parameters[2], 'myokit.kappa')
        self.assertEqual(parameters[3], 'myokit.lambda_1')

    def test_set_outputs(self):
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

    def test_set_ouputs_bad_input(self):
        # Set not existing output output
        self.assertRaisesRegex(
            KeyError, 'The variable <', self.model.set_outputs, ['some.thing'])

        # Set output to constant
        with self.assertRaisesRegex(ValueError, 'Outputs have to be state or'):
            self.model.set_outputs(['myokit.kappa'])

    def test_set_output_names(self):
        # Set output name
        names = {'myokit.tumour_volume': 'Some name'}
        self.model.set_output_names(names)
        outputs = self.model.outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'Some name')

        # Provide a dictionary of unnecessary names
        names = {'Some name': 'Some other name', 'does not exist': 'bla'}
        self.model.set_output_names(names)
        outputs = self.model.outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'Some other name')

        # Set name back to default
        names = {'Some other name': 'myokit.tumour_volume'}
        self.model.set_output_names(names)
        outputs = self.model.outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'myokit.tumour_volume')

    def test_set_output_names_bad_input(self):
        # List input is not ok!
        names = ['TV', 'some name']
        with self.assertRaisesRegex(TypeError, 'Names has to be a dictionary'):
            self.model.set_output_names(names)

        # New names are not unique
        names = {'param 1': 'Some name', 'param 2': 'Some name'}
        with self.assertRaisesRegex(ValueError, 'The new output names'):
            self.model.set_output_names(names)

        # New names exist already
        names = {'param 1': 'myokit.tumour_volume', 'param 2': 'Some name'}
        with self.assertRaisesRegex(ValueError, 'The output names cannot'):
            self.model.set_output_names(names)

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

        # New names are not unique
        names = {'param 1': 'Some name', 'param 2': 'Some name'}
        with self.assertRaisesRegex(ValueError, 'The new parameter names'):
            self.model.set_parameter_names(names)

        # New names exist already
        names = {'param 1': 'myokit.tumour_volume', 'param 2': 'Some name'}
        with self.assertRaisesRegex(ValueError, 'The parameter names cannot'):
            self.model.set_parameter_names(names)

    def test_simulate(self):
        times = [0, 1, 2, 3]

        # Test simulation of output
        parameters = [0.1, 1, 1, 1, 1]
        output = self.model.simulate(parameters, times)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (1, 4))

        # Test simulation with sensitivities
        parameters = [0.1, 1, 1, 1, 1]
        self.model.enable_sensitivities(True)
        output, sens = self.model.simulate(parameters, times)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (1, 4))
        self.assertIsInstance(sens, np.ndarray)
        self.assertEqual(sens.shape, (4, 1, 5))

    def test_supports_dosing(self):
        self.assertFalse(self.model.supports_dosing())


class TestPKPDModel(unittest.TestCase):
    """
    Tests `chi.PKPDModel`.
    """
    @classmethod
    def setUpClass(cls):
        cls.model = ModelLibrary().one_compartment_pk_model()

    def test_model_is_pkpd_model(self):
        self.assertIsInstance(self.model, chi.PKPDModel)

    def test_administration(self):
        model = ModelLibrary().one_compartment_pk_model()

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

    def test_enable_sensitivities(self):
        model = ModelLibrary().one_compartment_pk_model()

        # Disable sensitivities before setting administration
        model.enable_sensitivities(False)
        self.assertFalse(model.has_sensitivities())

        # Set administration and check that sensitivities are still
        # disabled
        model.set_administration(compartment='central', direct=False)
        self.assertFalse(model.has_sensitivities())

        # Enable sensitivities
        model.enable_sensitivities(True)
        self.assertTrue(model.has_sensitivities())
        times = [0, 1, 2, 3]
        parameters = [1, 1, 1, 1, 1]
        output, sens = model.simulate(parameters, times)
        self.assertEqual(output.shape, (1, 4))
        self.assertEqual(sens.shape, (4, 1, 5))

        # Enable sensitivities before setting an administration
        model = ModelLibrary().one_compartment_pk_model()
        model.enable_sensitivities(True)
        self.assertTrue(model.has_sensitivities())
        times = [0, 1, 2, 3]
        parameters = [1, 1, 1]
        output, sens = model.simulate(parameters, times)
        self.assertEqual(output.shape, (1, 4))
        self.assertEqual(sens.shape, (4, 1, 3))

        # Set administration
        model.set_administration(compartment='central', direct=False)
        self.assertFalse(model.has_sensitivities())

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

    def test_set_administration(self):
        model = ModelLibrary().one_compartment_pk_model()

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
        model = ModelLibrary().one_compartment_pk_model()

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

        # Test case V: Set myokit.Protocol
        protocol = myokit.pacing.blocktrain(period=1, duration=0.1, limit=1)
        model.set_dosing_regimen(protocol)

        events = model.dosing_regimen().events()
        self.assertEqual(len(events), 1)

        start = 0
        num = 1
        period = 1
        duration = 0.1
        event = events[0]
        self.assertEqual(event.level(), 1)
        self.assertEqual(event.start(), start)
        self.assertEqual(event.period(), period)
        self.assertEqual(event.duration(), duration)
        self.assertEqual(event.multiplier(), num)

    def test_set_dosing_regimen_bad_input(self):
        # Not setting an administration prior to setting a dosing regimen
        # should raise an error
        model = ModelLibrary().one_compartment_pk_model()

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

    def set_output_names(self):
        # Rename only one output
        outputs = ['central.drug_amount', 'central.drug_concentration']
        self.model.set_outputs(outputs)
        self.model.set_output_names({'central.drug_amount': 'Some name'})
        outputs = self.model.outputs()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], 'Some name')
        self.assertEqual(outputs[1], 'central.drug_concentration')

        # Rename the other output
        self.model.set_output_names({
            'central.drug_concentration': 'Some other name'})
        outputs = self.model.outputs()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], 'Some name')
        self.assertEqual(outputs[1], 'Some other name')

        # Set output back to default
        outputs = ['Some name']
        self.model.set_outputs(outputs)
        self.assertEqual(self.model.outputs(), outputs)
        self.assertEqual(self.model.n_outputs(), 1)

        # Set output name to default
        self.model.set_output_names({'Some name': 'central.drug_amount'})
        outputs = self.model.outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'central.drug_amount')

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

    def test_simulate(self):
        times = [0, 1, 2, 3]

        # Test model with bare parameters
        parameters = [1, 1, 1]
        output = self.model.simulate(parameters, times)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (1, 4))

    def test_supports_dosing(self):
        self.assertTrue(self.model.supports_dosing())


class TestReducedMechanisticModel(unittest.TestCase):
    """
    Tests the chi.ReducedMechanisticModel class.
    """

    @classmethod
    def setUpClass(cls):
        # Set up model
        cls.pd_model = chi.ReducedMechanisticModel(
            ToyMechanisticModel())

        model = ModelLibrary().one_compartment_pk_model()
        model.set_administration('central')
        cls.pkpd_model = chi.ReducedMechanisticModel(model)

    def test_bad_instantiation(self):
        model = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The mechanistic model'):
            chi.ReducedMechanisticModel(model)

    def test_copy(self):
        # Fix some parameters
        self.pkpd_model.fix_parameters({
            'central.size': 1})

        # Copy model and make sure the are identical
        model = self.pkpd_model.copy()

        self.assertFalse(model.has_sensitivities())
        self.assertFalse(self.pkpd_model.has_sensitivities())
        self.assertEqual(model.n_outputs(), 1)
        self.assertEqual(self.pkpd_model.n_outputs(), 1)
        outputs_c = model.outputs()
        outputs = self.pkpd_model.outputs()
        self.assertEqual(outputs_c[0], 'central.drug_concentration')
        self.assertEqual(outputs[0], 'central.drug_concentration')
        self.assertEqual(model.n_parameters(), 2)
        self.assertEqual(self.pkpd_model.n_parameters(), 2)
        params_c = model.parameters()
        params = self.pkpd_model.parameters()
        self.assertEqual(params_c[0], 'central.drug_amount')
        self.assertEqual(params_c[1], 'myokit.elimination_rate')
        self.assertEqual(params[0], 'central.drug_amount')
        self.assertEqual(params[1], 'myokit.elimination_rate')

        # Rename the output
        model.set_output_names({'central.drug_concentration': 'test'})
        self.assertEqual(model.n_outputs(), 1)
        self.assertEqual(self.pkpd_model.n_outputs(), 1)
        outputs_c = model.outputs()
        outputs = self.pkpd_model.outputs()
        self.assertEqual(outputs_c[0], 'test')
        self.assertEqual(outputs[0], 'central.drug_concentration')

        # Set new ouputs
        model.set_outputs(['test', 'central.drug_amount'])
        self.assertEqual(model.n_outputs(), 2)
        self.assertEqual(self.pkpd_model.n_outputs(), 1)
        outputs_c = model.outputs()
        outputs = self.pkpd_model.outputs()
        self.assertEqual(outputs_c[0], 'test')
        self.assertEqual(outputs_c[1], 'central.drug_amount')
        self.assertEqual(outputs[0], 'central.drug_concentration')

        # Fix different parameters
        model.fix_parameters({
            'central.size': None,
            'central.drug_amount': 1})
        self.assertEqual(model.n_parameters(), 2)
        self.assertEqual(self.pkpd_model.n_parameters(), 2)
        params_c = model.parameters()
        params = self.pkpd_model.parameters()
        self.assertEqual(params_c[0], 'central.size')
        self.assertEqual(params_c[1], 'myokit.elimination_rate')
        self.assertEqual(params[0], 'central.drug_amount')
        self.assertEqual(params[1], 'myokit.elimination_rate')

        # Unfix model parameters
        self.pkpd_model.fix_parameters({'central.size': None})

    def test_enable_sensitivities(self):
        # Enable sensitivities
        self.pd_model.enable_sensitivities(True)
        self.assertTrue(self.pd_model.has_sensitivities())

        # Disable sensitvities
        self.pd_model.enable_sensitivities(False)
        self.assertFalse(self.pd_model.has_sensitivities())

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        self.pd_model.fix_parameters(name_value_dict={
            'Parameter': 1})

        n_parameters = self.pd_model.n_parameters()
        self.assertEqual(n_parameters, 0)

        parameter_names = self.pd_model.parameters()
        self.assertEqual(len(parameter_names), 0)

        # Test case II: unfix all parameters
        self.pd_model.fix_parameters(name_value_dict={
            'Parameter': None})

        n_parameters = self.pd_model.n_parameters()
        self.assertEqual(n_parameters, 1)

        parameter_names = self.pd_model.parameters()
        self.assertEqual(len(parameter_names), 1)
        self.assertEqual(parameter_names[0], 'Parameter')

    def test_fix_parameters_bad_input(self):
        name_value_dict = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The name-value dictionary'):
            self.pd_model.fix_parameters(name_value_dict)

    def test_mechanistic_model(self):
        self.assertIsInstance(
            self.pd_model.mechanistic_model(), chi.MechanisticModel)

    def test_n_outputs(self):
        self.assertEqual(self.pd_model.n_outputs(), 1)

    def test_n_parameters(self):
        self.assertEqual(self.pd_model.n_parameters(), 1)

    def test_set_get_dosing_regimen(self):
        # Test case I: dosing regimen unset
        # Test PD model
        self.assertIsNone(self.pd_model.dosing_regimen())

        # Test PK model
        self.assertIsNone(self.pkpd_model.dosing_regimen())

        # Test case II: dosing regimen set
        # Test PD model
        with self.assertRaisesRegex(AttributeError, 'The mechanistic model'):
            self.pd_model.set_dosing_regimen(1, 1)
        self.assertIsNone(self.pd_model.dosing_regimen())

        # Test PK model
        self.pkpd_model.set_dosing_regimen(1, 1)
        self.assertIsInstance(
            self.pkpd_model.dosing_regimen(), myokit.Protocol)

    def test_set_get_outputs(self):
        # Test case I: Set outputs
        self.pkpd_model.set_outputs([
            'central.drug_amount',
            'central.drug_amount'])

        outputs = self.pkpd_model.outputs()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], 'central.drug_amount')
        self.assertEqual(outputs[1], 'central.drug_amount')

        self.pkpd_model.set_output_names(
            {'central.drug_amount': 'Some name'})
        outputs = self.pkpd_model.outputs()
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], 'Some name')
        self.assertEqual(outputs[1], 'Some name')

        # Test case II: Set outputs back to default
        self.pkpd_model.set_output_names(
            {'Some name': 'central.drug_amount'})
        self.pkpd_model.set_outputs(['central.drug_concentration'])

        outputs = self.pkpd_model.outputs()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0], 'central.drug_concentration')

        # Test case III: Set ouputs and output names for chi.MechanisticModel
        with self.assertRaisesRegex(NotImplementedError, 'The mechanistic'):
            self.pd_model.set_outputs(['some names'])
        with self.assertRaisesRegex(NotImplementedError, 'The mechanistic'):
            self.pd_model.set_output_names('some names')

    def test_set_get_parameters(self):
        # Test case I: set some parameter names
        self.pkpd_model.set_parameter_names({
            'central.drug_amount': 'Test'})

        parameters = self.pkpd_model.parameters()
        self.assertEqual(len(parameters), 3)
        self.assertEqual(parameters[0], 'Test')
        self.assertEqual(parameters[1], 'central.size')
        self.assertEqual(parameters[2], 'myokit.elimination_rate')

        # Test case II: set back to default
        self.pkpd_model.set_parameter_names({
            'Test': 'central.drug_amount'})

        parameters = self.pkpd_model.parameters()
        self.assertEqual(len(parameters), 3)
        self.assertEqual(parameters[0], 'central.drug_amount')
        self.assertEqual(parameters[1], 'central.size')
        self.assertEqual(parameters[2], 'myokit.elimination_rate')

        # Test case III: Set parameter names for chi.MechanisticModel
        with self.assertRaisesRegex(NotImplementedError, 'The mechanistic'):
            self.pd_model.set_parameter_names('some names')

    def test_simulate(self):
        # Test case I: fix some parameters
        self.pd_model.fix_parameters(name_value_dict={
            'Parameter': 2})

        # Simulate
        times = [1, 2, 3]
        parameters = []
        output = self.pd_model.simulate(parameters, times).flatten()

        # Simulate unfixed model with the same parameters
        model = self.pd_model.mechanistic_model()
        parameters = [2]
        ref_output = model.simulate(parameters, times).flatten()

        self.assertEqual(len(output), 3)
        self.assertEqual(len(ref_output), 3)
        self.assertEqual(output[0], ref_output[0])
        self.assertEqual(output[1], ref_output[1])
        self.assertEqual(output[2], ref_output[2])

        # Enable sensitivities
        self.pkpd_model.fix_parameters(name_value_dict={
            'central.drug_amount': 2})
        self.pkpd_model.enable_sensitivities(True)

        # Simulate
        times = [1, 2, 3]
        parameters = [0.5, 0.3]
        output, sens = self.pkpd_model.simulate(parameters, times)
        output = output.squeeze()

        # Simulate unfixed model with the same parameters
        model = self.pkpd_model.mechanistic_model()
        parameters = [2, 0.5, 0.3]
        ref_output, ref_sens = model.simulate(parameters, times)
        ref_output = ref_output.squeeze()

        self.assertEqual(len(output), 3)
        self.assertEqual(len(ref_output), 3)
        self.assertEqual(output[0], ref_output[0])
        self.assertEqual(output[1], ref_output[1])
        self.assertEqual(output[2], ref_output[2])

        self.assertEqual(sens.shape, (3, 1, 2))
        self.assertEqual(ref_sens.shape, (3, 1, 2))
        self.assertEqual(sens[0, 0, 0], ref_sens[0, 0, 0])
        self.assertEqual(sens[1, 0, 0], ref_sens[1, 0, 0])
        self.assertEqual(sens[2, 0, 0], ref_sens[2, 0, 0])
        self.assertEqual(sens[0, 0, 1], ref_sens[0, 0, 1])
        self.assertEqual(sens[1, 0, 1], ref_sens[1, 0, 1])
        self.assertEqual(sens[2, 0, 1], ref_sens[2, 0, 1])

        # Unfix parameters
        self.pd_model.fix_parameters(name_value_dict={
            'Parameter': None})
        self.pkpd_model.fix_parameters(name_value_dict={
            'central.drug_amount': None})

    def test_supports_dosing(self):
        self.assertTrue(self.pkpd_model.supports_dosing())
        self.assertFalse(self.pd_model.supports_dosing())

    def test_time_unit(self):
        self.assertIsInstance(self.pkpd_model.time_unit(), myokit.Unit)

        with self.assertRaisesRegex(NotImplementedError, 'The mechanistic'):
            self.pd_model.time_unit()


if __name__ == '__main__':
    unittest.main()
