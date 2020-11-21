#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import myokit
import myokit.formats.sbml as sbml
import numpy as np


class Model(object):
    """
    A base class for models that are specified by sbml files and converted to
    :class:`myokit.Model`s that can be simulated.

    Parameters
    ----------
    sbml_file
        A path to the SBML model file that specifies the model.

    """

    def __init__(self, sbml_file):
        super(Model, self).__init__()

        model = sbml.SBMLImporter().model(sbml_file)

        # Get the number of states and parameters
        self._n_states = model.count_states()
        n_const = model.count_variables(const=True)
        self._n_parameters = self._n_states + n_const

        # Get constant variable names and state names
        self._state_names = sorted(
            [var.qname() for var in model.states()])
        self._const_names = sorted(
            [var.qname() for var in model.variables(const=True)])

        # Get time unit
        self._time_unit = self._get_time_unit(model)

        # Set default parameter names
        self._parameter_names = self._state_names + self._const_names

        # Set default outputs
        self._output_names = self._state_names
        self._n_outputs = self._n_states

        # Create simulator
        self._sim = myokit.Simulation(model)

    def _get_time_unit(self, model):
        """
        Gets the model's time unit.
        """
        # Get bound variables
        bound_variables = [var for var in model.variables(bound=True)]

        # Get the variable that is bound to time
        # (only one can exist in myokit.Model)
        for var in bound_variables:
            if var._binding == 'time':
                return var.unit()

    def _set_const(self, parameters):
        """
        Sets values of constant model parameters.
        """
        for id_var, var in enumerate(self._const_names):
            self._sim.set_constant(var, float(parameters[id_var]))

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the number of states.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        return self._n_parameters

    def outputs(self):
        """
        Returns the output names of the model.
        """
        return self._output_names

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        return self._parameter_names

    def set_outputs(self, outputs):
        """
        Sets outputs of the model.

        Parameters
        ----------
        outputs
            A list of quantifiable variable names of the :class:`myokit.Model`,
            e.g. `compartment.variable`.
        """
        # Check that outputs are valid
        for output in outputs:
            try:
                self._sim._model.get(output)
            except KeyError:
                raise KeyError(
                    'The variable <' + str(output) + '> does not exist in the '
                    'model.')

        self._output_names = outputs
        self._n_outputs = len(outputs)

    def set_parameter_names(self, names):
        """
        Assigns names to the parameters. By default the :class:`myokit.Model`
        names are assigned to the parameters.

        Parameters
        ----------
        names
            A dictionary that maps the current parameter names to new names.
        """
        if not isinstance(names, dict):
            raise TypeError(
                'Names has to be a dictionary with the current parameter names'
                'as keys and the new parameter names as values.')

        parameter_names = self._parameter_names
        for index, parameter in enumerate(self._parameter_names):
            try:
                parameter_names[index] = str(names[parameter])
            except KeyError:
                # KeyError indicates that a current parameter is not being
                # replaced.
                pass

        self._parameter_names = parameter_names

        # Rename pk input
        try:
            self._pk_input = str(names[self._pk_input])
        except KeyError:
            # KeyError indicates that the current name is not being
            # replaced.
            pass

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs for specified
        parameters and times.

        The result is returned as a 2 dimensional NumPy array of shape
        (n_outputs, n_times).

        Parameters
        ----------
        parameters
            An array-like object with values for the model parameters.
        times
            An array-like object with time points at which the output
            values are returned.
        """
        # Reset simulation
        self._sim.reset()

        # Set initial conditions
        self._sim.set_state(parameters[:self._n_states])

        # Set constant model parameters
        self._set_const(parameters[self._n_states:])

        # Simulate
        output = self._sim.run(
            times[-1] + 1, log=self._output_names, log_times=times)
        result = [output[name] for name in self._output_names]

        return np.array(result)

    def time_unit(self):
        """
        Returns the model's unit of time.
        """
        return self._time_unit


class PharmacodynamicModel(Model):
    """
    Converts a pharmacodynamic model specified by an SBML file into a forward
    model that can be solved numerically.

    Parameters
    ----------
    sbml_file
        A path to the SBML model file that specifies the pharmacodynamic model.

    """

    def __init__(self, sbml_file):
        super(PharmacodynamicModel, self).__init__(sbml_file)

        # Set default pharmacokinetic input variable
        # (Typically drug concentration)
        self._pk_input = None
        if 'myokit.drug_concentration' in self._parameter_names:
            self._pk_input = 'myokit.drug_concentration'

    def pk_input(self):
        """
        Returns the pharmacokinetic input variable. In most models this will be
        the concentration of the drug.

        Defaults to ``None`` or ``myokit.drug_concentration`` if the latter is
        among the model parameters.
        """
        return self._pk_input

    def set_pk_input(self, name):
        """
        Sets the pharmacokinetic input variable. In most models this will be
        the concentration of the drug.

        The name has to match a parameter of the model.
        """
        if name not in self._parameter_names:
            raise ValueError(
                'The name does not match a model parameter.')

        self._pk_input = name


class PharmacokineticModel(Model):
    """
    Converts a pharmacokinetic model specified by an SBML file into a forward
    model that can be solved numerically.

    Parameters
    ----------
    sbml_file
        A path to the SBML model file that specifies the pharmacokinetic model.

    """

    def __init__(self, sbml_file):
        super(PharmacokineticModel, self).__init__(sbml_file)

        # Set default dose input and regimen
        self._dose_input = None

        # Set default output variable that interacts with the pharmacodynamic
        # model
        # (Typically drug concentration in central compartment)
        self._pd_output = None
        if 'central.drug_concentration' in self._parameter_names:
            self._pd_output = 'central.drug_concentration'

    def dose_input(self):
        """
        Returns the dose input variable.

        Any dose is administered directly to this variable.

        By default no dose input variable is set and ``None`` is returned.

        The dose input variable can be set with :meth:`set_dose_input`.
        """
        return self._dose_input

    def set_dose_input(self, input_name, dose_rate_name='dose_rate'):
        r"""
        Administers the dose to the selected variable, often the drug amount
        in the central or dose compartment.

        The selected variable :math:`A` has to be a state variable. A dose
        rate variable :math:`r` will be added to the rate expression of
        :math:`A`

        .. math::

            \frac{\text{d}A}{\text{d}t} = \text{RHS} + r,

        where :math:`\text{RHS}` symbolises the previous rate expression of
        :math:`A`.

        The value of :math:`r` is determined by the chosen dosing regimen.

        Parameters
        ----------
        name
            A quantifiable variable name of the :class:`myokit.Model`, e.g.
            `compartment.variable` to which the dose is administered.
        """
        if self._dose_input is not None:
            raise ValueError(
                'Dose input has been set before. To change the dose input '
                'please instantiate the model again.')

        # Get variable
        try:
            drug_amount = self._sim._model.get(input_name)
        except KeyError:
            raise KeyError(
                'The variable <' + str(input_name) + '> does not exist in the '
                'model.')

        if not drug_amount.is_state():
            raise ValueError(
                'The variable <' + str(drug_amount) + '> is not a state '
                'variable, and can therefore not be dosed directly.')

        # Register the dose rate variable to the compartment and bind it to
        # pace, i.e. tells myokit that its value is set by the dosing regimen/
        # myokit.Protocol
        compartment = drug_amount.parent()
        dose_rate = compartment.add_variable_allow_renaming(
            str(dose_rate_name))
        dose_rate.set_binding('pace')

        # Set initial value to 0 and unit to unit of drug amount over unit of
        # time
        dose_rate.set_rhs(0)
        dose_rate.set_unit(drug_amount.unit() / self.time_unit())

        # Add the dose rate to the rhs of the drug amount variable
        rhs = drug_amount.rhs()
        drug_amount.set_rhs(
            myokit.Plus(
                rhs,
                myokit.Name(dose_rate)
            )
        )

        # Remember the dose input variable and dose rate variable
        self._dose_input = input_name

        # Update simulator
        # (otherwise simulator won't know about pace bound variable)
        self._sim = myokit.Simulation(self._sim._model)

    def set_dosing_regimen(self, dose, start, period, duration=0.01, num=0):
        """
        Sets the dosing regimen with which the compound is administered.

        The route of administration is partially determined by setting the
        :meth:`dose_input` variable, and by setting the duration of the
        administration.

        By default the dose is administered as a bolus injection (duration on
        a time scale that is 100 fold smaller than the basic time unit). To
        model a infusion of the dose over a longer time period, the
        ``duration`` can be adjusted to the appropriate time scale.

        By default the doses are administered periodically indefinitely. If
        only a finite number of doses are applied, ``num`` can be set to a
        positive integer.

        Parameters
        ----------
        dose
            The amount of the compound that is injected at each administration.
        start
            Start time of the treatment.
        period
            Periodicity at which doses are administered.
        duration
            Duration of dose administration. For bolus injection setting the
            duration to 1% of the time unit should suffice. By default the
            duration is set to 0.01 (bolus).
        num
            Number of administered doses. For ``num=0`` the dose is applied
            indefinitely. By default ``num`` is set to ``0``.
        """
        if self._dose_input is None:
            raise ValueError(
                'The dose input of the model has not been set.')

        # Translate dose to dose rate
        dose_rate = dose / duration

        # Set dosing regimen
        dosing_regimen = myokit.pacing.blocktrain(
            period=period, duration=duration, offset=start, level=dose_rate,
            limit=num)
        self._sim.set_protocol(dosing_regimen)

    def pd_output(self):
        """
        Returns the variable which interacts with the pharmacodynamic model.
        In most models this will be the concentration of the drug in the
        central compartment.

        This variable is mapped to the
        :meth:`erlotinib.PharmacodynamicModel.pk_input` variable when a
        :class:`PKPDModel` is instantiated.

        Defaults to ``None`` or ``central.drug_concentration`` if the latter is
        among the model parameters.
        """
        return self._pd_output

    def set_pd_output(self, name):
        """
        Sets the variable which interacts with the pharmacodynamic model.
        In most models this will be the concentration of the drug in the
        central compartment.

        The name has to match a parameter of the model.

        This variable is mapped to the
        :meth:`erlotinib.PharmacodynamicModel.pk_input` variable when a
        :class:`PKPDModel` is instantiated.
        """
        if name not in self._parameter_names:
            raise ValueError(
                'The name does not match a model parameter.')

        self._pd_output = name
