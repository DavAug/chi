#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import myokit
import myokit.formats.sbml as sbml
import numpy as np


class PharmacodynamicModel(object):
    """
    Converts a pharmacodynamic model specified by an SBML file into forward
    model that can be solved numerically.

    Parameters
    ----------
    sbml_file
        A path to the SBML model file that specifies the pharmacodynamic model.

    """

    def __init__(self, sbml_file):
        super(PharmacodynamicModel, self).__init__()

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

        # Set default parameter names
        self._parameter_names = self._state_names + self._const_names

        # Set default pharmacokinetic input variable
        # (Typically drug concentration)
        self._pk_input = None
        if 'myokit.drug_concentration' in self._parameter_names:
            self._pk_input = 'myokit.drug_concentration'

        # Set default outputs
        self._output_names = self._state_names
        self._n_outputs = self._n_states

        # Create simulator
        self._sim = myokit.Simulation(model)

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

    def pk_input(self):
        """
        Returns the pharmacokinetic input variable. In most models this will be
        the concentration of the drug.

        Defaults to ``None`` or ``myokit.drug_concentration`` if the latter is
        among the model parameters.
        """
        return self._pk_input

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
