.. _SBML: https://sbml.org/
.. _Myokit: http://myokit.org/

.. currentmodule:: chi

*********************
The mechanistic model
*********************

In Chi, mechanistic model is an umbrella term for time
series models describing the dynamics of treatment responses. In general, these models do
not have to be of a *mechanistic* nature, but in most cases they will have at least
a mechanistic element to them.
Popular mechanistic models include for example PKPD models, PBPK models and
QSP models which are based on mechanistic descriptions of the treatment response.

Mechanistic models can be implemented in Chi in two ways: 1. using SBML_ files
(recommended); and 2. using the :class:`chi.MechanisticModel` interface.
The :class:`chi.MechanisticModel` interface has, perhaps, the
lower entry barrier, as it just involves implementing models
using any preferred Python code. This brings an enourmous amount of flexibility
and, for simple models, gets your model implemented fast. However, it has the
drawback that sensitivities of the model parameters and
mechanisms to administer dosing regimens have to implemented manually. It also
limits the ability to share models.

In comparison, using SBML files to implement your models requires a small
amount of potentially unfamiliar SBML syntax, and therefore may initially take
a little longer to get started. However, for complex models, SBML
simplifies the model implementation and reduces the risk for implementation
errors. In Chi, SBML will also automate differentiation, the evaluation of parameter
sensitivities, and the implementation of dose administrations. Another benefit
of SBML files is that they are programming language-agnostic, meaning that SBML
files are supported by many simulation softwares facilitating the sharing and
reimplementation of models without forcing the continued use of Chi. We therefore
recommend using SBML files to implement your models.

Below we will show how we can use either of those implementation strategies to
implement a 1-compartment PK model.

**Use case: 1-compartment PK model:**

A 1-compartment PK model is a semi-mechanistic description of the absorption,
distribution, metabolism and elimination of a drug using a simple differential
equation of the form

.. math::
    \frac{\mathrm{d}a}{\mathrm{d}t} = -k_e\, a,
    \quad c = \frac{a}{v},
    \quad a(t=0) = a_0,

where :math:`a` is the drug amount in the compartment, :math:`t` is the time,
:math:`k_e` is the elimination rate of the drug, :math:`c` is the drug
concentration, :math:`v` is the volume of the compartment and :math:`a_0` is
the initial drug amount.

Defining mechanistic models using the MechanisticModel interface
****************************************************************

The simplest way to implement mechanistic models in Chi is to use the
:class:`chi.MechanisticModel` interface. The :class:`MechanisticModel` is a
base class for mechanistic models with a small number of mandatory methods that
you need to implement in order to be able to use your implementation with all
of Chi's functionality. The base class takes the following form
(methods that don't have to be implemented are omitted):

.. code-block:: python

    class MechanisticModel(object):
        def simulate(self, parameters, times):
            """
            Returns the numerical solution of the model outputs for the
            specified parameters and times.

            The model outputs are returned as a 2 dimensional NumPy array of shape
            ``(n_outputs, n_times)``.

            :param parameters: An array-like object with values for the model
                parameters.
            :type parameters: list, numpy.ndarray
            :param times: An array-like object with time points at which the output
                values are returned.
            :type times: list, numpy.ndarray

            :rtype: np.ndarray of shape (n_outputs, n_times)
            """
            raise NotImplementedError

        def has_sensitivities(self):
            """
            Returns a boolean indicating whether sensitivities have been enabled.
            """
            raise NotImplementedError

        def n_outputs(self):
            """
            Returns the number of output dimensions.
            """
            raise NotImplementedError

        def n_parameters(self):
            """
            Returns the number of parameters in the model.
            """
            raise NotImplementedError

        def outputs(self):
            """
            Returns the output names of the model.
            """
            raise NotImplementedError

        def parameters(self):
            """
            Returns the parameter names of the model.
            """
            raise NotImplementedError

For full details of the base class, we refer to the API reference:
:class:`chi.MechanisticModel`.

The main method of the mechanistic model is the :meth:`MechanisticModel.simulate`
method, implementing the simulation of the model. The other methods are
used for book-keeping and for enabling the computation of sensitivities. In
this example, we will not implement the sensitivities, so we can start
the model implementation by setting the return of the
:meth:`MechanisticModel.has_sensitivities` method to ``False``

.. code-block:: python

    class OneCompPKModel(chi.MechanisticModel):
        def __init__(self):
            super().__init__()

        def has_sensitivities(self):
            # Model does not implement sensitivities, so output of this method
            # is always False
            return False

In the above code block, we define a new class called ``OneCompPKModel`` which will
implement the 1-compartment PK model. In the first line, the class
inherits from the :class:`chi.MechanisticModel` base class. In the next two
lines, we use ``super().__init__()`` to initialise properties of the the base
class. If you are not familiar with inheritance in Python, you can use these
three lines as a default setup of your model implementation. The remainder of
the code block implments the :meth:`MechanisticModel.has_sensitivities` method
by setting its return
to ``False``, indicating to other classes and functions in Chi
that the model does not implement sensitivities. This is a necessary method of
:class:`chi.MechanisticModel` in Chi, because some optimisation and inference
algorithms require sensitivites, and setting the output to ``False`` will
indicate to those algorithms that they cannot be used with this model.

The other methods, besides the :meth:`MechanisticModel.simulate`, are used for
book-keeping and define the number of outputs of the model, the number of
parameters of the model and their names. In this case the 1-compartment PK
model has only 1 output, the concentration of the drug in the central
compartment. We can therefore continue the implementation of the model by setting
the return of :meth:`MechanisticModel.n_outputs` to ``1`` and the return of
:meth:`MechanisticModel.outputs` to ``['Drug concentration']``

.. code-block:: python

    class OneCompPKModel(chi.MechanisticModel):
        def __init__(self):
            super().__init__()

        def has_sensitivities(self):
            # Model does not implement sensitivities, so output of this method
            # is always False
            return False

        def n_outputs(self):
            return 1

        def outputs(self):
            return ['Drug concentration']

Those two methods communicate to other classes and functions in Chi that the
model has only one output, the drug concentration. Note that the return of
:meth:`MechanisticModel.outputs` is a list of strings in order to facilitate
returning the names of mutliple outputs, in which case the return would take
the form ``['Output name 1', 'Output name 2', ..., 'Output name n']``.

The remaining book-keeping methods return the number of parameters and
the names of the parameters, :meth:`MechanisticModel.n_parameters` and
:meth:`MechanisticModel.parameters`. A 1-compartment PK model has three model
parameters: 1. the initial drug amount, :math:`a_0`; 2. the elimination rate,
:math:`k_e`: and 3. the volume of distribution, :math:`v`. We can, thus,
implement the methods by setting the return of
:meth:`MechanisticModel.n_parameters` to ``3`` and the return of
:meth:`MechanisticModel.parameters` to
``['Initial amount', 'Elimination rate', 'Volume of distribution']``.

.. code-block:: python

    class OneCompPKModel(chi.MechanisticModel):
        def __init__(self):
            super().__init__()

        def has_sensitivities(self):
            # Model does not implement sensitivities, so output of this method
            # is always False
            return False

        def n_outputs(self):
            return 1

        def outputs(self):
            return ['Drug concentration']

        def n_parameters(self):
            return 3

        def parameters(self):
            return ['Initial amount', 'Elimination rate', 'Volume of distribution']

This leaves only :meth:`MechanisticModel.simulate` to implement. The
:meth:`MechanisticModel.simulate` is the heart of the mechanistic model, taking
the parameter values and the evaluation time points of the simulation as inputs
and returning the simulated model outputs. How you find the simulated
model outputs is in the :meth:`MechanisticModel` interface up to you!

In thise case, the 1-compartment PK model is simple
enough so that we can find and implement the analytical solution to the
differential equation directly. Solving the differential equation shows that
the 1-compartment PK model yields an exponential decay of the drug
concentration in the central compartment

.. math::
    c(t) = \frac{a_0}{v}\, \mathrm{e}^{-k_e t}.

We can implement this exponential decay using numpy as follows

.. literalinclude:: code/2_mechanistic_model_1.py
    :lines: 2-36

The inputs to the :meth:`MechanisticModel.simulate` method are the values of
the model parameters and the time points for the model evaluation. As defined
by the base class, the parameter values, ``parameters``, have to take the form
of a list or an np.ndarray of length ``n_parameters``. These parameter values
are deconstructed in the first line of the method and assigned to the model
parameters. The time point input, ``times``, also has to take the form of a list or
an np.ndarray, but can be of any length, ``n_times``. In the second line of the method,
we use ``np.array(times)`` to convert the ``times`` input to an np.ndarray to enable the
computation of the drug concentration at all time points in parallel using numpy's vectorised operations.
The next line implements the model solution and calculates the drug
concentration for all time points in ``times``. Finally, the last two lines of
the method make sure that the output of the method adheres to the format defined
by the interface, :class:`chi.MechanisticModel`, reshaping the simulation result
to an np.ndarray of shape ``(n_outputs, n_times)`` by first instantiating an
empty array of shape ``(self.n_outputs(), len(times))``, i.e. ``(1, len(times))``,
and then populating the array with the simulated concentration values.

This completes the implementation of the 1-compartment PK model and we can start
to model the pharmacokinetics of drugs. For example, we can simulate the time course
of the drug concentration following a bolus dose of 10 drug amount units

.. literalinclude:: code/2_mechanistic_model_1.py
    :lines: 41-68

.. raw:: html
   :file: images/2_mechanistic_model_1.html

As expected, the drug concentration decays exponentially with the rate defined
by the elimination rate, starting from an initial value of 5, defined by the
initial drug amount and the volume of distribution (:math:`10 / 2`).

Strengths & limitations
^^^^^^^^^^^^^^^^^^^^^^^

The above implementation of the 1-compartment PK model can now be used together
with other classes and functions in Chi to simulate drug concentrations and to
infer parameter values from measurements. However, to get access to all of
Chi's features additional methods need to be implemented, including methods
mirroring :class:`PKPDModel.set_administration` and
:class:`PKPDModel.set_dosing_regimen` in order to facilitate dose
administrations that go beyond bolus doses at the start of the simulation.
Below, we provide an overview of the strengths and limitations of using the
:class:`chi.MechanisticModel` interface

**Strengths**

- Simple models can be implemented quickly.
- The interface is extremely flexible and allows for the use of any preferred Python code.
- Only knowledge of Python is required to implement the model.

**Limitations**

- No support for the implementation of complex models.
- No automated implementation of output names and parameter names.
- No automated implementation of changing model outputs.
- No automated support of renaming model outputs and model parameters.
- No automated implementation of dose administrations.
- No automated implementation of sensitivities.
- Sharing of models is limited to the use of Chi.


Defining mechanistic models using SBML files
********************************************

The systems biology markup language (SBML) can be used to define
mechanistic models in a programming language-agnostic way, facilitating the
sharing, the implementation, and the reproduction of computational models. In
chi, SBML also allow us to automatically implement sensitivities and dose
administrations of treatment response models. But before we go into more
detail of why SBML is a great choice to implement your models, let us implement
our first SBML model to see how it works in practice.

Setting up a template
^^^^^^^^^^^^^^^^^^^^^

The first thing to note about SBML is that mechanistic models are not defined
in your Python scripts directly, but instead are defined in SBML files external
to your code (this makes the programming language-agnostic sharing of models possible).
These SBML files have a minimal biolerplate of five lines

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="template">

      <!-- Your model definition goes here -->

      </model>
    </sbml>

Create a new file called ``template.xml`` and copy-paste the above lines in there.
For all future models you will implement, you can use these lines as a starting point.
The first line specifies the XML version and the encoding, while the second and last line
in the file specify the XML namespace. This namespace is what makes an XML file
an SBML file. The remaining two lines begin the model definition.

You can see
that the model tag has an ``id`` property with the value ``"template"``.
This id is not really used by chi, but Myokit_ uses it internally to name models,
and we can use this in this tutorial for debugging / making sure that the model
is implemented as expected. To this end, let us instantiate
a model from the SBML file using the code below

.. literalinclude:: code/2_mechanistic_model_2.py
    :lines: 18-27

In this example, we instantiate the model using :class:`chi.PKPDModel` from
the SBML file. The first two lines define the absolute path to the SBML file
by first getting the absolute path of the Python script and then pointing to
the SBML file in the same directory as the script. The last line prints the
name of the ``_model`` property of the :class:`chi.PKPDModel` instance, which
is the compiled Myokit_ model. If everything works correctly, executing this
script should print the name of the model to the terminal, i.e. ``template``.

.. note::
    We do not recommend accessing the ``_model`` property in your scripts
    directly when you are modelling treatment responses (as indicated by the
    leading ``_``, the Myokit_ model is supposed to be a private property of
    :class:`chi.PKPDModel`), but for debugging SBML files it can be useful to
    investigate ``_model`` directly.

Implementing the model
^^^^^^^^^^^^^^^^^^^^^^

With this template in place, we can now implement our model. Let us
start by first sketching out the two blocks of definitions that we need for
this model: 1. parameter/variable definitions; and 2. rule definitions (including
assignment rules like :math:`c = \frac{a}{v}` and rate rules like
:math:`\mathrm{d}a/\mathrm{d}t = -k_e \,a`)

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="template">

        <listOfParameters>
          <!-- This is were the parameter/variable definitions go -->
        </listOfParameters>

        <listOfRules>
          <!-- This is were the rule definitions go -->
        </listOfRules>

      </model>
    </sbml>

All parameters and variables of the model will need to be defined inside the
``listOfParameters`` tags, while all assignment rules and rate rules will have
to be defined inside the ``listOfRules`` tags.

Parameters are defined using parameter tags of the form ``<parameter id=your_id value=42/>``.
The ``id`` property defines the name of the parameter and the ``value`` property defines
the value of the parameter to which it is initialised. Parameter tags can have more, optional
properties (see SBML_), but the ``id`` and the ``value`` properties are required.
Using the parameter tags, we can define the parameters and variables of the one-compartment
PK model :math:`(a, c, k_e, v)`.

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="one_compartment_pk_model">

        <listOfParameters>
          <parameter id="drug_amount" value="1"/>
          <parameter id="drug_concentration" value="1"/>
          <parameter id="elimination_rate" value="1"/>
          <parameter id="volume" value="1"/>
        </listOfParameters>

        <listOfRules>
          <!-- This is were the rule definitions go -->
        </listOfRules>

      </model>
    </sbml>

Let us check that our SBML file is implemented correctly by instantiating the
model and printing the compiled model back to us. To this end, let us
copy-paste the above model definition into an XML file with the name
``one_compartment_pk_model.xml`` and execute the below script

.. literalinclude:: code/2_mechanistic_model_2.py
    :lines: 31-40

Here, we are using the ``code()`` method of the Myokit_ model to print
the model specification back to us. If the implementation is correct, you
should see a print out like this

.. code-block:: python

    [[model]]
    name: one_compartment_pk_model

    [global]
    drug_amount = 1
    drug_concentration = 1
    elimination_rate = 1
    time = 0 bind time
        in [1]
    volume = 1

The model definition shows that the model contains 5 parameters, the 4 parameters
that we have defined, initialised to the value as specified in the SBML file,
and a fifth variable called ``time``. The time variable is automatically added
to the model and, as the name suggest, is the variable that will keep track of
time.

At this point, the model is pretty useless because we haven't implemented the
rate rule that defines how the drug amount changes over time, and we also haven't
defined how the drug concentration is related to the drug amount. So let us
implement the rate rule, :math:`\mathrm{d}a/\mathrm{d}t = -k_e\, a` to bring this
PK model to life

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="one_compartment_pk_model">

        <listOfParameters>
          <parameter id="drug_amount" value="1"/>
          <parameter id="drug_concentration" value="1"/>
          <parameter id="elimination_rate" value="1"/>
          <parameter id="volume" value="1"/>
        </listOfParameters>

        <listOfRules>
          <rateRule variable="drug_amount">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <times/>
                <cn> -1 </cn>
                <ci> elimination_rate </ci>
                <ci> drug_amount </ci>
              </apply>
            </math>
          </rateRule>
        </listOfRules>

      </model>
    </sbml>

Before explaining the syntax in more detail, let us first see how the addition of
the rate rule has changed our model implementation. Executing the above Python
script again, now yields a model print out that looks like this

.. code-block:: python

    [[model]]
    name: one_compartment_pk_model
    # Initial values
    global.drug_amount = 1

    [global]
    dot(drug_amount) = -1 * elimination_rate * drug_amount
    drug_concentration = 1
    elimination_rate = 1
    time = 0 bind time
        in [1]
    volume = 1

We can see that in comparison to our previous model print out two things have happened:
1. a new section with title ``# Initial values`` has appeared; and 2. the expression
of the ``drug_amount`` variable has changed. This is because rate rules promote (constant)
parameters to variables whose dynamics are now defined by ordinary differential equations.
In the model print out the ``dot(drug_amount)`` denotes the time derivative of the drug amount
variable, i.e. :math:`\mathrm{d}a/\mathrm{d}t`. The initial value of the drug amount at :math:`t=0`
is defined by the ``value`` property of the corresponding parameter, indicated in the
model print out by ``global.drug_amount = 1``.

The SBML syntax for defining rate rules like this is to add ``rateRule`` tags to
the ``listOfRules`` that point to the relevant parameter in the ``listOfParameters``

.. code-block:: xml

    <listOfParameters>
      <parameter id="your_parameter" value="1"/>
    </listOfParameters>

    <listOfRules>
      <rateRule variable="your_parameter">
        <math xmlns="http://www.w3.org/1998/Math/MathML">

          <!-- This is were the definition of the right hand side goes -->

        </math>
      </rateRule>
    </listOfRules>

      </model>
    </sbml>

This promotes the parameter to a variable that can change over time. The right
hand side of the differential equation is defined inside the ``math`` tags which
point to the MathML XML namespace which is used in SBML to define mathematical
expression.

In MathML, mathematical expressions are encapsulated by the ``<apply></apply>``
tags followed by a tag that indicates the mathetical operation and the relevant
parameters / variables. The most common operations are

- addition: ``</plus>``
- subtraction: ``</minus>``
- multiplication: ``</times>``
- division: ``</divide>``
- exponentiation: ``</power>``

Commutative operations, such as addition and multiplication, work with any
number of parameters / variables while other operators can only be applied
pairwise. For more information about the MathML syntax, we refer to
http://www.w3.org/1998/Math/MathML.

Revisiting the ``rateRule`` in our model definition

.. code-block:: XML

    <rateRule variable="drug_amount">
      <math xmlns="http://www.w3.org/1998/Math/MathML">
        <apply>
          <times/>
          <cn> -1 </cn>
          <ci> elimination_rate </ci>
          <ci> drug_amount </ci>
        </apply>
      </math>
    </rateRule>

we can now see that the right hand side of the differential equation for the
drug amount is defined by the product of the constant ``-1``,
the ``elimination_rate`` parameter, and the ``drug_amount`` variable. Note that
constants are labelled by ``<cn></cn>`` tags, while parameters / variables
are labelled by ``<ci></ci>``. This labelling is used in SBML to indicate whether
parameters definitions need to be looked up in the list of parameters.

We can now move on and complete the implementation of our model by adding the
assignment rule for the drug concentration, which relates the drug concentration
to the drug amount, :math:`c = \frac{a}{v}`. Fortunately, at this point you have
already learned almost all the necessary SBML syntax to do this. The only
difference is that assignment rules are indicated with ``assignmentRule`` tags
instead of the rate rule tag

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="one_compartment_pk_model">

        <listOfParameters>
          <parameter id="drug_amount" value="1"/>
          <parameter id="drug_concentration" value="1"/>
          <parameter id="elimination_rate" value="1"/>
          <parameter id="volume" value="1"/>
        </listOfParameters>

        <listOfRules>
          <rateRule variable="drug_amount">
            <math xmlns="http://www.w3.org/1998/Math/MathML">
              <apply>
                <times/>
                <cn> -1 </cn>
                <ci> elimination_rate </ci>
                <ci> drug_amount </ci>
              </apply>
            </math>
          </rateRule>
        </listOfRules>

        <assignmentRule variable="drug_concentration">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
              <divide/>
              <ci> drug_amount </ci>
              <ci> volume </ci>
            </apply>
          </math>
        </assignmentRule>

      </model>
    </sbml>

Executing our Python script again, we obtain an updated model print out

.. code-block:: python

    [[model]]
    name: one_compartment_pk_model
    # Initial values
    global.drug_amount = 1

    [global]
    dot(drug_amount) = -1 * elimination_rate * drug_amount
    drug_concentration = drug_amount / volume
    elimination_rate = 1
    time = 0 bind time
        in [1]
    volume = 1

We can see that in contrast to the rate rule, the assignment rule did not promote
the expression for the drug concentration to a differential equation. Instead
it simply assigned the drug concentration to be equal to the drug amount divided
by the volume at all times.

This completes the implementation of the one-compartment PK model and we can start
to model the pharmacokinetics of drugs. For example, we can simulate the time course
of the drug concentration following a bolus dose of 10 drug amount units

.. literalinclude:: code/2_mechanistic_model_2.py
    :lines: 44-78

.. raw:: html
   :file: images/2_mechanistic_model_2.html

As expected, the drug concentration decays exponentially with the rate defined
by the elimination rate, starting from an initial value of 5, defined by the
initial drug amount and the volume of distribution (:math:`10 / 2`).

.. note::
    The order of the parameter values when simulating the model follows a
    simple pattern: the initial values of state variables come first, followed
    by the values of constant parameters. If multiple initial values or parameters
    exist, they are order alphabetically. However, to simplify keeping track of
    the order of parameters, the model implements a ``parameters()`` method,
    :meth:`chi.PKPDModel.parameters`, which returns the parameter names of the
    model.

    Similarly, you can get the names of the simulation outputs of the model
    with :meth:`chi.PKPDModel.outputs`. The outputs of the model can be changed
    using :meth:`chi.PKPDModel.set_outputs`

SBML in chi has many more functionalities than those outlined in this tutorial,
including the definition of units for parameters and variables, the definition of
compartments, and the definition of reaction equations. While being clear about
units is of utmost importance across all applications of treatment response
modelling, the ability to define compartments and rate equations in order to
incrementally implement a model in modular blocks is particularly useful when
implementing large mechanistic models, such as PBPK models and QSP models.

We
will not go into further detail about these elements of SBML in this tutorial.
However, for illustrative purposes we will show a version of the 1-compartment
PK model implementation which includes compartments, rate equations and units

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
      <model id="one_compartment_pk_model" timeUnits="day">

        <listOfUnitDefinitions>
          <unitDefinition id="day">
            <listOfUnits>
              <unit kind="second" exponent="1" scale="0" multiplier="86400"/>
            </listOfUnits>
          </unitDefinition>
          <unitDefinition id="per_day">
            <listOfUnits>
              <unit kind="second" exponent="-1" scale="0" multiplier="86400"/>
            </listOfUnits>
          </unitDefinition>
          <unitDefinition id="mg">
            <listOfUnits>
              <unit kind="gram" exponent="1" scale="-3" multiplier="1"/>
            </listOfUnits>
          </unitDefinition>
        </listOfUnitDefinitions>

        <listOfCompartments>
          <compartment id="central" name="central" size="1" units="liter"/>
        </listOfCompartments>

        <listOfSpecies>
          <species id="drug" name="drug" compartment="central" initialAmount="0" hasSubstanceUnits="false" substanceUnits="mg"/>
        </listOfSpecies>

        <listOfParameters>
          <parameter id="elimination_rate" value="1" constant="true" units="per_day"/>
        </listOfParameters>

        <listOfReactions>
          <reaction id="central_reaction" name="central_reaction" reversible="false" fast="false">
            <listOfReactants>
              <speciesReference species="drug"/>
            </listOfReactants>
            <kineticLaw>
              <math xmlns="http://www.w3.org/1998/Math/MathML">
                <apply>
                  <times/>
                  <ci> central </ci>
                  <ci> elimination_rate </ci>
                  <ci> drug </ci>
                </apply>
              </math>
            </kineticLaw>
          </reaction>
        </listOfReactions>

      </model>
    </sbml>

For a complete description of SBML, we refer to the SBML documentation: https://sbml.org/.

Setting dosing regimens
^^^^^^^^^^^^^^^^^^^^^^^

In Chi, models implemented using SBML files have a simple two-step interface to
simulate treatment responses for different dosing strategies. The first
step is to specify the route of administration using
:meth:`PKPDModel.set_administration`. Once this step is completed, the dosing
regimen can be scheduled using :meth:`PKPDModel.set_dosing_regimen`.

The :meth:`PKPDModel.set_administration` method defines the drug amount
variable of the model that gets elevated when dosages are administered. In our
1-compartment PK model from above, there is of course only one drug amount
variable in the model and this flexibility seems unnecessary, however for more
complicated models specifying drug amount
variables in different compartments can be used to emulate different
routes of administration. To facilitate this flexibility of the route of
administration the :meth:`PKPDModel.set_administration` method has one mandatory input parameter
and two optional input parameters. The mandatory input parameter, ``compartment``,
specifies the
compartment of the drug amount variable, the other two input parameters specify
the name of the drug amount
variable, ``amount_var``, (default is ``amount_var='drug_amount'``) and whether the administration is
direct, ``direct``, (default is ``direct=True``). If ``direct`` is set to ``False`` the drug is administered
to the drug amount variable through a dose copmartment.

Let us first illustrate the direct aministration. In our above example,
``one_compartment_pk_model.xml``, we used rules to define our model and did not
specify any compartments. In this case, Myokit_ assigns all variables to a global
compartment called ``global``. We can therefore administer doses directly to
the drug amount variable by setting the compartment input to ``'global'``


.. literalinclude:: code/2_mechanistic_model_2.py
    :lines: 86

To see what happens to the model when we set an administration, we can use
``print(model._model.code())`` to print the model.

.. code-block:: python

    [[model]]
    name: one_compartment_pk_model
    # Initial values
    global.drug_amount = 1

    [global]
    dose_rate = 0 bind pace
    dot(drug_amount) = -1 * elimination_rate * drug_amount + dose_rate
    drug_concentration = drug_amount / volume
    elimination_rate = 1
    time = 0 bind time
        in [1]
    volume = 1

We can see that two things have happened: 1. a ``dose_rate`` variable has been added
to the model; and 2. the dose rate variable has been added to the right hand
side of the ``drug_amount`` rate equation. The ``dose_rate`` will be set by the
dosing regimen and implements the administration of dosages.

We can now simulate the drug concentration for different dosing regimens. For
example, assuming the dose amount is in units of mg and the time is in units of
days, we may want to administer a dose of 2 mg every day, or a dose of 4 mg
every 2 days, or administer a total dose of 2 mg every day with an infusion
that lasts for 12 hours.

.. literalinclude:: code/2_mechanistic_model_2.py
    :lines: 91-131

.. raw:: html
   :file: images/2_mechanistic_model_3.html

For full details of which dosing regimens are possible, we refer to API
documentation, :meth:`PKPD.set_dosing_regimen`.

For completeness, let us administer the same dosing regimens indirectly to the
drug amount variable.

.. literalinclude:: code/2_mechanistic_model_2.py
    :lines: 138-187

.. raw:: html
   :file: images/2_mechanistic_model_4.html

We can see that relative to the direct administration two things have happened:
1. Somehow the number of parameters have changed from 3 parameters to 5 parameters (see line 10 in the code block);
and 2. the peaks of the drug concentrations appear to be rounded off in the figure.

To understand this, let us again investigate the Myokit_ model using
``print(model._model.code())``, and see how the route of administration has
changed the model

.. code-block:: python

    [[model]]
    name: one_compartment_pk_model
    # Initial values
    global.drug_amount = 1
    dose.drug_amount   = 0

    [dose]
    absorption_rate = 1
        in [1]
    dot(drug_amount) = -absorption_rate * drug_amount + global.dose_rate

    [global]
    dose_rate = 0 bind pace
    dot(drug_amount) = -1 * elimination_rate * drug_amount + dose.absorption_rate * dose.drug_amount
    drug_concentration = drug_amount / volume
    elimination_rate = 1
    time = 0 bind time
        in [1]
    volume = 1

We can see that the indirect administration makes changes to the model that are
conceptually similar to the direct administration, but instead of adding the
``dose_rate`` variable directly to the right hand side of the ``drug_amount``
rate equation in the ``global`` compartment, it is added to the rate equation
of a ``drug_amount`` variable in a new ``dose`` compartment. From this ``dose``
compartment the drug transitions to the ``global`` compartment at a constant rate --
the ``absorption_rate``.

These changes of the model explain the shaving off of the drug concentration
peaks: the drug is first adminstered to the dose compartment and then transitions
at a reduced rate to the observed compartment. It also explains why the number of
model parameters increases from 3 to 5: indirect administrations add the initial
value of the ``dose`` compartment ``drug_amount`` variable and the ``absorption_rate``
parameter to the model.

.. note::
    Whenever you are in doubt about the order of the parameters, use
    :meth:`PKPDModel.parameters` to look up their order.

Calculating sensitivities
^^^^^^^^^^^^^^^^^^^^^^^^^

In Chi, models implemented using SBML files also have a simple interface to
simulate the sensitivities of model parameters. The simulation of sensitivities
can be enabled using the :meth:`PKPDModel.enable_sensitivities` method which
modifies the output of :meth:`PKPDModel.simulate`. If sensitivities are enabled,
:meth:`PKPDModel.simulate` returns a tuple of numpy arrays instead of just a
single numpy array. The first numpy array is the simulated model output, as
before. The second numpy array contains the sensitivities of the
model outputs for each model parameter at all simulated time points and
is of shape ``(n_times, n_outputs, n_parameters)``.

To see the simulation of sensitivities in action, we use again our 1-compartment
PK model example from above as a use case, ``'one_compartment_pk_model.xml'``, and simulate
the model output and its sensitivities for daily dose administrations of 2 mg.
For simplicity, we choose to visualise the sensitivities only for the initial
drug amount in the ``global`` compartment and the elimination rate.

.. literalinclude:: code/2_mechanistic_model_2.py
    :lines: 203-251

.. raw:: html
   :file: images/2_mechanistic_model_5.html

In the code block, you can see that we select the sensitivities by
1. indexing the relevant time
points in the first axis (in this case, we select all simulated time points with
``:``), 2. indexing the relevant output in the second axis, and 3. indexing
the relevant parameter in the third axis. The order of the parameters is
consistent across all methods of the :class:`PKPDModel` and can be looked up
using :meth:`PKPDModel.parameters`.

In the figure, the sensitivities show that the influence of the initial drug
amount on the drug concentration decays exponentially over time, as indicated
by the decreasing magnitude of its sensitivity. The more intitial drug amount
is available, the larger the drug concentration is for early times of the
simulation. However,
drug concentration levels at later times will not be affected by elevated initial
drug amounts. This aligns with our
understanding of the model, as we know that the model clears drug amounts at an
exponential rate from the system. The initial drug amount can therefore only
influence the drug concentration at early times.

We can similarly understand
the sensitivity with respect to the elimination rate. The figure shows that
the elimination rate has no influence on the drug concentration at :math:`t=0`.
Only for :math:`t>0`, the elimination rate begins to affect the drug
concentration level. This is because for early times, the drug concentration
value is dominiated by the initial drug amount. As the influence of the
initial drug amount on the drug concentration diminishes, the influence of the
elimination rate increases. Note that the sensitivity with respect to the
elimination rate is negative, in contrast to the positive sensitivity with
respect to the initial drug amount. As a result, larger elimination rates
negatively impact drug conentration levels.

This concludes the tutorial on mechanistic models in Chi. For the full
documentation of mechanistic models, we refer to the API reference
below.

Reference to MechanisticModel API
*********************************

.. autosummary::

    chi.MechanisticModel
    chi.SBMLModel
    chi.PKPDModel
    chi.ReducedMechanisticModel