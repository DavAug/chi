.. _SBML: https://sbml.org/

.. currentmodule:: chi

*********************
The mechanistic model
*********************

In chi, mechanistic model is an umbrella term for time
series models describing the dynamics of treatment responses. In general, these models do
not have to be of a *mechanistic* nature, but in most cases they will have at least
a mechanistic element to them.
Popular mechanistic models include for example PKPD models, PBPK models and
QSP models which are based on mechanistic descriptions of the treatment response.

Mechanistic models can be implemented in chi in two ways: 1. using SBML_ files
(recommended); and 2. using the :class:`chi.MechanisticModel` interface.
The :class:`chi.MechanisticModel` way has, perhaps, the
lower entry barrier, as it just involves implementing models
in any Python code you fancy, while the SBML way requires you to learn a small
amount of potentially unfamiliar SBML syntax. But in the long run, SBML will
make life easier for you as, in chi, it will automate differentiation,
sensitivities and dose administration. Below
we will show how we can use either of those ways to implement a 1-compartment
PK model.

**Use case: 1-compartment PK model:**

A 1-compartment PK model describes the absorption, distribution, metabolism and
elimination of a drug semi-mechanistically using a simple differential equation
of the form

.. math::
    \frac{\mathrm{d}a}{\mathrm{d}t} = -k_e\, a,
    \quad c = \frac{a}{v},
    \quad a(t=0) = a_0,

where :math:`a` is the drug amount in the compartment, :math:`t` is the time,
:math:`k_e` is the elimination rate of the drug, :math:`c` is the drug
concentration, :math:`v` is the volume of the compartment and :math:`a_0` is
the initial drug amount.

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
---------------------

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
This id is not really used by chi, but myokit uses it internally to name models,
and we can use this in this tutorial for debugging / making sure that the model
is implemented as expected. To this end, let us instantiate
a model from the SBML file using the code below

.. literalinclude:: code/2_mechanistic_model_1.py
    :lines: 18-27

In this example, we instantiate the model using :class:`chi.PKPDModel` from
the SBML file. The first two lines define the absolute path to the SBML file
by first getting the absolute path of the Python script and then pointing to
the SBML file in the same directory as the script. The last line prints the
name of the ``_model`` property of the :class:`chi.PKPDModel` instance, which
is the compiled ``myokit`` model. If everything works correctly, executing this
script should print the name of the model to the terminal, i.e. ``template``.

.. note::
    We do not recommend accessing the ``_model`` property in your scripts
    directly when you are modelling treatment responses (as indicated by the
    leading ``_``, the myokit model is supposed to be a private property of
    :class:`chi.PKPDModel`), but for debugging SBML files it can be useful to
    investigate ``_model`` directly.

Implementing the model
----------------------

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

.. literalinclude:: code/2_mechanistic_model_1.py
    :lines: 31-40

Here, we are using the ``code()`` method of the ``myokit`` model to print
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
script again, now yields a model print our that looks like this

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

We can see that in comparision to our previous model print out two things have happened:
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
assignment rule for the drug concentration, which related the drug concentration
to the drug amount, :math:`c = \frac{a}{v}`. Fortunately, at this point you have
already learned almost all the necessary SBML syntax to this. The only
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

Executing our Python script again we obtain an updated model print out

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

This completes the implementation of the one-compartment PK model, and we can start
to model the pharmacokinetics of drugs. For example, we can simulate the time course
of the drug concentration following a bolus dose of 10 drug amount units

.. literalinclude:: code/2_mechanistic_model_1.py
    :lines: 44-78

.. raw:: html
   :file: images/2_mechanistic_model_1.html

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
However, for illsutrative purposes we will show a version of the 1-compartment
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

.. note::
    Mathematical expressions in MathML can be nested to be able to implement
    more complex expressions. For example, the MathML equivalent of
    :math:`a + b / c` is

    .. code-block:: xml

        <math xmlns="http://www.w3.org/1998/Math/MathML">
          <apply>
            <plus/>
            <ci> a </ci>
            <apply>
              <divide/>
              <ci> b </ci>
              <ci> c </ci>
            </apply>
          </apply>
        </math>

    where the second ``<apply></apply>`` bracket implements the fraction
    :math:`b / c`.

.. autosummary::

    chi.MechanisticModel
    chi.SBMLModel
    chi.PKPDModel
    chi.ReducedMechanisticModel