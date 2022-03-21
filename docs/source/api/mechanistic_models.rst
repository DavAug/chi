.. _SBML: http://sbml.org

******************
Mechanistic Models
******************

.. currentmodule:: chi

Mechanistic models in chi refer to any deterministic model that describes the
evolution of quantities of interest in time. In systems biology
such models are often inspired by biological mechanisms, which is why we go
with the name :class:`chi.MechanisticModel`. :class:`chi.MechanisticModel` by
no means have to be mechanism-based though, but may be any function of time
that you may deem interesting.

Chi provides two ways to specify mechanistic models: 1. you can use the
:class:`chi.MechanisticModel` base class an implement its methods yourself;
2. you can specify the mechanistic model using the System Biology Markup
Language (SBML_) and instantiate the model using :class:`chi.SBMLModel`.
For detailed examples how either of those can be done, we refer to the Getting
started.

Classes
-------

- :class:`MechanisticModel`
- :class:`SBMLModel`
- :class:`PKPDModel`
- :class:`ReducedMechanisticModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: MechanisticModel
    :members:

.. autoclass:: SBMLModel
    :members:
    :inherited-members:

.. autoclass:: PKPDModel
    :members:
    :inherited-members:

.. autoclass:: ReducedMechanisticModel
    :members:
    :inherited-members: