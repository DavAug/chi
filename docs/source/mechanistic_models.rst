.. _erlotinib: https://github.com/DavAug/erlotinib
.. _SBML: http://sbml.org

******************
Mechanistic Models
******************

.. currentmodule:: erlotinib

Mechanistic models in erlotinib_ model the pharmacokinetics and
pharmacodynamics of patients based on models specified by SBML
files (System Biology Markup Language (SBML_)).

Some SBML files relevant to the modelling of erlotinib are provided
in the :class:`ModelLibrary`.

Functional classes
------------------

- :class:`PharmacodynamicModel`
- :class:`PharmacokineticModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: PharmacodynamicModel
    :members:
    :inherited-members:

.. autoclass:: PharmacokineticModel
    :members:
    :inherited-members:

Base classes
------------

- :class:`MechanisticModel`
- :class:`ReducedMechanisticModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: MechanisticModel
    :members:

.. autoclass:: ReducedMechanisticModel
    :members: