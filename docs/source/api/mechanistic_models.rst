.. _chi: https://github.com/DavAug/chi
.. _SBML: http://sbml.org

******************
Mechanistic Models
******************

.. currentmodule:: chi

Mechanistic models in chi_ model the pharmacokinetics and
pharmacodynamics of patients based on models specified by SBML
files (System Biology Markup Language (SBML_)).

Some SBML files relevant to the modelling of chi are provided
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