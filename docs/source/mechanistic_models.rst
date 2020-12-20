.. _erlotinib: https://github.com/DavAug/erlotinib

******************
Mechanistic Models
******************

.. currentmodule:: erlotinib

Mechanistic models in erlotinib_ convert SBML files into model
classes which can be used to simulate the pharmacokinetics or
pharmacodynamics of erlotinib.

Any :class:`PharmacokineticModel` may be combined with any
:class:`PharmacodynamicModel` using the :class:`PKPDModel`
interface.

Base classes
------------

- :class:`MechanisticModel`

Functional classes
------------------

- :class:`PharmacodynamicModel`
- :class:`PharmacokineticModel`

Detailed API
------------

.. autoclass:: MechanisticModel
    :members:

.. autoclass:: PharmacodynamicModel
    :members:
    :inherited-members:

.. autoclass:: PharmacokineticModel
    :members:
    :inherited-members: