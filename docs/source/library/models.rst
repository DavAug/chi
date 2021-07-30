*************
Model Library
*************

.. currentmodule:: chi

The model library contains a number of pharmacokinetic and
pharmacodynamic models in SBML file format which have been
used to model the PKPD of erlotinib.

Those SBML models can be passed to either a
:class:`PharmacokineticModel` or a
:class:`PharmacodynamicModel` for simulation or to learn the
model parameters from data.

.. currentmodule:: chi.library

Functional classes
------------------

- :class:`ModelLibrary`

Detailed API
^^^^^^^^^^^^

.. autoclass:: ModelLibrary
    :members: