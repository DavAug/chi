
.. _erlotinib: https://github.com/DavAug/erlotinib
.. _pints: https://pints.readthedocs.io/en/stable/

********
Problems
********

.. currentmodule:: erlotinib

Inverse problems in erlotinib_ heavily rely on the inference package pints_.

An :class:`InverseProblem` allows you to easily convert a
:class:`PharmacokineticModel`, :class:`PharmacodynamicModel`, or :class:`PKPDModel`
into an inverse problem. Such an inverse problem can be used in conjuntion with
pints_ to infer the model parameters of the PK, PD or PKPD model.

To simplify the process of builiding a log-posterior, which models either individuals
separately or as a population, the :class:`ProblemModellingController`
has been implemented. Behind the scences it builds on an :class:`InverseProblem`, but
also simplifies the process of selecting error models, population models, and parameter
prior distributions.

Overview:

- :class:`InverseProblem`
- :class:`ProblemModellingController`


.. autoclass:: InverseProblem
    :members:

.. autoclass:: ProblemModellingController
    :members: