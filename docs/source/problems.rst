
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


.. autoclass:: InverseProblem
    :members: