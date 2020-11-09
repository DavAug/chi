
.. _erlotinib: https://github.com/DavAug/erlotinib
.. _pints: https://pints.readthedocs.io/en/stable/

********
Problems
********

.. currentmodule:: erlotinib

Inverse problems in erlotinib_ heavily rely on the inference package pints_.

The :class:`InverseProblem` allows you to easily convert a
:class:`Pharmacodynamic` into an inverse problem. These inverse problems
can be used in conjuntion with pints_ log-likelihoods to create a versatile
range of error models.

For sampling or optimisition pints_ also allows to covert these log-likeilhoods
into log-posteriors by providing a number of different log-priors.


.. autoclass:: InverseProblem
    :members: