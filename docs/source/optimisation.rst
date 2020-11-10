
.. _erlotinib: https://github.com/DavAug/erlotinib
.. _pints: https://pints.readthedocs.io/en/stable/

************
Optimisation
************

.. currentmodule:: erlotinib

Optimisation in erlotinib_ heavily relies on the inference package pints_.

The :class:`OptimisationController` allows you to easily explore different
optimisation settings, e.g. fixing some parameters, or applying different
transformations to the search space.


.. autoclass:: OptimisationController
    :members: