
.. _erlotinib: https://erlotinib.readthedocs.io/en/latest/index.html
.. _pints: https://pints.readthedocs.io/en/stable/

*********
Inference
*********

.. currentmodule:: erlotinib

Inference in erlotinib_ heavily relies on the inference package pints_.

The :class:`OptimisationController` and :class:`SamplingController` allow you
to easily explore different optimisation or sampling settings, e.g. using different
methods, fixing some parameters, or applying different transformations to the
search space.

Base classes
------------

- :class:`InferenceController`

Functional classes
------------------

- :class:`OptimisationController`
- :class:`SamplingController`

Detailed API
------------

.. autoclass:: InferenceController
    :members:

.. autoclass:: OptimisationController
    :members:
    :inherited-members:

.. autoclass:: SamplingController
    :members:
    :inherited-members: