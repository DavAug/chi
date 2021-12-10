
.. _chi: https://chi.readthedocs.io/en/latest/index.html
.. _pints: https://pints.readthedocs.io/en/stable/

*********
Inference
*********

.. currentmodule:: chi

Inference in chi_ heavily relies on the inference package pints_.

The :class:`OptimisationController` and :class:`SamplingController` allow you
to easily explore different optimisation or sampling settings, e.g. using different
methods, fixing some parameters, or applying different transformations to the
search space.

Functional classes
------------------

- :class:`OptimisationController`
- :class:`SamplingController`

Detailed API
^^^^^^^^^^^^

.. autoclass:: OptimisationController
    :members:
    :inherited-members:

.. autoclass:: SamplingController
    :members:
    :inherited-members:


Utility functions
-----------------

- :func:`compute_pointwise_loglikelihood`

Detailed API
^^^^^^^^^^^^

.. autofunction:: compute_pointwise_loglikelihood


Base classes
------------

- :class:`InferenceController`

Detailed API
^^^^^^^^^^^^

.. autoclass:: InferenceController
    :members: