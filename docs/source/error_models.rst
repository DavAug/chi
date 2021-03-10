.. _erlotinib: https://github.com/DavAug/erlotinib

************
Error Models
************

.. currentmodule:: erlotinib

Error models in erlotinib_ model the deviations of experimentally
observed pharmacokinetic and pharmacodynamic biomarkers and
the predictions of a :class:`MechanisticModel`.

Functional classes
------------------

- :class:`ConstantAndMultiplicativeGaussianErrorModel`
- :class:`GaussianErrorModel`
- :class:`MultiplicativeGaussianErrorModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: ConstantAndMultiplicativeGaussianErrorModel
    :members:
    :inherited-members:

.. autoclass:: GaussianErrorModel
    :members:
    :inherited-members:

.. autoclass:: MultiplicativeGaussianErrorModel
    :members:
    :inherited-members:

Base classes
------------

- :class:`ErrorModel`
- :class:`ReducedErrorModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: ErrorModel
    :members:

.. autoclass:: ReducedErrorModel
    :members:
