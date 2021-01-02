.. _erlotinib: https://github.com/DavAug/erlotinib

*****************
Population Models
*****************

.. currentmodule:: erlotinib

Population models in erlotinib_ can be used to model the variation
of mechanistic model parameters or error model parameters across individuals.

Functional classes
------------------

- :class:`HeterogeneousModel`
- :class:`LogNormalModel`
- :class:`PooledModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: HeterogeneousModel
    :members:
    :inherited-members:
    :special-members:
    :exclude-members: __init__, __weakref__

.. autoclass:: LogNormalModel
    :members:
    :inherited-members:
    :special-members:
    :exclude-members: __init__, __weakref__

.. autoclass:: PooledModel
    :members:
    :inherited-members:
    :special-members:
    :exclude-members: __init__, __weakref__

Base classes
------------
- :class:`PopulationModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: PopulationModel
    :members:
    :special-members:
    :exclude-members: __init__, __weakref__