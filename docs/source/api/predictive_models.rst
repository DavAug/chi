.. _chi: https://github.com/DavAug/chi

*****************
Predictive Models
*****************

.. currentmodule:: chi

Predictive models in chi_ can be used to predict observable
pharmacokinetic and pharmacodynamic biomarker values.

Each predictive model consists of a :class:`MechanisticModel` and one
:class:`ErrorModel` for each observable biomarker.

Functional classes
------------------

- :class:`PAMPredictiveModel`
- :class:`PopulationPredictiveModel`
- :class:`PosteriorPredictiveModel`
- :class:`PredictiveModel`
- :class:`PriorPredictiveModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: PAMPredictiveModel
    :members:
    :inherited-members:

.. autoclass:: PopulationPredictiveModel
    :members:
    :inherited-members:

.. autoclass:: PosteriorPredictiveModel
    :members:
    :inherited-members:

.. autoclass:: PredictiveModel
    :members:

.. autoclass:: PriorPredictiveModel
    :members:
    :inherited-members:

Base classes
------------

- :class:`AveragedPredictiveModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: AveragedPredictiveModel
    :members: