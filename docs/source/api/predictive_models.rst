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

- :class:`PosteriorPredictiveModel`
- :class:`PredictiveModel`
- :class:`PredictivePopulationModel`
- :class:`PriorPredictiveModel`
- :class:`StackedPredictiveModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: PosteriorPredictiveModel
    :members:
    :inherited-members:

.. autoclass:: PredictiveModel
    :members:

.. autoclass:: PredictivePopulationModel
    :members:
    :inherited-members:

.. autoclass:: PriorPredictiveModel
    :members:
    :inherited-members:

.. autoclass:: StackedPredictiveModel
    :members:
    :inherited-members:

Base classes
------------

- :class:`GenerativeModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: GenerativeModel
    :members: