.. _chi: https://github.com/DavAug/chi

**********************
Measurement Data Plots
**********************

.. currentmodule:: chi

Measurement data plots in chi_ are primarily intended to
illustrate measurements of pharmacokinetic or pharmacodynamic
biomarkers from different individuals or patients over time.

However, simple :class:`MechanisticModel` simulations may also
be added to the figure. This is ideally suited to explore the
dynamic properties of a given model for a number of parameter values
and compare them to preclinical or clinical data.

.. currentmodule:: chi.plots

Functional classes
------------------

- :class:`PDTimeSeriesPlot`
- :class:`PKTimeSeriesPlot`

Detailed API
^^^^^^^^^^^^

.. autoclass:: PDTimeSeriesPlot
    :members:
    :inherited-members:

.. autoclass:: PKTimeSeriesPlot
    :members:
    :inherited-members: