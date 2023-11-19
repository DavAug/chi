.. currentmodule:: chi

**********************************
Fitting mechanistic models to data
**********************************

In the previous tutorial, :doc:`mechanistic_model`, we have seen how we can
implement and simulate treatment response models in Chi. For example, using the same
1-compartment PK model from before, ``one_compartment_pk_model.xml``, we can simulate
the time course of drug concentration levels following repeated bolus
adminstrations

.. literalinclude:: code/3_fitting_models_1.py
    :lines: 18-51

.. raw:: html
   :file: images/3_fitting_models_1.html

This ability to simulate treatment response is pretty cool in its own right,
but, at this point, our model is completely made up and has nothing to do with
real treatment responses. If our goal is to describe *real* treatment
responses, we need to somehow connect our model to reality.

The most common approach to relate models to real treatment responses is to
compare the model predictions to measurements.

.. csv-table:: Drug concentration measurements
   :file: data/3_fitting_models_1.csv
   :widths: 4, 12, 12, 12, 12, 12, 12, 12, 12
   :header-rows: 1

.. autosummary::

    chi.ErrorModel
    chi.GaussianErrorModel
    chi.LogNormalErrorModel
    chi.MultiplicativeGaussianErrorModel
    chi.ConstantAndMultiplicativeGaussianErrorModel
    chi.ReducedErrorModel
    chi.LogLikelihood
    chi.LogPosterior
    chi.SamplingController