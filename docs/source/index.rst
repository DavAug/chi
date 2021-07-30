
.. Root of all pints docs

.. _GitHub: https://github.com/DavAug/chi

.. module:: chi

.. toctree::
    :hidden:
    :maxdepth: 1

    covariate_models
    error_models
    inference
    library/index
    log_pdfs
    mechanistic_models
    plots/index
    population_models
    predictive_models
    problems

Welcome to Chi's documentation!
=====================================

**Chi** is an open source Python package hosted on GitHub_,
which is designed for pharmacokinetic and pharmacodynamic (PKPD) modelling.

The main features of chi are

- Simulation of mechanistic dose response models (differential equations) for arbitrary dosing regimens.
- Inference of mechanistic model parameters from data (classical or Bayesian).
- Simulation of the dose response variability in a population (hierarchical models/non-linear mixed effects models).
- Inference of population parameters from data (classical or Bayesian).
- Simulation of structured populations, where inter-individual variability can be partly explained by covariates.
- Inference of model parameters in a structured population from data (classical or Bayesian).

This page provides the API, or developer documentation for
chi.

.. note::
    This package is still in its infancy and is continuously being developed.
    So if you find any bugs, please don't hesitate to reach out to us and share
    your feedback.
