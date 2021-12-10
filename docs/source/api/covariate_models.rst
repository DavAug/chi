.. _chi: https://github.com/DavAug/chi

*****************
Covariate Models
*****************

.. currentmodule:: chi

Covariate models in chi_ can be used to construct complex population
structures that depend on certain characteristics of subpopulations, i.e. covariates
of the inter-individual variability. Simple population models from
`Population Models <https://chi.readthedocs.io/en/latest/population_models.html>`_
are used to describe the variability in a subpopulation, while covariate models are
used to define how those subpopulations differ based on the covariates.

Besides being able to construct rich population structures, covariate models also allow users
to reparametrise population models, e.g. moving from a non-centred to a centred parametrisation.

Functional classes
------------------

- :class:`CentredLogNormalModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: CentredLogNormalModel
    :members:
    :inherited-members:

Base classes
------------
- :class:`CovariateModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: CovariateModel
    :members: