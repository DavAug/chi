.. _chi: https://github.com/DavAug/chi

*****************
Covariate Models
*****************

.. currentmodule:: chi

Covariate models in chi_ can be used to construct complex population
structures that depend on characteristics of individuals or subpopulations,
i.e. covariates of the inter-individual variability. Simple population models
from
`Population Models <https://chi.readthedocs.io/en/latest/population_models.html>`_
are used to describe the variability in a subpopulation, while covariate models
are used to describe the cross-subpopulation variability.

Functional classes
------------------

- :class:`LogNormalLinearCovariateModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: LogNormalLinearCovariateModel
    :members:
    :inherited-members:

Base classes
------------
- :class:`CovariateModel`

Detailed API
^^^^^^^^^^^^

.. autoclass:: CovariateModel
    :members: