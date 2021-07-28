#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#


class CovariateModel(object):
    r"""
    A base class for covariate models.

    A CovariateModel assumes that the individual parameters :math:`\psi` are
    distributed according to a population model that is conditional on the
    model parameters :math:`\vartheta` and the covariates :math:`\chi`

    .. math::
        \psi \sim \mathbb{P}(\cdot | \vartheta, \chi).

    Here, covariates can vary from one individual to the next, while the model
    parameters :math:`\vartheta` are the same for all individuals.

    To simplify this dependence, CovariateModels make the assumption that the
    distribution :math:`\mathbb{P}(\psi | \vartheta, \chi)` deterministically
    varies with the covariates, such that the problem can be recast in terms
    of a covariate-independent distribution of inter-individual
    fluctuations :math:`\eta`

    .. math::
        \eta \sim \mathbb{P}(\cdot | \theta)

    and a set of deterministic relationships for the individual parameters
    :math:`\psi`  and the new population parameters :math:`\theta`

    .. math::
        \theta = f(\vartheta)
        \psi = g(\vartheta , \eta, \chi ).

    This base class provides an API to implement the functions :math:`f` and
    :math:`g`.
    """

    def __init__(self):
        super(CovariateModel, self).__init__()

    def compute_individual_parameters(self, parameters, eta, covariates):
        r"""
        Returns the individual parameters :math:`\psi`.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters :math:`\psi`.
        :rtype: np.ndarray of length (n,)
        """
        raise NotImplementedError

    def compute_population_parameters(self, parameters):
        r"""
        Returns the population model parameters :math:`\theta` for the
        inter-indiviudal fluctuations :math:`\eta`.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of length (p,)
        :returns: Population parameters :math:`\theta` for :math:`\eta`.
        :rtype: np.ndarray of length (p',)
        """
        raise NotImplementedError
