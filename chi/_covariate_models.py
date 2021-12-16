#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

from warnings import warn

import numpy as np

import chi


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
        \theta = f(\vartheta)  \quad \mathrm{and} \quad
        \psi = g(\vartheta , \eta, \chi ).

    This base class provides an API to implement the functions :math:`f` and
    :math:`g`.
    """

    def __init__(self):
        super(CovariateModel, self).__init__()

        self._parameter_names = None

    def check_compatibility(self, population_model):
        r"""
        Takes an instance of a :class:`PopulationModel` and checks
        the compatibility with this CovariateModel.

        If the model is not compatible, a warning is raised.

        :param population_model: A population model for :math:`\eta`.
        :type population_model: PopulationModel
        """
        raise NotImplementedError

    def compute_individual_parameters(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi`.

        By default ``covariates`` are set to ``None``, such that model
        does not rely on covariates. Each derived :class:`CovariateModel`
        needs to make sure that model reduces to sensible values for
        this edge case.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters :math:`\psi`.
        :rtype: np.ndarray of length (n,)
        """
        raise NotImplementedError

    def compute_individual_sensitivities(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi` and their sensitivities
        with respect to the model parameters :math:`\vartheta` and the relevant
        fluctuation :math:`\eta`.

        By default ``covariates`` are set to ``None``, such that model
        does not rely on covariates. Each derived :class:`CovariateModel`
        needs to make sure that model reduces to sensible values for
        this edge case.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters and sensitivities of shape (p + 1, n).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        raise NotImplementedError

    def compute_population_parameters(self, parameters):
        r"""
        Returns the population model parameters :math:`\theta` for the
        inter-individual fluctuations :math:`\eta`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :returns: Population parameters :math:`\theta` for :math:`\eta`.
        :rtype: np.ndarray of length (p',)
        """
        raise NotImplementedError

    def compute_population_sensitivities(self, parameters):
        r"""
        Returns the population model parameters :math:`\theta` for the
        inter-individual fluctuations :math:`\eta` and their sensitivities
        with respect to the model parameters :math:`\vartheta`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :returns: Population parameters and sensitivities of shape (p, p').
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        raise NotImplementedError

    def get_parameter_names(self):
        """
        Returns the names of the model parameters.
        """
        return self._parameter_names

    def n_covariates(self):
        """
        Returns the number of covariates c.
        """
        raise NotImplementedError

    def n_parameters(self):
        """
        Returns the number of model parameters p.
        """
        raise NotImplementedError

    def set_parameter_names(self, names=None):
        """
        Sets the names of the model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List
        """
        raise NotImplementedError


class CentredLogNormalModel(CovariateModel):
    r"""
    This model implements a reparametrisation of the
    :class:`LogNormalModel` to

    .. math::
        \log \psi = \mu _{\mathrm{log}} + \sigma _{\mathrm{log}} \eta ,

    where :math:`\mu _{\mathrm{log}}` and :math:`\sigma _{\mathrm{log}}` are
    the mean and variance of :math:`\log \psi` and :math:`\eta` are the
    inter-individual fluctuations. :math:`\psi` are the parameters across
    individuals.

    Note that for :math:`\psi` to be log-normally distributed, :math:`\eta` has
    to be distributed according to a standard normal distribution

    .. math::
        \eta \sim \mathcal{N}(0, 1).

    In the notation from :class:`CovariateModel`, the mappings :math:`f` and
    :math:`g` are therefore

    .. math::
        \theta = (0, 1) \quad \mathrm{and} \quad
        \psi = \exp \left(\mu _{\mathrm{log}} + \sigma _{\mathrm{log}} \eta
            \right),

    where the model parameters are
    :math:`\vartheta = (\mu _{\mathrm{log}}, \sigma _{\mathrm{log}})` and the
    new population parameters :math:`\theta` are the mean and variance of the
    Gaussian distribution of :math:`\eta`.

    .. note::
        This model does not implement a model for covariates, but demonstrates
        how the :class:`CovariateModel` interface may be used to reparametrise
        a :class:`PopulationModel`.

    Extends :class:`CovariateModel`.
    """

    def __init__(self):
        super(CentredLogNormalModel, self).__init__()

        # Set number of parameters and covariates
        self._n_parameters = 2
        self._n_covariates = 0

        # Set default parameter names
        self._parameter_names = ['Mean log', 'Std. log']

    def check_compatibility(self, population_model):
        r"""
        Takes an instance of a :class:`PopulationModel` and checks
        the compatibility with this CovariateModel.

        If the model is not compatible, a warning is raised.

        A CentredLogNormalModel is only compatible with a
        :class:`GaussianModel`.

        :param population_model: A population model for :math:`\eta`.
        :type population_model: PopulationModel
        """
        if not isinstance(population_model, chi.GaussianModel):
            warn(
                'This CovariateModel is only intended for the use with a '
                'GaussianModel.', UserWarning)

    def compute_individual_parameters(self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`. In this model
            the covariates do not influence the output.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters :math:`\psi`.
        :rtype: np.ndarray of length (n,)
        """
        # Unpack parameters
        mu, sigma = parameters

        # Compute individual parameters
        eta = np.array(eta)
        psi = np.exp(mu + sigma * eta)

        return psi

    def compute_individual_sensitivities(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi` and their sensitivities
        with respect to the model parameters :math:`\vartheta` and the relevant
        fluctuation :math:`\eta`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`. In this model
            the covariates do not influence the output.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters of shape (n,) and sensitivities of
            shape (p + 1, n).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Unpack parameters
        mu, sigma = parameters

        # Compute individual parameters
        eta = np.array(eta)
        psi = np.exp(mu + sigma * eta)

        # Compute sensitivities
        deta = sigma * psi
        dmu = psi
        dsigma = eta * psi
        sensitivities = np.vstack((deta, dmu, dsigma))

        return (psi, sensitivities)

    def compute_population_parameters(self, parameters):
        r"""
        Returns the population model parameters :math:`\theta` for the
        inter-individual fluctuations :math:`\eta`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :returns: Population parameters :math:`\theta` for :math:`\eta`.
        :rtype: np.ndarray of length (p',)
        """
        # Check that parameters are strictly positive, else return
        # np.nan for the parameters
        _, std_log = parameters
        if std_log <= 0:
            # The standard deviation of a log-normal distribution is
            # strictly positive
            return np.array([np.nan, np.nan])

        # As a result of the `centering` the population parameters for
        # eta (mean and std.) are constant.
        return np.array([0, 1])

    def compute_population_sensitivities(self, parameters):
        r"""
        Returns the population model parameters :math:`\theta` for the
        inter-individual fluctuations :math:`\eta` and their sensitivities
        with respect to the model parameters :math:`\vartheta`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :returns: Population parameters and sensitivities of shape (p, p').
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # As a result of the `centering` the population parameters for
        # eta (mean and std.) are constant.
        return (np.array([0, 1]), np.array([[0, 0]] * len(parameters)))

    def n_covariates(self):
        """
        Returns the number of covariates c.
        """
        return self._n_covariates

    def n_parameters(self):
        """
        Returns the number of model parameters.
        """
        return self._n_parameters

    def set_parameter_names(self, names=None):
        """
        Sets the names of the model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List
        """
        if names is None:
            # Reset names to defaults
            self._parameter_names = ['Mean log', 'Std. log']
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]
