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

        self._n_parameters = None
        self._n_covariates = None
        self._parameter_names = None
        self._covariate_names = None

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

    def get_covariate_names(self):
        """
        Returns the names of the covariates.
        """
        return self._covariate_names

    def get_parameter_names(self):
        """
        Returns the names of the model parameters.
        """
        return self._parameter_names

    def n_covariates(self):
        """
        Returns the number of covariates c.
        """
        return self._n_covariates

    def n_parameters(self):
        """
        Returns the number of model parameters p.
        """
        return self._n_parameters

    def set_covariate_names(self, names=None):
        """
        Sets the names of the covariates.

        :param names: A list of covariate names. If ``None``, covariate names
            are reset to defaults.
        :type names: List
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

        # Set default names
        self._parameter_names = ['Mean log', 'Std. log']
        self._covariate_names = []

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
        if sigma <= 0:
            # The standard deviation of a log-normal distribution is
            # strictly positive
            return np.array([np.nan] * len(eta))

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
        if sigma <= 0:
            # The standard deviation of a log-normal distribution is
            # strictly positive
            return (
                np.full(fill_value=np.nan, shape=len(eta)),
                np.full(fill_value=np.nan, shape=(3, len(eta)))
            )

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

    def set_covariate_names(self, names=None):
        """
        Sets the covariate names.

        Model does not have covariates, so any input will be ignored.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List
        """
        return None

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


class LogNormalLinearCovariateModel(CovariateModel):
    r"""
    This model implements a reparametrisation of the
    :class:`LogNormalModel` population model, where the mean on the log-scale
    is a linear function of covariates

    .. math::
        \psi \sim \mathrm{Lognormal}\left(
            \mu _{\mathrm{log}}(\chi ),  \sigma _{\mathrm{log}}\right) ,

    where :math:`\psi` denotes the parameters of the individual models.
    :math:`\mu _{\mathrm{log}}` and :math:`\sigma _{\mathrm{log}}` are
    the mean and standard deviation of :math:`\log \psi`.
    The mean :math:`\mu _{\mathrm{log}}` depends linearly on the covariates
    :math:`\chi`

    .. math::
        \mu _{\mathrm{log}}(\chi ) =
            \mu _{\mathrm{log}, 0} + \Delta \mu _{\mathrm{log}} \chi ,

    where :math:`\mu _{\mathrm{log}, 0}` is the mean when the covariates are
    zero, and :math:`\Delta \mu _{\mathrm{log}}` are the relative shifts per
    unit from this baseline.
    The population standard deviation is assumed to be independent of the
    covariates.

    The model is instantiated in the non-centered parametrisation.
    In the non-centered parametrisation the parameters are transformed to
    standardised inter-individual fluctations which are modelled as Gaussian
    random variables

    .. math::
        \eta \sim \mathcal{N}(0, 1).

    These inter-individual fluctuations are related to the individual
    parameters by

    .. math::
        \psi =
            \exp \left(
                \mu _{\mathrm{log}(\chi )} + \sigma _{\mathrm{log}} \eta
            \right) .

    The model parameters are
    :math:`\vartheta = (\mu _{\mathrm{log}, 0}, \sigma _{\mathrm{log}},
    \Delta \mu _{\mathrm{log}, 1}, \ldots , \Delta \mu _{\mathrm{log}, c}`.

    Extends :class:`CovariateModel`.
    """
    def __init__(self, n_covariates=0):
        super(LogNormalLinearCovariateModel, self).__init__()

        # Set number of parameters and covariates
        self._n_covariates = int(n_covariates)
        self._n_parameters = 2 + self._n_covariates

        # Set default names
        self._covariate_names = [
            'Covariate %d' % int(c + 1) for c in range(self._n_covariates)]
        self._parameter_names = ['Base mean log', 'Std. log'] + [
            'Shift %s' % n for n in self._covariate_names]

    def check_compatibility(self, population_model):
        r"""
        Takes an instance of a :class:`PopulationModel` and checks
        the compatibility with this CovariateModel.

        If the model is not compatible, a warning is raised.

        In the non-centered parametrisation the model is only
        compatible with a :class:`GaussianModel`.

        :param population_model: A population model for the individuals'
            parameters.
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
        mu_base, sigma = parameters[:2]
        shifts = np.array(parameters[2:])
        if sigma <= 0:
            # The standard deviation of a log-normal distribution is
            # strictly positive
            return np.array([np.nan] * len(eta))

        # Compute individual parameters
        mu = mu_base
        if self._n_covariates > 0:
            mu = mu_base + covariates @ shifts

        eta = np.array(eta)
        psi = np.exp(mu + sigma * eta)

        return psi

    def compute_individual_sensitivities(
            self, parameters, eta, covariates=None):
        r"""
        Returns the individual parameters :math:`\psi` and their sensitivities
        with respect to the model parameters :math:`\vartheta` and the relevant
        fluctuations :math:`\eta`.

        :param parameters: Model parameters :math:`\vartheta`.
        :type parameters: np.ndarray of length (p,)
        :param eta: Inter-individual fluctuations :math:`\eta`.
        :type eta: np.ndarray of length (n,)
        :param covariates: Individual covariates :math:`\chi`. In this model
            the covariates do not influence the output.
        :type covariates: np.ndarray of length (n, c)
        :returns: Individual parameters of shape (n,) and sensitivities of
            shape (1 + p, n).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Unpack parameters
        mu_base, sigma = parameters[:2]
        shifts = np.array(parameters[2:])
        if sigma <= 0:
            # The standard deviation of a log-normal distribution is
            # strictly positive
            return (
                np.full(fill_value=np.nan, shape=len(eta)),
                np.full(fill_value=np.nan, shape=(3, len(eta)))
            )

        # Compute individual parameters
        mu = mu_base
        if self._n_covariates > 0:
            mu = mu_base + covariates @ shifts

        eta = np.array(eta)
        psi = np.exp(mu + sigma * eta)

        # Compute sensitivities
        deta = sigma * psi
        dmu_base = psi
        dsigma = eta * psi

        sensitivities = np.vstack((deta, dmu_base, dsigma))
        if self._n_covariates > 0:
            dshifts = covariates.T * psi[np.newaxis, :]
            sensitivities = np.vstack((sensitivities, dshifts))

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

    def set_covariate_names(self, names=None, update_param_names=False):
        """
        Sets the covariate names.

        :param names: A list of parameter names. If ``None``, covariate names
            are reset to defaults.
        :type names: List
        :param update_param_names: Boolean flag indicating whether parameter
            names should be updated according to new covariate names. By
            default parameter names are not updated.
        :type update_param_names: bool, optional
        """
        if names is None:
            # Reset names to defaults
            self._covariate_names = [
                'Covariate %d' % int(c + 1) for c in range(self._n_covariates)
            ]
            return None

        if len(names) != self._n_covariates:
            raise ValueError(
                'Length of names does not match n_covariates.')

        self._covariate_names = [str(label) for label in names]

        if update_param_names:
            self.set_parameter_names()

    def set_parameter_names(self, names=None):
        """
        Sets the names of the model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List
        """
        if names is None:
            # Reset names to defaults
            covariate_names = self.get_covariate_names()
            self._parameter_names = ['Base mean log', 'Std. log'] + [
                'Shift %s' % name for name in covariate_names]
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match n_parameters.')

        self._parameter_names = [str(label) for label in names]
