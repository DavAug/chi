#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy

import numpy as np


class CovariateModel(object):
    r"""
    A base class for covariate models.

    Covariate models model parameters of population models as functions of
    covariates

    .. math::
        \vartheta = \vartheta(\theta, \chi),

    where :math:`\vartheta` are the parameters of a
    :class:`chi.PopulationModel`, :math:`\chi` are the covariates and
    :math:`\theta` are the new parameters that govern the
    inter-individual variability of the population.

    This allows you to distinguish subpopulations in the population

    .. math::
        p(\psi | \theta) =
            \int \mathrm{d}\chi \,
            p(\psi | \vartheta(\theta, \chi) )\, p(\chi),

    where :math:`p(\chi)` is the distribution of the covariates in the
    population (for discrete covariates the integrals becomes a sum).
    Each subpopulation is characterised by a unique set of covariates.

    By default, only the first population parameter is transformed. The
    parameters to transform can be selected with
    :meth:`set_population_parameters`.

    :param n_cov: Number of covariates.
    :type n_cov: int, optional
    :param cov_names: Names of the covariates.
    :type cov_names: List[str], optional
    """

    def __init__(self, n_cov=1, cov_names=None):
        super(CovariateModel, self).__init__()
        # Check inputs
        n_cov = int(n_cov)
        if n_cov < 1:
            raise ValueError(
                'The number of covariates has to be greater or equal to 1.')
        self._n_cov = n_cov
        if cov_names:
            if len(cov_names) != self._n_cov:
                raise ValueError(
                    'The number of covariate names has to match the number of '
                    'covariates.')
            cov_names = [str(name) for name in cov_names]
        else:
            cov_names = [
                'Cov. %d' % (id_cov + 1) for id_cov in range(self._n_cov)]
        self._cov_names = cov_names

        # Set defaults
        self._pidx = np.array([0])
        self._didx = np.array([0])
        self._n_selected = 1
        self._n_parameters = self._n_selected * self._n_cov
        self._parameter_names = ['Param. 1'] * self._n_cov

    def compute_population_parameters(self, parameters):
        """
        Returns the transformed population model parameters.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_selected, n_cov)``
        :param pop_parameters: Population model parameters.
        :type pop_parameters: np.ndarray of shape
            ``(n_pop_params_per_dim, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: np.ndarray of shape ``(n_ids, n_pop_params_per_dim, n_dim)``
        """
        raise NotImplementedError

    def compute_sensitivities(
            self, parameters, pop_parameters, covariates, dlogp_dvartheta):
        """
        Returns the sensitivities of the likelihood with respect to
        the model parameters and the population model parameters.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_selected, n_cov)``
        :param pop_parameters: Population model parameters.
        :type pop_parameters: np.ndarray of shape
            ``(n_pop_params_per_dim, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :param dlogp_dvartheta: Unflattened sensitivities of the population
            model to the transformed parameters.
        :type dlogp_dvartheta: np.ndarray of shape
            ``(n_ids, n_param_per_dim, n_dim)``
        :rtype: Tuple[np.ndarray of shape ``(n_pop_params,)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        raise NotImplementedError

    def get_covariate_names(self):
        """
        Returns the names of the covariates.
        """
        return self._cov_names

    def get_parameter_names(self, exclude_cov_names=False):
        """
        Returns the names of the model parameters.

        :param exclude_cov_names: A boolean flag that indicates whether the
            covariate name is appended to the parameter name.
        :type exclude_cov_names: bool, optional
        """
        # Append covariate names
        param_names = copy.copy(self._parameter_names)
        if not exclude_cov_names:
            names = []
            for name_id, name in enumerate(param_names):
                cov_name = self._cov_names[name_id % self._n_cov]
                names += [name + ' ' + cov_name]
            param_names = names

        return param_names

    def get_set_population_parameters(self):
        """
        Returns the indices of the population parameters that are transformed.

        Indices are returned as a tuple of arrays, where the first array are
        parameters indices and the second array are the dimension indicies.

        :rtype: Tuple[np.ndarray of shape ``(n_selected,)``,
            np.ndarray of shape ``(n_selected,)``]
        """
        return (self._pidx.copy(), self._didx.copy())

    def n_covariates(self):
        """
        Returns the number of covariates c.
        """
        return self._n_cov

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
        if names is None:
            # Reset names to defaults
            self._cov_names = [
                'Cov. %d' % (id_cov + 1) for id_cov in range(self._n_cov)]
            return None

        if len(names) != self._n_cov:
            raise ValueError(
                'Length of names does not match number of covariates.')

        self._cov_names = [str(label) for label in names]

    def set_parameter_names(self, names=None, mask_names=False):
        """
        Sets the names of the model parameters.

        :param names: A list of parameter names. If ``None``, parameter names
            are reset to defaults.
        :type names: List
        """
        if names is None:
            # Reset names to defaults
            names = []
            for id_p in range(self._n_selected):
                names += ['Param. %d' % (id_p + 1)] * self._n_cov
            self._parameter_names = names
            return None

        if len(names) != self._n_parameters:
            raise ValueError(
                'Length of names does not match number of covariates.')

        self._parameter_names = [str(label) for label in names]

    def set_population_parameters(self, indices):
        """
        Sets the parameters of the population model that are transformed by the
        covariate model.

        Note that this influences the number of model parameters.

        .. warning::
            Whether or not the indices are out of bounds cannot be checked at
            this point. It is assumed that for any future call which requires
            ``pop_parameters``, the selected indices are compatible with the
            input.

        :param indices: A list of parameter indices
            [param index per dim, dim index].
        :type indices: List[List[int]]
        """
        raise NotImplementedError


class LinearCovariateModel(CovariateModel):
    r"""
    A linear covariate model.

    A linear covariate model transforms the parameters of the
    population model :math:`\vartheta` linearly in the covariates

    .. math::
        \vartheta (\theta, \chi ) =
            \vartheta _{0} + \sum _c \beta _c \, \chi _c,

    where :math:`\chi = \{ \chi _c\}` are the covariates,
    :math:`\vartheta _0` are the original parameters of the population model,
    and :math:`\theta = (\vartheta _0, \beta)` are the new population
    model parameters.

    By default, only the first population parameter is transformed. The
    parameters can be selected with :meth:`set_population_parameters`.

    Extends :class:`CovariateModel`.

    :param n_cov: Number of covariates.
    :type n_cov: int, optional
    :param cov_names: Names of the covariates.
    :type cov_names: List[str], optional
    """
    def __init__(self, n_cov=1, cov_names=None):
        super(LinearCovariateModel, self).__init__(n_cov, cov_names)

        # Set number of parameters
        self._n_parameters = self._n_cov

    def compute_population_parameters(
            self, parameters, pop_parameters, covariates):
        """
        Returns the transformed population model parameters.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_selected, n_cov)``
        :param pop_parameters: Population model parameters.
        :type pop_parameters: np.ndarray of shape
            ``(n_pop_params_per_dim, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :rtype: np.ndarray of shape ``(n_ids, n_pop_params_per_dim, n_dim)``
        """
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters.reshape(self._n_selected, self._n_cov)
        parameters = parameters.T

        # Compute population parameters
        n_pop, n_dim = pop_parameters.shape
        n_ids = len(covariates)
        vartheta = np.zeros((n_ids, n_pop, n_dim))
        vartheta += pop_parameters[np.newaxis, ...]
        vartheta[:, self._pidx, self._didx] += covariates @ parameters

        return vartheta

    def compute_sensitivities(
            self, parameters, pop_parameters, covariates, dlogp_dvartheta):
        """
        Returns the sensitivities of the likelihood with respect to
        the model parameters and the population model parameters.

        :param parameters: Model parameters.
        :type parameters: np.ndarray of shape ``(n_parameters,)`` or
            ``(n_selected, n_cov)``
        :param pop_parameters: Population model parameters.
        :type pop_parameters: np.ndarray of shape
            ``(n_pop_params_per_dim, n_dim)``
        :param covariates: Covariates of individuals.
        :type covariates: np.ndarray of shape ``(n_ids, n_cov)``
        :param dlogp_dvartheta: Unflattened sensitivities of the population
            model to the transformed parameters.
        :type dlogp_dvartheta: np.ndarray of shape
            ``(n_ids, n_param_per_dim, n_dim)``
        :rtype: Tuple[np.ndarray of shape ``(n_pop_params,)``,
            np.ndarray of shape ``(n_parameters,)``]
        """
        parameters = np.asarray(parameters)
        if parameters.ndim == 1:
            parameters = parameters.reshape(self._n_selected, self._n_cov)
        parameters = parameters.T

        # Compute sensitivities
        n_pop, n_dim = pop_parameters.shape
        n_pop = n_pop * n_dim
        dpop = np.sum(dlogp_dvartheta, axis=0).flatten()
        dparams = np.sum(
            dlogp_dvartheta[:, self._pidx, self._didx, np.newaxis]
            * covariates[:, np.newaxis, :], axis=0).flatten()

        return dpop, dparams

    def set_population_parameters(self, indices):
        """
        Sets the parameters of the population model that are transformed by the
        covariate model.

        Note that this influences the number of model parameters.

        .. warning::
            Whether or not the indices are out of bounds cannot be checked at
            this point. It is assumed that for any future call which requires
            ``pop_parameters``, the selected indices are compatible with the
            input.

        :param indices: A list of parameter indices
            [param index per dim, dim index].
        :type indices: List[List[int]]
        """
        # Keep only unique index pairs
        unique = []
        for idx in indices:
            if idx not in unique:
                unique.append([int(idx[0]), int(idx[1])])

        # Split into param index and dim index
        # (sort 1. dimension axis, 2. parameter axis, so indexing preserves
        # order of np.ndarray.flatten)
        indices = np.array(unique)
        indices = indices[np.argsort(indices[:, 1]), :]
        indices = indices[np.argsort(indices[:, 0]), :]
        self._pidx = indices[:, 0]
        self._didx = indices[:, 1]

        # Update number of parameters and parameters names
        n_params = len(self._pidx)
        self._n_parameters = self._n_cov * n_params
        self._n_selected = n_params
        self.set_parameter_names()
