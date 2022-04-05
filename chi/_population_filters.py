#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np


class PopulationFilter(object):
    r"""
    A base class for population filters.

    A population filter estimates the likelihood with which simulated
    observations :math:`\tilde{y}_{sj}` come from the same distribution as the
    measurements :math:`y_{ij}`, where :math:`s` indexes simulated individuals,
    :math:`j` time points and :math:`i` measured individuals.

    Formally the log-likelihood of the simulated observations with respect to
    the population filter is defined as

    .. math::
        \log p(Y | \tilde{Y}) = \sum _{ij} \log p(y_{ij} | \tilde{Y}_j ),

    where :math:`\tilde{Y}_j = \{ \tilde{}_{sj} \}` are all simulated
    observations at time point :math:`t_j`.

    The measurements are expected to be arranged into a 3 dimensional numpy
    array of shape ``(n_ids, n_observables, n_times)``, where
    ``n_ids`` is the number of measured individuals at a given time point,
    ``n_observables`` is the number of unique observables that were
    measurement, and ``n_times`` is the number of unique time points.
    The population filter expects the simulated measurements to be ordered in
    the same way, so no record of the measurement times is needed.

    If varying numbers of individuals were measured at different time points,
    or not all observables were measured for each individual, the missing
    values can be filled with ``np.nan`` to be able to shape the observations
    into the format ``(n_ids, n_observables, n_times)``. The missing values
    will be filtered internally.

    :param observations: Snapshot measurements.
    :type observations: np.ndarray of shape
        ``(n_ids, n_observables, n_times)``
    """
    def __init__(self, observations):
        observations = np.asarray(observations)
        if observations.ndim != 3:
            raise ValueError(
                'The observations must be of shape '
                '(n_ids, n_observables, n_times).')

        # Transform into masked array
        mask = np.isnan(observations)
        if np.any(mask):
            observations = np.ma.array(observations, mask=mask)

        self._observations = observations
        _, self._n_observables, self._n_times = observations.shape

    def compute_log_likelihood(self, simulated_obs):
        """
        Returns the log-likelihood of the simulated observations with respect
        to the data and filter.

        :param simulated_obs: Simulated snapshot observations.
        :type simulated_obs: np.ndarray of shape
            (n_sim, n_observables, n_times)
        :rtype: float
        """
        raise NotImplementedError

    def compute_sensitivities(self, simulated_obs):
        """
        Returns the log-likelihood of the simulated observations with respect
        to the data and filter, and the sensitivities of the log-likelihood
        with respect to the simulated observations.

        :param simulated_obs: Simulated snapshot observations.
        :type simulated_obs: np.ndarray of shape
            (n_sim, n_observables, n_times)
        :rtype: Tuple[float, np.ndarray] where the array has shape
            ``(n_sim, n_observables, n_times)``
        """
        raise NotImplementedError

    def n_observables(self):
        """
        Returns the number of observables in the dataset.
        """
        return self._n_observables

    def n_times(self):
        """
        Returns the number of measurement times in the dataset.
        """
        return self._n_times

    def sort_times(self, order):
        """
        Sorts the observations along the time dimension according to the
        provided indices.

        :param order: An array with indices that orders the observations along
            thetime dimension.
        :type order: np.ndarray of shape ``(n_times,)``
        """
        order = np.asarray(order)
        if len(order) != self._n_times:
            raise ValueError('Order has to be of length n_times.')
        if len(order) != len(np.unique(order)):
            raise ValueError('Order has to contain n_times unique elements.')

        self._observations = self._observations[:, :, order]


class GaussianPopulationFilter(PopulationFilter):
    r"""
    Implements a Gaussian population filter.

    A Gaussian population filter approximates the distribution of snapshot
    snapshot measurements at time point :math:`t_j` by a Gaussian distribution
    whose mean and variance are estimated from simulated
    measurements

    .. math::
        \log p(\mathcal{D} | \tilde{Y}) =
            \sum _{ij} \log \mathcal{N} (y_{ij} | \mu _j, \sigma ^2_j),

    where the mean :math:`\mu _j` and the variance are given by the empirical
    estimates

    .. math::
        \mu _j = \frac{1}{n_s} \sum _{s=1}^{n_s} \tilde{y}_{sj}
        \quad \text{and} \quad
        \sigma ^2 _j = \frac{1}{n_s-1} \sum _{s=1}^{n_s} \left(
            \tilde{y}_{sj} - \mu _j \right) ^2.

    Here, we use :math:`i` to index measured individuals from the dataset,
    :math:`j` to index measurement time points and :math:`s` to index simulated
    measurements. :math:`n_s` denotes the number of simulated measurements per
    time point.

    For multiple measured observables the above expression can be
    straightforwardly extended to

    .. math::
        \log p(\mathcal{D} | \tilde{Y}) =
            \sum _{ijr} \log \mathcal{N} (y_{ijr} | \mu _{jr}, \sigma ^2_{jr}),

    where :math:`r` indexed observables and :math:`\mu _{jr}` and
    :math:`\sigma^2 _{jr}` are the empirical mean and variance over the
    simulated measurements of the observable :math:`r` at time point
    :math:`t_j`.

    Extends :class:`PopulationFilter`

    :param observations: Snapshot measurements.
    :type observations: np.ndarray of shape
        ``(n_ids, n_observables, n_times)``
    """
    def __init__(self, observations):
        super().__init__(observations)

    def _compute_log_likelihood(self, mu, var):
        """
        Returns the log-likelihood.

        mu of shape (1, n_observables, n_times)
        var of shape (1, n_observables, n_times)
        """
        score = -np.sum(
            np.log(2*np.pi) + np.log(var)
            + (self._observations - mu)**2 / var) / 2

        return score

    def compute_log_likelihood(self, simulated_obs):
        """
        Returns the log-likelihood of the simulated observations with respect
        to the data and filter.

        :param simulated_obs: Simulated snapshot observations.
        :type simulated_obs: np.ndarray of shape
            (n_sim, n_observables, n_times)
        :rtype: float
        """
        mu = np.mean(simulated_obs, axis=0)[np.newaxis, ...]
        var = np.var(simulated_obs, ddof=1, axis=0)[np.newaxis, ...]

        score = self._compute_log_likelihood(mu, var)
        if np.ma.is_masked(score):
            return -np.inf

        return score

    def compute_sensitivities(self, simulated_obs):
        """
        Returns the log-likelihood of the simulated observations with respect
        to the data and filter, and the sensitivities of the log-likelihood
        with respect to the simulated observations.

        :param simulated_obs: Simulated snapshot observations.
        :type simulated_obs: np.ndarray of shape
            (n_sim, n_observables, n_times)
        :rtype: Tuple[float, np.ndarray] where the array has shape
            ``(n_sim, n_observables, n_times)``
        """
        mu = np.mean(simulated_obs, axis=0)[np.newaxis, ...]
        var = np.var(simulated_obs, ddof=1, axis=0)[np.newaxis, ...]

        score = self._compute_log_likelihood(mu, var)
        if np.ma.is_masked(score):
            return -np.inf, np.empty(simulated_obs.shape)

        # Compute sensitivities
        # dscore/dsim = dscore/dmu dmu/dsim + dscore/dvar dvar/dsim
        n_sim = len(simulated_obs)
        dscore_dsim = \
            np.sum((self._observations - mu) / var, axis=0) / n_sim \
            + np.sum(
                - 1 / var + (self._observations - mu)**2 / var**2,
                axis=0) * (simulated_obs - mu) / (n_sim - 1)

        return score, dscore_dsim
