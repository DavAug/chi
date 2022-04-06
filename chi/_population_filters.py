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

        self._observations = self._observations[..., order]


class GaussianFilter(PopulationFilter):
    r"""
    Implements a Gaussian population filter.

    A Gaussian population filter approximates the distribution of
    snapshot measurements at time point :math:`t_j` by a Gaussian distribution
    whose mean and variance are estimated from simulated
    measurements

    .. math::
        \log p(\mathcal{D} | \tilde{Y}) =
            \sum _{ij} \log \mathcal{N} (y_{ij} | \mu _j, \sigma ^2_j),

    where the mean :math:`\mu _j` and the variance :math:`\sigma ^2_j` are
    given by empirical estimates from the simulated measurements

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

    where :math:`r` indexes observables and :math:`\mu _{jr}` and
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
        mu = np.mean(simulated_obs, axis=0, keepdims=True)
        var = np.var(simulated_obs, ddof=1, axis=0, keepdims=True)

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
        mu = np.mean(simulated_obs, axis=0, keepdims=True)
        var = np.var(simulated_obs, ddof=1, axis=0, keepdims=True)

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


class GaussianKDEFilter(PopulationFilter):
    r"""
    Implements a Gaussian kernel density estimation population filter.

    A Gaussian KDE population filter approximates the distribution of
    snapshot measurements at time point :math:`t_j` by a Gaussian KDE
    approximation of the simulated measurements. The Gaussian KDE approximation
    is defined by the average over Gaussian probability densities whose means
    are equal to the simulated measurements and the standard deviation (or
    bandwidth) is a hyperparameter. By default the bandwidth is chosen by
    the an adapted rule of thumb.

    The log-likelihood of the simulated measurements with respect to the
    measurements and the filter is defined as

    .. math::
        \log p(\mathcal{D} | \tilde{Y}) =
            \sum _{ij} \log \left( \frac{1}{n_s} \sum _{s=1}^{n_s}
            \mathcal{N} (y_{ij} | \tilde{y}_{sj}, \sigma ^2_j) \right).

    Here, we use :math:`i` to index measured individuals from the dataset,
    :math:`j` to index measurement time points and :math:`s` to index simulated
    measurements. :math:`n_s` denotes the number of simulated measurements per
    time point.

    If ``bandwidth = None``, an adapted rule of thumb is used to estimate an
    appropriate bandwidth for each time point :math:`t_j`

    .. math::
        \sigma ^2_j =
            \left( \frac{4}{3n_s}\right) ^ {1/5}
            \sqrt{\frac{1}{n_j - 1}\sum _i (y_{ij} - \mu _j)^2},

    where :math:`\mu _j = \sum _i y_{ij} / n_j` is the empirical mean
    over the measurements and :math:`n_j` is the number of measurements at time
    :math:`t_j`. Note that this deviates from the standard definition of the
    rule of thumb, where the empirical variance would be estimated from the
    simulated measurements.

    For multiple measured observables the above expression can be
    straightforwardly extended to

    .. math::
        \log p(\mathcal{D} | \tilde{Y}) =
            \sum _{ijr} \log \left( \frac{1}{n_s} \sum _{s=1}^{n_s}
            \mathcal{N} (y_{ijr} | \tilde{y}_{sjr}, \sigma ^2_{jr}) \right),

    where :math:`r` indexes observables and
    :math:`\sigma^2 _{jr}` is the bandwidth for observable :math:`r` at time
    point :math:`t_j`.

    Extends :class:`PopulationFilter`

    :param observations: Snapshot measurements.
    :type observations: np.ndarray of shape
        ``(n_ids, n_observables, n_times)``
    :param bandwidth: Bandwidths of the Gaussian kernels for the different time
        points and observables. By default an adapted rule of thumb is used to
        determine appropriate bandwidths.
    :type bandwidth: np.ndarray of shape ``(n_observables, n_times)``, optional
    """
    def __init__(self, observations, bandwidth=None):
        super().__init__(observations)

        # Add dummy dimension to observations for later convenience
        self._observations = self._observations[np.newaxis, ...]

        # Compute unscaled bandwidth from data
        # (Only used if bandwidth is None)
        self._unscaled_bandwidth = np.std(
            self._observations, ddof=1, axis=1, keepdims=True)

        # Set bandwidth
        self._bandwidth = None
        if bandwidth is not None:
            bandwidth = np.asarray(bandwidth)
            if bandwidth.shape != (self._n_observables, self._n_times):
                raise ValueError(
                    'The bandwidth needs to be of shape '
                    '(n_observables, n_times).')
            if np.any(bandwidth <= 0):
                raise ValueError(
                    'The elements of the bandwidth need to be positive.')
            self._bandwidth = bandwidth[np.newaxis, np.newaxis, ...]

        if self._bandwidth is None:
            if np.ma.is_masked(self._unscaled_bandwidth) or np.any(
                    self._unscaled_bandwidth == 0):
                raise ValueError(
                    'The variance of the data is zero for at least one '
                    'observable and measurement time point, so the rule of '
                    'thumb cannot be used to estimate bandwidths. Please '
                    'provide a bandwidth upon instantiation.')

    def compute_log_likelihood(self, simulated_obs):
        """
        Returns the log-likelihood of the simulated observations with respect
        to the data and filter.

        :param simulated_obs: Simulated snapshot observations.
        :type simulated_obs: np.ndarray of shape
            (n_sim, n_observables, n_times)
        :rtype: float
        """
        n_sim = len(simulated_obs)
        simulated_obs = np.asarray(simulated_obs)[:, np.newaxis, :, :]
        bandwidth = self._bandwidth
        if bandwidth is None:
            bandwidth = (4 / 3 / n_sim) ** 0.2 * self._unscaled_bandwidth

        score = np.sum(logsumexp(
            - (simulated_obs - self._observations)**2
            / bandwidth**2 / 2, axis=0
            ) - np.log(n_sim) - np.log(2 * np.pi) / 2 - np.log(bandwidth))
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
        n_sim = len(simulated_obs)
        simulated_obs = np.asarray(simulated_obs)[:, np.newaxis, :, :]
        bandwidth = self._bandwidth
        if bandwidth is None:
            bandwidth = (4 / 3 / n_sim) ** 0.2 * self._unscaled_bandwidth

        # Compute log-likelihood
        scores = \
            - (simulated_obs - self._observations)**2 / bandwidth**2 / 2
        score = np.sum(
            logsumexp(scores, axis=0) - np.log(n_sim) - np.log(2 * np.pi) / 2
            - np.log(bandwidth))
        if np.ma.is_masked(score):
            n_sim, _, n_obs, n_times = simulated_obs.shape
            return -np.inf, np.empty((n_sim, n_obs, n_times))

        # Compute sensitivities
        # score = log mean exp scores + constant
        # dscore/dsim = exp(scores) / sum(exp(scores)) * dscores / dsim
        dscore_dsim = np.sum(
            softmax(scores, axis=0)
            * (self._observations - simulated_obs) / bandwidth**2, axis=1)

        return score, dscore_dsim


class LogNormalFilter(PopulationFilter):
    r"""
    Implements a lognormal population filter.

    A lognormal population filter approximates the distribution of
    snapshot measurements at time point :math:`t_j` by a lognormal distribution
    whose location and scale are estimated from simulated
    measurements

    .. math::
        \log p(\mathcal{D} | \tilde{Y}) =
            \sum _{ij} \log \mathrm{LN} (y_{ij} | \mu _j, \sigma _j),

    where the location :math:`\mu _j` and the location :math:`\sigma _j` are
    given by empirical estimates of the log-mean and the log-standard deviation
    of the simulated measurements

    .. math::
        \mu _j = \frac{1}{n_s} \sum _{s=1}^{n_s} \log \tilde{y}_{sj}
        \quad \text{and} \quad
        \sigma _j = \sqrt{\frac{1}{n_s-1} \sum _{s=1}^{n_s} \left(
            \log \tilde{y}_{sj} - \mu _j \right) ^2}.

    Here, we use :math:`i` to index measured individuals from the dataset,
    :math:`j` to index measurement time points and :math:`s` to index simulated
    measurements. :math:`n_s` denotes the number of simulated measurements per
    time point.

    For multiple measured observables the above expression can be
    straightforwardly extended to

    .. math::
        \log p(\mathcal{D} | \tilde{Y}) =
            \sum _{ijr} \log \mathrm{LN} (y_{ijr} | \mu _{jr}, \sigma _{jr}),

    where :math:`r` indexes observables and :math:`\mu _{jr}` and
    :math:`\sigma _{jr}` are the empirical mean and variance over the
    simulated log-measurements of the observable :math:`r` at time point
    :math:`t_j`.

    Extends :class:`PopulationFilter`

    :param observations: Snapshot measurements.
    :type observations: np.ndarray of shape
        ``(n_ids, n_observables, n_times)``
    """
    def __init__(self, observations):
        super().__init__(observations)

        # Log-transform the observations for later convenience
        self._log_observations = np.log(self._observations)

    def _compute_log_likelihood(self, mu, var):
        """
        Returns the log-likelihood.

        mu of shape (1, n_observables, n_times)
        var of shape (1, n_observables, n_times)
        """
        score = -np.sum(
            np.log(2*np.pi) + np.log(var) + self._log_observations
            + (self._log_observations - mu)**2 / var) / 2

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
        simulated_obs = np.log(simulated_obs)
        mu = np.mean(simulated_obs, axis=0, keepdims=True)
        var = np.var(simulated_obs, ddof=1, axis=0, keepdims=True)

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
        simulated_obs = np.asarray(simulated_obs)
        log_simulated_obs = np.log(simulated_obs)
        mu = np.mean(log_simulated_obs, axis=0, keepdims=True)
        var = np.var(log_simulated_obs, ddof=1, axis=0, keepdims=True)

        score = self._compute_log_likelihood(mu, var)
        if np.ma.is_masked(score):
            return -np.inf, np.empty(simulated_obs.shape)

        # Compute sensitivities
        # dscore/dsim = dscore/dmu dmu/dsim + dscore/dvar dvar/dsim
        n_sim = len(simulated_obs)
        dscore_dsim = (
            np.sum((self._log_observations - mu) / var, axis=0) / n_sim
            + np.sum(
                (self._log_observations - mu)**2 / var**2 - 1 / var, axis=0
            ) / (n_sim - 1) * ((log_simulated_obs - mu) - np.mean(
                log_simulated_obs - mu, axis=0))
        ) / simulated_obs

        return score, dscore_dsim


def logsumexp(a, axis=None, keepdims=False):  # pragma: no cover
    """
    Returns log of the sum of the exponentiated values of a.

    Code is copied from scipy.special.logsumexp and adapted to work for masked
    arrays.
    """
    a_max = np.max(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        out = np.log(s)

    # Add back shift
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out


def softmax(a, axis=None):  # pragma: no cover
    """
    Returns the softmax of a.

    Code is copied from scipy.special.softmax and adapted to work for masked
    arrays.
    """
    return np.exp(a - logsumexp(a, axis=axis, keepdims=True))
