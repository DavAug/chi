#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

# TODO: rename models to mechanistic models

import numpy as np


class PopulationModel(object):
    """
    A base class for population models.

    Parameters
    ----------

    n_ids
        Number of individual bottom level models.
    n_parameters_per_id
        Number of parameters that are modelled per individual. By default
        the value is set to ``None`` and it is assumed that only one
        parameter per individual is modelled.
    """

    def __init__(self, n_ids):
        super(PopulationModel, self).__init__()

        # This is going to be used to define the number of parameters.
        self._n_ids = n_ids

    def __call__(self, parameters):
        """
        Returns the log-likelihood score of the population model.

        The parameters are expected to be of length :meth:`n_parameters`. The
        first :meth:`n_bottom_parameters` parameters are treated as the
        'observations' of the individual model parameters, and the remaining
        :meth:`n_top_parameters` specify the values of the population
        model parameters.

        If ``n_parameters_per_id > 1``, it is assumed that the parameter values
        of one type of parameter across individuals are grouped together.
        """
        raise NotImplementedError

    def n_bottom_parameters(self):
        """
        Returns the number of bottom-level parameters of the population model.

        This is the total number of input parameters from the individual
        likelihoods.
        """
        raise NotImplementedError

    def n_ids(self):
        """
        Returns the number of modelled individuals of the population model.
        """
        return self._n_ids

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        raise NotImplementedError

    def n_parameters_per_id(self):
        """
        Returns the number of parameters per likelihood that are modelled by
        the population model.
        """
        return 1

    def n_top_parameters(self):
        """
        Returns the number of top parameters of the population.

        This is the number of population parameters.
        """
        raise NotImplementedError

    def sample(self, top_parameters, n=None):
        r"""
        Returns `n` random samples from the underlying population distribution.

        The returned value is a numpy array with shape :math:`(n, 1)` where
        :math:`n` is the requested number of samples.
        """
        raise NotImplementedError


class PooledModel(PopulationModel):
    """
    A population model that pools the model parameters across indiviudals.

    A pooled model assumes that the parameters across individuals do not vary.
    As a result, all individual parameters are set to the same value.

    Calling the PooledModel returns a constant, irrespective of the parameter
    values. We chose this constant to be ``0``.

    Extends :class:`erlotinib.PopulationModel`.
    """

    def __init__(self, n_ids):
        super(PooledModel, self).__init__(n_ids)

        # Set number of input individual parameters
        self._n_bottom_parameters = 0

        # Set number of population parameters
        self._n_top_parameters = 1

        # Set number of parameters
        self._n_parameters = self._n_bottom_parameters + self._n_top_parameters

    def __call__(self, parameters):
        """
        Returns the log-likelihood score of the population model.

        The log-likelihood score of a PooledModel is independent of the input
        parameters. We choose to return a score of ``0``.

        The parameters are expected to be of length :meth:`n_parameters`. The
        first :meth:`nids` parameters are treated as the 'observations' of the
        individual model parameters, and the remaining
        :meth:`n_top_parameters` specify the values of the population
        model parameters.
        """
        return 0

    def n_bottom_parameters(self):
        """
        Returns the number of bottom-level parameters of the population model.

        This is the total number of input parameters from the individual
        likelihoods.
        """
        return self._n_bottom_parameters

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters

    def n_top_parameters(self):
        """
        Returns the number of top parameters of the population.

        This is the number of population parameters.
        """
        return self._n_top_parameters

    def sample(self, top_parameters, n=None):
        r"""
        Returns :math:`n` random samples from the underlying population
        distribution.

        For a PooledModel the input top-level parameters are copied for each
        individual and are returned :math:`n` times.

        The returned value is a numpy array with shape :math:`(n, d)` where
        :math:`n` is the requested number of samples.
        """
        if len(top_parameters) != self._n_top_parameters:
            raise ValueError(
                'The number of provided parameters does not match the expected'
                ' number of top-level parameters.')

        # Expand dimension of top level parameters
        top_parameters = np.asarray(top_parameters)
        samples = np.expand_dims(top_parameters, axis=0)

        if n is None:
            return samples

        samples = np.broadcast_to(samples, shape=(n, self._n_top_parameters))
        return samples
