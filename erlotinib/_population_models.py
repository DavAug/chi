#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

#TODO: rename models to mechanistic models


class PopulationModel(object):
    """
    To be completed.
    """

    def __init__(self, n_ids, n_bottom_params=None):
        super(PopulationModel, self).__init__()

        # This is going to be used to define the number of parameters.
        self._n_ids = n_ids
        self._n_bottom_params = n_bottom_params

    def __call__(self, parameters):
        """
        Returns the log-likelihood score of the population model.

        The parameters are expected to be of length :meth:`n_parameters`. The
        first :meth:`nids` parameters are treated as the 'observations' of the
        individual model parameters, and the remaining
        :meth:`n_population_parameters` specify the values of the population
        model parameters.
        """
        raise NotImplementedError

    def n_bottom_params(self):
        """
        Returns number of bottom-level parameters per individual that are
        modelled by the population model.
        """
        return self._n_bottom_params

    def n_ids(self):
        """
        Returns the number of modelled individuals of the population model.
        """
        return self._n_ids

    def n_parameters(self):
        """
        Returns the number of parameters of the population model.
        """
        return self._n_parameters
