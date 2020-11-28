#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pints


class LogPosterior(pints.LogPosterior):
    """
    A log-posterior class which can be used with the
    :class:`OptimisationController` or the :class:`SamplingController`
    to find either the maximum a posteriori
    estimates of the model parameters, or to sample from the posterior
    probability distribution of the model parameters directly.

    Extends :class:`pints.LogPosterior`.

    Parameters
    ----------

    log_likelihood
        An instance of a :class:`pints.LogPDF`.
    log_prior
        An instance of a :class:`pints.LogPrior` which represents the prior
        probability distributions for the parameters of the log-likelihood.
    """

    def __init__(self, log_likelihood, log_prior):
        super(LogPosterior, self).__init__(log_likelihood, log_prior)

        # Set defaults
        self._id = None
        n_params = self._n_parameters
        self._parameter_names = ['Param %d' % (n+1) for n in range(n_params)]

    def get_id(self):
        """
        Returns the id of the log-posterior. If no id is set, ``None`` is
        returned.
        """
        return self._id

    def get_parameter_names(self):
        """
        Returns the names of the model parameters. By default the parameters
        are enumerated and assigned with the names 'Param #'.
        """
        return self._parameter_names

    def set_id(self, posterior_id):
        """
        Sets the posterior id.

        This can be used to tag the log-posterior to distinguish it from
        other structurally identical log-posteriors, e.g. when the same
        model is used to describe the PKPD of different individuals.

        Parameters
        ----------

        posterior_id
            An ID that can be used to identify the log-posterior. A valid ID
            has to be convertable to a string object.
        """
        self._id = str(posterior_id)

    def set_parameter_names(self, names):
        """
        Sets the names of the model parameters.

        The list of parameters has to match the length of the number of
        parameters. The first parameter name in the list is assigned to the
        first parameter, the second name in the list is assigned to second
        parameter, and so on.

        Parameters
        ----------

        names
            A list of string-convertable objects that is used to assign names
            to the model parameters.
        """
        if len(names) != self._n_parameters:
            raise ValueError(
                'The list of parameter names has to match the number of model '
                'parameters.')
        self._parameter_names = [str(name) for name in names]


class ReducedLogPDF(pints.LogPDF):
    """
    A wrapper for a :class:`pints.LogPDF` to fix the values of a subset of
    model parameters.

    This allows to reduce the parameter dimensionality of the log-pdf
    at the cost of fixing some parameters at a constant value.

    Extends :class:`pints.LogPDF`.

    Parameters
    ----------
    log_pdf
        An instance of a :class:`pints.LogPDF`.
    mask
        A boolean array of the length of the number of parameters. ``True``
        indicates that the parameter is fixed at a constant value, ``False``
        indicates that the parameter remains free.
    values
        A list of values the parameters are fixed at.
    """

    def __init__(self, log_pdf, mask, values):
        super(ReducedLogPDF, self).__init__()

        if not isinstance(log_pdf, pints.LogPDF):
            raise ValueError(
                'The log-pdf has to be an instance of a pints.LogPDF.')

        self._log_pdf = log_pdf

        if len(mask) != self._log_pdf.n_parameters():
            raise ValueError(
                'Length of mask has to match the number of log-pdf '
                'parameters.')

        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise ValueError(
                'Mask has to be a boolean array.')

        n_fixed = int(np.sum(mask))
        if n_fixed != len(values):
            raise ValueError(
                'There have to be as many value inputs as the number of '
                'fixed parameters.')

        # Create a parameter array for later calls of the log-pdf
        self._parameters = np.empty(shape=len(mask))
        self._parameters[mask] = np.asarray(values)

        # Allow for updating the 'free' number of parameters
        self._mask = ~mask
        self._n_parameters = int(np.sum(self._mask))

    def __call__(self, parameters):
        # Fill in 'free' parameters
        self._parameters[self._mask] = np.asarray(parameters)

        return self._log_pdf(self._parameters)

    def n_parameters(self):
        """
        Returns the number of free parameters of the log-posterior.
        """
        return self._n_parameters
