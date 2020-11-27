#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import numpy as np
import pints


class ReducedLogPDF(pints.LogPDF):
    """
    A wrapper for a :class:`pints.LogPDF` to fix the values of a subset of
    model parameters.

    This allows to reduce the parameter dimensionality of the log-pdf
    at the cost of fixing some parameters at a constant value.

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
