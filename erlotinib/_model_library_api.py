#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os


class ModelLibrary(object):
    """
    Contains references to SBML models.
    """

    def __init__(self):
        # Get path to model library
        self._path = os.path.dirname(os.path.abspath(__file__))
        self._path += '/model_library/'

    def tumour_growth_inhibition_model_koch(self):
        r"""
        Returns the absolute path to a SBML file, specifying the tumour growth
        inhibition pharmacodynamic model introduced by Koch et al. in [1]_.

        In this model the tumour growth inhibition is modelled by an empirical
        model of the tumour volume :math:`V_T` over time

        .. math::
            \frac{\text{d}V_T}{\text{d}t} =
                \frac{2\lambda_0\lambda_1 V_T}
                {2\lambda_0V_T + \lambda_1} - \kappa C V_T.


        Here, the tumour growth in absence of the drug is assumed to grow
        exponentially at rate :math:`2\lambda_0` for tumour volumes below some
        critical volume :math:`V_{\text{crit}}`. For volumes beyond
        :math:`V_{\text{crit}}` the growth dynamics is assumed to slow down
        and transition to a linear growth at rate :math:`\lambda_0`. The tumour
        growth inhibitory effect of the compound is modelled proportionally to
        its concentration :math:`C` and the current tumour volume. The
        proportionality factor :math:`\kappa` can be interpreted as the potency
        of the drug.

        Note that the critical tumour volume :math:`V_{\text{crit}}` at which
        the growth dynamics transtions from exponential to linear growth is
        given by the two growth rates

        .. math::
            V_{\text{crit}} = \frac{\lambda _1}{2\lambda _0}.

        .. [1] Koch, G. et al. Modeling of tumor growth and anticancer effects
               of combination therapy. J Pharmacokinet Pharmacodyn 36, 179–197
               (2009).
        """
        file_name = 'TGI_Koch_2009.xml'

        return self._path + file_name

    def pk_one_compartment_model(self):
        """
        Returns a one compartment pharmacokinetic model.
        """
        file_name = 'pk_one_comp.xml'

        return self._path + file_name
