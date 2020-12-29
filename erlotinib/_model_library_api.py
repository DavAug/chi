#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os


class ModelLibrary(object):
    """
    Contains references to pharmacokinetic and pharmacodynamic models in SBML
    file format.

    These models can be instantiated for simulation or inference with a
    :class:`MechanisticModel`.

    References
    ----------
    .. [1] Koch, G. et al. Modeling of tumor growth and anticancer effects
        of combination therapy. J Pharmacokinet Pharmacodyn 36, 179–197
        (2009).
    """

    def __init__(self):
        # Get path to model library
        self._path = os.path.dirname(os.path.abspath(__file__))
        self._path += '/model_library/'

    def erlotinib_tumour_growth_inhibition_model(self):
        """
        .. warning::
            This model is going to be deprecated soon in favour of a dynamic
            PK model-PD model composition.

        This model is a combination of a
        :meth:`ModelLibrary.one_compartment_pk_model` and a
        :meth:`tumour_growth_inhibition_model_koch_reparametrised`.
        """
        file_name = 'temporary_full_pkpd_model.xml'

        return self._path + file_name

    def one_compartment_pk_model(self):
        r"""
        Returns the absolute path to a SBML file, specifying a one compartment
        pharmacokinetic model.

        In this model the distribution of the drug is modelled by one
        compartment with a linear elimination rate :math:`k_e`

        .. math ::
            \frac{\text{d}A}{\text{d}t} = -k_e A \quad C = \frac{A}{V}.

        Here, :math:`A` and :math:`C` are the amount and the concentration of
        the drug in the body, respectively. :math:`V` is the effective volume
        of distribution of the drug in the compartment.

        This model may be interpreted as modelling the blood plasma
        concentration of the drug, with the assumption that the clearance of
        the drug through the liver may be approximated by an exponential decay
        with the rate :math:`k_e`.

        With a :class:`erlotinib.PharmacokineticModel` the drug may be either
        directly administered to :math:`A` or indirectly through a dosing
        compartment.
        """
        file_name = 'pk_one_comp.xml'

        return self._path + file_name

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
        the growth dynamics transitions from exponential to linear growth is
        given by the two growth rates

        .. math::
            V_{\text{crit}} = \frac{\lambda _1}{2\lambda _0}.
        """
        file_name = 'tgi_Koch_2009.xml'

        return self._path + file_name

    def tumour_growth_inhibition_model_koch_reparametrised(self):
        r"""
        Returns the absolute path to a SBML file, specifying the tumour growth
        inhibition pharmacodynamic model introduced by Koch et al. in [1]_ with
        modified parametrisation.

        In this model the tumour growth inhibition is modelled by an empirical
        model of the tumour volume :math:`V_T` over time

        .. math::
            \frac{\text{d}V_T}{\text{d}t} =
                \frac{\lambda V_T}
                {V_T / V_{\text{crit}} + 1} - \kappa C V_T.


        Here, the tumour growth in absence of the drug is assumed to grow
        exponentially at rate :math:`\lambda` for tumour volumes below some
        critical volume :math:`V_{\text{crit}}`. For volumes beyond
        :math:`V_{\text{crit}}` the growth dynamics is assumed to slow down
        and transition to a linear growth at rate
        :math:`\lambda V_{\text{crit}}`. The tumour growth inhibitory effect
        of the compound is modelled proportionally to its concentration
        :math:`C` and the current tumour volume. The proportionality factor
        :math:`\kappa` can be interpreted as the potency of the drug.

        Note that this parameterisation of the model is related to the original
        parametersation in [1]_ by

        .. math::
            V_{\text{crit}} = \frac{\lambda _1}{2\lambda _0} \quad \text{and}
            \quad \lambda = 2\lambda _1 .
        """
        file_name = 'tgi_Koch_2009_reparametrised.xml'

        return self._path + file_name
