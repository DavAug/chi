#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import os

import chi


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
        This model is a combination of a
        :meth:`ModelLibrary.one_compartment_pk_model` and a
        :meth:`tumour_growth_inhibition_model_koch_reparametrised`.
        """
        file_name = 'temporary_full_pkpd_model.xml'

        return chi.PKPDModel(self._path + file_name)

    def one_compartment_pk_model(self, elimination_rate=True):
        r"""
        Returns an instantiation of a 1-compartment PK model.

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

        The drug may be either directly administered to :math:`A` or indirectly
        through a dosing compartment.

        :param bool elimination_rate: If True, the model uses the elimination
            rate :math:`k_e` as a parameter. If False, the model uses the
            clearance :math:`CL` as a parameter, which is related to the
            elimination rate by :math:`CL = k_e V`.

        :rtype: chi.PKPDModel
        """
        file_name = 'pk_one_comp_clearance.xml'
        if elimination_rate:
            file_name = 'pk_one_comp.xml'
        model = chi.PKPDModel(self._path + file_name)
        model.set_outputs(['central.drug_concentration'])

        return model

    def two_compartment_pk_model(self):
        r"""
        Returns an instantiation of a 2-compartment PK model.

        In this model the distribution of the drug is modelled by twp
        compartments: 1. the central compartment; and 2. the peripheral
        compartment. The drug transtions between the two compartments at the
        inter-compartment clearance rate :math:`q` and is eliminated from the
        central compartment at the clearance rate :math:`k_{cl}`.

        .. math ::
            \frac{\text{d}a_c}{\text{d}t} = -k_{cl} c_c - q c_c + q c_p
            \quad \frac{\text{d}a_p}{\text{d}t} = q c_c - q c_p,

        where :math:`a_c` denotes the drug amount in the central compartment
        and :math:`a_p` the drug amount in the peripheral compartment. The
        concentrations :math:`c_c` and :math:`c_p` are given by
        :math:`c_c = \frac{a_c}{v_c}` and :math:`c_p = \frac{a_p}{v_p}`, where
        :math:`v_c` and :math:`v_p` denote the effective volumes of
        distribution of the drug in the compartments. `k_{cl}` and `q` denote
        the clearance and inter-compartment clearance rates, respectively.

        :rtype: chi.PKPDModel
        """
        file_name = 'pk_two_comp_clearance.xml'
        model = chi.PKPDModel(self._path + file_name)
        model.set_outputs(['central.drug_concentration'])

        return model

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

        :rtype: chi.SBMLModel
        """
        file_name = 'tgi_Koch_2009.xml'

        return chi.SBMLModel(self._path + file_name)

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

        :rtype: chi.SBMLModel
        """
        file_name = 'tgi_Koch_2009_reparametrised.xml'

        return chi.SBMLModel(self._path + file_name)
