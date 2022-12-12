#
# This file is part of the chi repository
# (https://github.com/DavAug/chi/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pints
import pints.toy

import chi
from chi.library import ModelLibrary


class ToyExponentialModel(chi.MechanisticModel):
    """
    A toy mechanistic model for testing.
    """
    def __init__(self):
        super(ToyExponentialModel, self).__init__()

        self._has_sensitivities = False

    def enable_sensitivities(self, enabled, parameter_names=None):
        r"""
        Enables the computation of the model output sensitivities to the model
        parameters if set to ``True``.

        The sensitivities of the model outputs are defined as the partial
        derviatives of the ouputs :math:`\bar{y}` with respect to the model
        parameters :math:`\psi`

        .. math:
            \frac{\del \bar{y}}{\del \psi}.

        :param enabled: A boolean flag which enables (``True``) / disables
            (``False``) the computation of sensitivities.
        :type enabled: bool
        """
        self._has_sensitivities = bool(enabled)

    def has_sensitivities(self):
        """
        Returns a boolean indicating whether sensitivities have been enabled.
        """
        return self._has_sensitivities

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the number of states.
        """
        return 1

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        return 2

    def outputs(self):
        """
        Returns the output names of the model.
        """
        return ['Count']

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        return ['Initial count', 'Growth rate']

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs (and optionally
        the sensitivites) for the specified parameters and times.

        The model outputs are returned as a 2 dimensional NumPy array of shape
        ``(n_outputs, n_times)``. If sensitivities are enabled, a tuple is
        returned with the NumPy array of the model outputs and a NumPy array of
        the sensitivities of shape ``(n_times, n_outputs, n_parameters)``.

        :param parameters: An array-like object with values for the model
            parameters.
        :type parameters: list, numpy.ndarray
        :param times: An array-like object with time points at which the output
            values are returned.
        :type times: list, numpy.ndarray

        :rtype: np.ndarray of shape (n_outputs, n_times) or
            (n_times, n_outputs, n_parameters)
        """
        y0, growth_rate = parameters
        times = np.asarray(times)

        # Solve model
        y = y0 * np.exp(growth_rate * times)

        if not self._has_sensitivities:
            return y[np.newaxis, :]

        sensitivities = np.empty(shape=(len(times), 1, 2))
        sensitivities[:, 0, 0] = np.exp(growth_rate * times)
        sensitivities[:, 0, 1] = times * y

        return y[np.newaxis, :], sensitivities


class TestHierarchicalLogLikelihood(unittest.TestCase):
    """
    Tests the chi.HierarchicalLogLikelihood class.
    """

    @classmethod
    def setUpClass(cls):
        # Create data
        obs_1 = [1, 1.1, 1.2, 1.3]
        times_1 = [1, 2, 3, 4]
        obs_2 = [2, 2.1, 2.2]
        times_2 = [2, 5, 6]
        cls.observations = [obs_1, obs_2]
        cls.times = [times_1, times_2]

        # Set up mechanistic and error models
        cls.model = ModelLibrary().one_compartment_pk_model()
        cls.model.set_administration('central', direct=False)
        cls.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        cls.error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel()] * 2

        # Create log-likelihoods
        cls.log_likelihoods = [
            chi.LogLikelihood(
                cls.model, cls.error_models, cls.observations, cls.times),
            chi.LogLikelihood(
                cls.model, cls.error_models, cls.observations, cls.times)]

        # Create population models
        cls.population_model = chi.ComposedPopulationModel([
            chi.PooledModel(n_dim=2, dim_names=['Dim. 1', 'Dim. 2']),
            chi.LogNormalModel(dim_names=['Dim. 3'], centered=False),
            chi.PooledModel(dim_names=['Dim. 4']),
            chi.HeterogeneousModel(dim_names=['Dim. 5']),
            chi.PooledModel(
                n_dim=4, dim_names=['Dim. 6', 'Dim. 7', 'Dim. 8', 'Dim. 9'])
            ])

        # Test case I: simple population model
        cls.hierarchical_model = chi.HierarchicalLogLikelihood(
            cls.log_likelihoods, cls.population_model)

        # Test case II: Covariate population model
        cpop_model2 = chi.CovariatePopulationModel(
            chi.GaussianModel(centered=False),
            chi.LinearCovariateModel(n_cov=2))
        population_model = chi.ComposedPopulationModel([
            chi.PooledModel(dim_names=['Dim. 1']),
            chi.GaussianModel(),
            chi.PooledModel(n_dim=6),
            cpop_model2
        ])
        covariates = np.array([[1, 2], [3, 4]])
        cls.hierarchical_model3 = chi.HierarchicalLogLikelihood(
            cls.log_likelihoods, population_model, covariates)

        # Test reduced population model inside compose pop model
        reduced_pop1 = chi.ReducedPopulationModel(
            chi.PooledModel(dim_names=['Dim. 1']))
        reduced_pop1.fix_parameters({'Pooled Dim. 1': 10})
        reduced_pop2 = chi.ReducedPopulationModel(chi.GaussianModel())
        reduced_pop2.fix_parameters({'Mean Dim. 1': 0.2})
        population_model = chi.ComposedPopulationModel([
            reduced_pop1,
            reduced_pop2,
            chi.PooledModel(n_dim=6),
            cpop_model2
        ])
        covariates = np.array([[1, 2], [3, 4]])
        cls.hierarchical_model4 = chi.HierarchicalLogLikelihood(
            cls.log_likelihoods, population_model, covariates)

        # Test outer reduced pop model
        population_model = chi.ComposedPopulationModel([
            chi.PooledModel(dim_names=['Dim. 1']),
            chi.GaussianModel(),
            chi.PooledModel(n_dim=6),
            cpop_model2
        ])
        population_model = chi.ReducedPopulationModel(population_model)
        population_model.fix_parameters({
            'Pooled Dim. 1': 10,
            'Pooled Dim. 5': 1,
            'Mean Dim. 9 Cov. 1': 1
        })
        covariates = np.array([[1, 2], [3, 4]])
        cls.hierarchical_model5 = chi.HierarchicalLogLikelihood(
            cls.log_likelihoods, population_model, covariates)

    def test_bad_instantiation(self):
        # Log-likelihoods are not pints.LogPDF
        log_likelihoods = ['bad', 'type']
        with self.assertRaisesRegex(ValueError, 'The log-likelihoods have'):
            chi.HierarchicalLogLikelihood(
                log_likelihoods, self.population_model)

        # Log-likelihoods are defined on different parameter spaces
        model = ModelLibrary().one_compartment_pk_model()
        model.set_administration('central', direct=False)
        error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel()]
        log_likelihoods = [
            self.log_likelihoods[0],
            chi.LogLikelihood(
                model, error_models, self.observations[0], self.times[0])]

        with self.assertRaisesRegex(ValueError, 'The dimension of the pop'):
            chi.HierarchicalLogLikelihood(
                log_likelihoods, self.population_model)

        # Population models are not chi.PopulationModel
        population_model = 'bad type'
        with self.assertRaisesRegex(ValueError, 'The population model has to'):
            chi.HierarchicalLogLikelihood(
                self.log_likelihoods, population_model)

        # No covariates have been passed for covariate dependent population
        # models
        population_model = chi.ComposedPopulationModel([
            chi.CovariatePopulationModel(
                chi.GaussianModel(),
                chi.LinearCovariateModel(n_cov=2)
            )
        ] * 9)
        with self.assertRaisesRegex(ValueError, 'The population model needs'):
            chi.HierarchicalLogLikelihood(
                self.log_likelihoods, population_model)

        # Covariates do not have shape (n, c)
        covariates = np.empty(shape=(3, 2))
        with self.assertRaisesRegex(ValueError, 'The covariates do not have'):
            chi.HierarchicalLogLikelihood(
                self.log_likelihoods, population_model, covariates)
        covariates = np.empty(shape=(2,))
        with self.assertRaisesRegex(ValueError, 'The covariates do not have'):
            chi.HierarchicalLogLikelihood(
                self.log_likelihoods, population_model, covariates)

        # Likelihoods have the same labels
        ll_1 = chi.LogLikelihood(
            self.model, self.error_models, self.observations, self.times)
        ll_1.set_id('some ID')
        with self.assertRaisesRegex(ValueError, 'Log-likelihood IDs need'):
            chi.HierarchicalLogLikelihood([ll_1, ll_1], self.population_model)

    def test_call(self):
        # Test case I: All parameters pooled
        model = chi.HierarchicalLogLikelihood(
            self.log_likelihoods, chi.PooledModel(n_dim=9))

        # Test case I.1
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        score = 0
        for ll in self.log_likelihoods:
            score += ll(parameters)

        self.assertEqual(model(parameters), score)

        # Test case I.2
        parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        score = 0
        for ll in self.log_likelihoods:
            score += ll(parameters)

        self.assertEqual(model(parameters), score)

        # Test case II.1: Heterogeneous model
        likelihood = chi.HierarchicalLogLikelihood(
            self.log_likelihoods,
            chi.HeterogeneousModel(n_dim=9))

        # Compute score from individual likelihoods
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        score = 0
        for ll in self.log_likelihoods:
            score += ll(parameters)

        n_parameters = 9
        n_ids = 2
        parameters = [1] * n_parameters * n_ids
        self.assertEqual(likelihood(parameters), score)

        # Test case II.2
        # Compute score from individual likelihoods
        parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        score = 0
        for ll in self.log_likelihoods:
            score += ll(parameters)

        parameters = parameters * n_ids
        self.assertEqual(likelihood(parameters), score)

        # Test case III.1: Non-trivial population model
        # Reminder of population model
        # cls.population_models = [
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.LogNormalModel(),
        #     chi.PooledModel(),
        #     chi.HeterogeneousModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel()]

        # Create reference pop model
        ref_pop_model = chi.LogNormalModel(centered=False)
        indiv_parameters_1 = [10, 1, 0.1, 1, 3, 1, 1, 2, 1.2]
        indiv_parameters_2 = [10, 1, 0.2, 1, 2, 1, 1, 2, 1.2]
        pop_params = [10, 1, 0.2, 1, 1, 3, 2, 1, 1, 2, 1.2]

        parameters = [0.1, 0.2] + pop_params

        # Transform the bottom parameters
        indiv_parameters_1[2] = np.exp(0.2 + 0.1)
        indiv_parameters_2[2] = np.exp(0.2 + 0.2)

        score = \
            ref_pop_model.compute_log_likelihood(
                parameters=pop_params[2:4],
                observations=[0.1, 0.2]) + \
            self.log_likelihoods[0](indiv_parameters_1) + \
            self.log_likelihoods[1](indiv_parameters_2)

        self.assertNotEqual(score, -np.inf)
        self.assertAlmostEqual(self.hierarchical_model(parameters), score)

        # Test case IV: Infinite log-pdf from population model
        # Reminder of population model
        # cls.population_models = [
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.LogNormalModel(),
        #     chi.PooledModel(),
        #     chi.HeterogeneousModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel()]

        indiv_parameters_1 = [10, 1, 0, 1, 3, 1, 1, 2, 1.2]
        indiv_parameters_2 = [10, 1, 0, 1, 2, 1, 1, 2, 1.2]
        pop_params = [10, 1, 0.2, -1, 1, 3, 2, 1, 1, 2, 1.2]

        parameters = \
            indiv_parameters_1[2:3] + \
            indiv_parameters_2[2:3] + \
            pop_params

        self.assertEqual(self.hierarchical_model(parameters), -np.inf)

        # Test case VI.1: Covariate population model
        # Reminder of population model
        # population_models = [
        #     chi.PooledModel(),
        #     chi.GaussianModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     cpop_model2]

        # Create reference pop model
        covariates = np.array([[1, 2], [3, 4]])
        ref_pop_model1 = chi.GaussianModel()
        ref_pop_model2 = chi.CovariatePopulationModel(
            chi.GaussianModel(centered=False),
            chi.LinearCovariateModel(n_cov=2))
        psis_1 = np.array([0.1, 0.2])
        etas_2 = np.array([1, 10])
        pop_params_1 = [0.2, 1]
        pop_params_2 = [1, 0.1, 1, 2, 3, 4]
        mu = pop_params_2[0] + covariates @ np.array(pop_params_2)[2:4]
        sigma = pop_params_2[1] + covariates @ np.array(pop_params_2)[4:]
        psis_2 = mu + sigma * etas_2
        pooled_params = [10, 1, 1, 1, 1, 2, 1.2]
        indiv_parameters_1 = [
            pooled_params[0],
            psis_1[0],
            pooled_params[1],
            pooled_params[2],
            pooled_params[3],
            pooled_params[4],
            pooled_params[5],
            pooled_params[6],
            psis_2[0]]
        indiv_parameters_2 = [
            pooled_params[0],
            psis_1[1],
            pooled_params[1],
            pooled_params[2],
            pooled_params[3],
            pooled_params[4],
            pooled_params[5],
            pooled_params[6],
            psis_2[1]]

        parameters = [
            psis_1[0],
            etas_2[0],
            psis_1[1],
            etas_2[1],
            pooled_params[0]
            ] + pop_params_1 + pooled_params[1:] + pop_params_2

        ref_score_3 = self.log_likelihoods[0](
            indiv_parameters_1)
        ref_score_4 = self.log_likelihoods[1](
            indiv_parameters_2)
        ref_score_1 = ref_pop_model1.compute_log_likelihood(
            parameters=pop_params_1, observations=psis_1)
        ref_score_2 = ref_pop_model2.compute_log_likelihood(
            parameters=pop_params_2, observations=etas_2,
            covariates=covariates)
        ref_score = ref_score_1 + ref_score_2 + ref_score_3 + ref_score_4

        score = self.hierarchical_model3(parameters)

        self.assertNotEqual(ref_score, -np.inf)
        self.assertAlmostEqual(score, ref_score)

        # Test that fixed likelihoods return same scores as unfixed likelihoods
        psis_1 = np.array([0.1, 0.2])
        etas_2 = np.array([1, 10])
        pop_params_1 = [0.2, 1]
        pop_params_2 = [1, 0.1, 1, 2, 3, 4]
        mu = pop_params_2[0] + covariates @ np.array(pop_params_2)[2:4]
        sigma = pop_params_2[1] + covariates @ np.array(pop_params_2)[4:]
        psis_2 = mu + sigma * etas_2
        pooled_params = [10, 1, 1, 1, 1, 2, 1.2]
        parameters = [
            psis_1[0],
            etas_2[0],
            psis_1[1],
            etas_2[1],
            pooled_params[0]
            ] + pop_params_1 + pooled_params[1:] + pop_params_2

        ref_score = self.hierarchical_model3(parameters)
        p = parameters[:4] + parameters[6:]
        score = self.hierarchical_model4(p)
        self.assertEqual(score, ref_score)

        p = parameters[:4] + parameters[5:9] + parameters[10:15] \
            + parameters[16:]
        score = self.hierarchical_model5(p)
        self.assertEqual(score, ref_score)

    def test_compute_pointwise_ll(self):
        # TODO:
        with self.assertRaisesRegex(NotImplementedError, None):
            self.hierarchical_model.compute_pointwise_ll('some input')

        # # Test case I: All parameters pooled
        # likelihood = chi.HierarchicalLogLikelihood(
        #     log_likelihoods=self.log_likelihoods,
        #     population_models=[chi.PooledModel()] * 9)

        # # Test case I.1
        # parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        # score = likelihood(parameters)
        # indiv_scores = likelihood.compute_pointwise_ll(
        #     parameters, per_individual=True)
        # pw_scores = likelihood.compute_pointwise_ll(
        #     parameters, per_individual=False)

        # self.assertEqual(len(indiv_scores), 2)
        # self.assertAlmostEqual(np.sum(indiv_scores), score)
        # self.assertEqual(len(pw_scores), 14)
        # self.assertAlmostEqual(np.sum(pw_scores), score)

        # # Test case I.2
        # parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        # score = likelihood(parameters)
        # indiv_scores = likelihood.compute_pointwise_ll(
        #     parameters, per_individual=True)
        # pw_scores = likelihood.compute_pointwise_ll(
        #     parameters, per_individual=False)

        # self.assertEqual(len(indiv_scores), 2)
        # self.assertAlmostEqual(np.sum(indiv_scores), score)
        # self.assertEqual(len(pw_scores), 14)
        # self.assertAlmostEqual(np.sum(pw_scores), score)

        # # Test case II.1: Heterogeneous model
        # likelihood = chi.HierarchicalLogLikelihood(
        #     log_likelihoods=self.log_likelihoods,
        #     population_models=[
        #         chi.HeterogeneousModel()] * 9)

        # # Compute score from individual likelihoods
        # n_ids = 2
        # n_parameters = 9
        # parameters = [1] * n_parameters * n_ids
        # score = likelihood(parameters)
        # indiv_scores = likelihood.compute_pointwise_ll(
        #     parameters, per_individual=True)
        # pw_scores = likelihood.compute_pointwise_ll(
        #     parameters, per_individual=False)

        # self.assertEqual(len(indiv_scores), 2)
        # self.assertAlmostEqual(np.sum(indiv_scores), score)
        # self.assertEqual(len(pw_scores), 14)
        # self.assertAlmostEqual(np.sum(pw_scores), score)

        # # Test case II.2
        # # Compute score from individual likelihoods
        # parameters = \
        #     [parameters[0]] * n_ids + \
        #     [parameters[1]] * n_ids + \
        #     [parameters[2]] * n_ids + \
        #     [parameters[3]] * n_ids + \
        #     [parameters[4]] * n_ids + \
        #     [parameters[5]] * n_ids + \
        #     [parameters[6]] * n_ids + \
        #     [parameters[7]] * n_ids + \
        #     [parameters[8]] * n_ids
        # score = likelihood(parameters)
        # indiv_scores = likelihood.compute_pointwise_ll(
        #     parameters, per_individual=True)
        # pw_scores = likelihood.compute_pointwise_ll(
        #     parameters, per_individual=False)

        # self.assertEqual(len(indiv_scores), 2)
        # self.assertAlmostEqual(np.sum(indiv_scores), score)
        # self.assertEqual(len(pw_scores), 14)
        # self.assertAlmostEqual(np.sum(pw_scores), score)

        # # Test case III.1: Non-trivial population model
        # # Reminder of population model
        # # cls.population_models = [
        # #     chi.PooledModel(),
        # #     chi.PooledModel(),
        # #     chi.LogNormalModel(),
        # #     chi.PooledModel(),
        # #     chi.HeterogeneousModel(),
        # #     chi.PooledModel(),
        # #     chi.PooledModel(),
        # #     chi.PooledModel(),
        # #     chi.PooledModel()]

        # # Create reference pop model
        # indiv_parameters_1 = [10, 1, 0.1, 1, 3, 1, 1, 2, 1.2]
        # indiv_parameters_2 = [10, 1, 0.2, 1, 2, 1, 1, 2, 1.2]
        # pop_params = [0.2, 1]
        # parameters = [
        #     indiv_parameters_1[0],
        #     indiv_parameters_1[1],
        #     indiv_parameters_1[2],
        #     indiv_parameters_2[2],
        #     pop_params[0],
        #     pop_params[1],
        #     indiv_parameters_1[3],
        #     indiv_parameters_1[4],
        #     indiv_parameters_2[4],
        #     indiv_parameters_1[5],
        #     indiv_parameters_1[6],
        #     indiv_parameters_1[7],
        #     indiv_parameters_1[8]]

        # score = self.hierarchical_model(parameters)
        # indiv_scores = self.hierarchical_model.compute_pointwise_ll(
        #     parameters, per_individual=True)
        # pw_scores = self.hierarchical_model.compute_pointwise_ll(
        #     parameters, per_individual=False)

        # self.assertEqual(len(indiv_scores), 2)
        # self.assertAlmostEqual(np.sum(indiv_scores), score)
        # self.assertEqual(len(pw_scores), 14)
        # self.assertAlmostEqual(np.sum(pw_scores), score)

        # # Test case V: CovariatePopulationModel
        # with self.assertRaisesRegex(
        #   NotImplementedError, 'This method is not'):
        #     self.hierarchical_model3.compute_pointwise_ll('some params')

    def test_evaluateS1(self):
        # Test case I: All parameters pooled
        model = chi.HierarchicalLogLikelihood(
            self.log_likelihoods, chi.PooledModel(n_dim=9))

        # Test case I.1
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ref_score, ref_sens = 0, np.zeros(shape=len(parameters))
        for ll in self.log_likelihoods:
            s, ss = ll.evaluateS1(parameters)
            ref_score += s
            ref_sens += ss
        # ref_score, ref_sens = pooled_log_pdf.evaluateS1(parameters)
        score, sens = model.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 9)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])

        # Test case I.2
        parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        ref_score, ref_sens = 0, np.zeros(shape=len(parameters))
        for ll in self.log_likelihoods:
            s, ss = ll.evaluateS1(parameters)
            ref_score += s
            ref_sens += ss
        score, sens = model.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 9)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])

        # Test case II.1: Heterogeneous model
        likelihood = chi.HierarchicalLogLikelihood(
            self.log_likelihoods, chi.HeterogeneousModel(n_dim=9))

        # Compute score from individual likelihoods
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ref_score = 0
        ref_senss = []
        for ll in self.log_likelihoods:
            s, se = ll.evaluateS1(parameters)
            ref_score += s
            ref_senss.append(se)

        n_parameters = 9
        n_ids = 2
        parameters = [1] * n_parameters * n_ids
        score, sens = likelihood.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 18)
        ref_sens = ref_senss[0]
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])
        ref_sens = ref_senss[1]
        self.assertEqual(sens[9], ref_sens[0])
        self.assertEqual(sens[10], ref_sens[1])
        self.assertEqual(sens[11], ref_sens[2])
        self.assertEqual(sens[12], ref_sens[3])
        self.assertEqual(sens[13], ref_sens[4])
        self.assertEqual(sens[14], ref_sens[5])
        self.assertEqual(sens[15], ref_sens[6])
        self.assertEqual(sens[16], ref_sens[7])
        self.assertEqual(sens[17], ref_sens[8])

        # Test case II.2
        # Compute score from individual likelihoods
        parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        ref_score = 0
        ref_senss = []
        for ll in self.log_likelihoods:
            s, se = ll.evaluateS1(parameters)
            ref_score += s
            ref_senss.append(se)

        n_ids = 2
        parameters = parameters + parameters
        score, sens = likelihood.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 18)
        ref_sens = ref_senss[0]
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])
        ref_sens = ref_senss[1]
        self.assertEqual(sens[9], ref_sens[0])
        self.assertEqual(sens[10], ref_sens[1])
        self.assertEqual(sens[11], ref_sens[2])
        self.assertEqual(sens[12], ref_sens[3])
        self.assertEqual(sens[13], ref_sens[4])
        self.assertEqual(sens[14], ref_sens[5])
        self.assertEqual(sens[15], ref_sens[6])
        self.assertEqual(sens[16], ref_sens[7])
        self.assertEqual(sens[17], ref_sens[8])

        # Test case III.1: Non-trivial population model
        # Reminder of population model
        # cls.population_models = [
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.LogNormalModel(),
        #     chi.PooledModel(),
        #     chi.HeterogeneousModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel()]

        # Create reference pop model
        ref_pop_model = chi.LogNormalModel(centered=False)
        indiv_parameters_1 = [10, 1, 0.1, 1, 3, 1, 1, 2, 1.2]
        indiv_parameters_2 = [10, 1, 0.2, 1, 2, 1, 1, 2, 1.2]
        pop_params = [10, 1, 0.2, 1, 1, 3, 2, 1, 1, 2, 1.2]

        parameters = [0.1, 0.2] + pop_params

        indiv_parameters_1[2] = np.exp(0.2 + 0.1)
        indiv_parameters_2[2] = np.exp(0.2 + 0.2)

        ref_s1, dpsi1, dtheta1 = ref_pop_model.compute_sensitivities(
                parameters=pop_params[2:4],
                observations=[0.1, 0.2])
        ref_s2, ref_sens2 = self.log_likelihoods[0].evaluateS1(
            indiv_parameters_1)
        ref_s3, ref_sens3 = self.log_likelihoods[1].evaluateS1(
            indiv_parameters_2)

        ref_score = ref_s1 + ref_s2 + ref_s3
        ref_sens = [
            dpsi1[0, 0] + ref_sens2[2],
            dpsi1[1, 0] + ref_sens3[2],
            ref_sens2[0] + ref_sens3[0],
            ref_sens2[1] + ref_sens3[1],
            dtheta1[0],
            dtheta1[1],
            ref_sens2[3] + ref_sens3[3],
            ref_sens2[4],
            ref_sens3[4],
            ref_sens2[5] + ref_sens3[5],
            ref_sens2[6] + ref_sens3[6],
            ref_sens2[7] + ref_sens3[7],
            ref_sens2[8] + ref_sens3[8]]

        # Compute score and sensitivities with hierarchical model
        score, sens = self.hierarchical_model.evaluateS1(parameters)

        self.assertNotEqual(score, -np.inf)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 13)
        self.assertAlmostEqual(sens[0], ref_sens[0])
        self.assertAlmostEqual(sens[1], ref_sens[1])
        self.assertAlmostEqual(sens[2], ref_sens[2])
        self.assertAlmostEqual(sens[3], ref_sens[3])
        self.assertAlmostEqual(sens[4], ref_sens[4])
        self.assertAlmostEqual(sens[5], ref_sens[5])
        self.assertAlmostEqual(sens[6], ref_sens[6])
        self.assertAlmostEqual(sens[7], ref_sens[7])
        self.assertAlmostEqual(sens[8], ref_sens[8])
        self.assertAlmostEqual(sens[9], ref_sens[9])
        self.assertAlmostEqual(sens[10], ref_sens[10])
        self.assertAlmostEqual(sens[11], ref_sens[11])
        self.assertAlmostEqual(sens[12], ref_sens[12])

        # Use a model that neither used pooled not heterogen. models
        model = chi.HierarchicalLogLikelihood(
            self.log_likelihoods, chi.LogNormalModel(n_dim=9))

        # Compute score and sensitivities with hierarchical model
        score, sens = model.evaluateS1(np.ones(model.n_parameters()))

        # Test case V: Infinite log-pdf from population model
        # Reminder of population model
        # cls.population_models = [
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.LogNormalModel(),
        #     chi.PooledModel(),
        #     chi.HeterogeneousModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel()]

        indiv_parameters_1 = [10, 1, -1, 1, 3, 1, 1, 2, 3]
        indiv_parameters_2 = [10, 1, 0, 1, 2, 1, 1, 2, 3]
        pop_params = [10, 1, 0.2, -1, 1, 3, 2, 1, 1, 2, 3]

        parameters = [
            indiv_parameters_1[2],
            indiv_parameters_2[2],
            ] + pop_params
        score, sens = self.hierarchical_model.evaluateS1(parameters)
        self.assertEqual(score, -np.inf)

        # Test case VI.1: Covariate population model
        # Reminder of population model
        # population_models = [
        #     chi.PooledModel(),
        #     chi.GaussianModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     chi.PooledModel(),
        #     cpop_model2]

        # Create reference pop model
        covariates = np.array([[1, 2], [3, 4]])
        ref_pop_model1 = chi.GaussianModel()
        ref_pop_model2 = chi.CovariatePopulationModel(
            chi.GaussianModel(centered=False),
            chi.LinearCovariateModel(n_cov=2))
        psis_1 = np.array([0.1, 0.2])
        etas_2 = np.array([1, 10])
        pop_params_1 = [0.2, 1]
        pop_params_2 = [1, 0.1, 1, 2, 3, 4]
        mu = pop_params_2[0] + covariates @ np.array(pop_params_2)[2:4]
        sigma = pop_params_2[1] + covariates @ np.array(pop_params_2)[4:]
        psis_2 = mu + sigma * etas_2
        pooled_params = [10, 1, 1, 1, 1, 2, 1.2]
        indiv_parameters_1 = [
            pooled_params[0],
            psis_1[0],
            pooled_params[1],
            pooled_params[2],
            pooled_params[3],
            pooled_params[4],
            pooled_params[5],
            pooled_params[6],
            psis_2[0]]
        indiv_parameters_2 = [
            pooled_params[0],
            psis_1[1],
            pooled_params[1],
            pooled_params[2],
            pooled_params[3],
            pooled_params[4],
            pooled_params[5],
            pooled_params[6],
            psis_2[1]]

        parameters = [
            psis_1[0],
            etas_2[0],
            psis_1[1],
            etas_2[1],
            pooled_params[0]
            ] + pop_params_1 + pooled_params[1:] + pop_params_2

        ref_score_3, ref_sens_3 = self.log_likelihoods[0].evaluateS1(
            indiv_parameters_1)
        ref_score_4, ref_sens_4 = self.log_likelihoods[1].evaluateS1(
            indiv_parameters_2)
        dlog_dpsi = np.empty((2, 1))
        dlog_dpsi[0, 0] = ref_sens_3[1]
        dlog_dpsi[1, 0] = ref_sens_4[1]
        ref_score_1, dpsi1, dtheta1 = ref_pop_model1.compute_sensitivities(
            parameters=pop_params_1, observations=psis_1, dlogp_dpsi=dlog_dpsi)
        dlog_dpsi[0, 0] = ref_sens_3[8]
        dlog_dpsi[1, 0] = ref_sens_4[8]
        ref_score_2, dpsi2, dtheta2 = ref_pop_model2.compute_sensitivities(
            parameters=pop_params_2, observations=etas_2, dlogp_dpsi=dlog_dpsi,
            covariates=covariates)
        ref_score = ref_score_1 + ref_score_2 + ref_score_3 + ref_score_4

        ref_sens = [
            dpsi1[0, 0],
            dpsi2[0, 0],
            dpsi1[1, 0],
            dpsi2[1, 0],
            ref_sens_3[0] + ref_sens_4[0],
            dtheta1[0],
            dtheta1[1],
            ref_sens_3[2] + ref_sens_4[2],
            ref_sens_3[3] + ref_sens_4[3],
            ref_sens_3[4] + ref_sens_4[4],
            ref_sens_3[5] + ref_sens_4[5],
            ref_sens_3[6] + ref_sens_4[6],
            ref_sens_3[7] + ref_sens_4[7],
            dtheta2[0],
            dtheta2[1],
            dtheta2[2],
            dtheta2[3],
            dtheta2[4],
            dtheta2[5]]

        score, sens = self.hierarchical_model3.evaluateS1(parameters)

        self.assertNotEqual(ref_score, -np.inf)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 19)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])
        self.assertEqual(sens[9], ref_sens[9])
        self.assertEqual(sens[10], ref_sens[10])
        self.assertEqual(sens[11], ref_sens[11])
        self.assertEqual(sens[12], ref_sens[12])
        self.assertEqual(sens[13], ref_sens[13])
        self.assertEqual(sens[14], ref_sens[14])
        self.assertEqual(sens[15], ref_sens[15])
        self.assertEqual(sens[16], ref_sens[16])
        self.assertEqual(sens[17], ref_sens[17])
        self.assertEqual(sens[18], ref_sens[18])

        # Test that fixed likelihoods return same scores as unfixed likelihoods
        ref_score, ref_sens = score, sens
        p = parameters[:4] + parameters[6:]
        score, sens = self.hierarchical_model4.evaluateS1(p)
        self.assertNotEqual(ref_score, -np.inf)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 17)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[6])
        self.assertEqual(sens[5], ref_sens[7])
        self.assertEqual(sens[6], ref_sens[8])
        self.assertEqual(sens[7], ref_sens[9])
        self.assertEqual(sens[8], ref_sens[10])
        self.assertEqual(sens[9], ref_sens[11])
        self.assertEqual(sens[10], ref_sens[12])
        self.assertEqual(sens[11], ref_sens[13])
        self.assertEqual(sens[12], ref_sens[14])
        self.assertEqual(sens[13], ref_sens[15])
        self.assertEqual(sens[14], ref_sens[16])
        self.assertEqual(sens[15], ref_sens[17])
        self.assertEqual(sens[16], ref_sens[18])

        p = parameters[:4] + parameters[5:9] + parameters[10:15] \
            + parameters[16:]
        score, sens = self.hierarchical_model5.evaluateS1(p)
        self.assertNotEqual(ref_score, -np.inf)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 16)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[5])
        self.assertEqual(sens[5], ref_sens[6])
        self.assertEqual(sens[6], ref_sens[7])
        self.assertEqual(sens[7], ref_sens[8])
        self.assertEqual(sens[8], ref_sens[10])
        self.assertEqual(sens[9], ref_sens[11])
        self.assertEqual(sens[10], ref_sens[12])
        self.assertEqual(sens[11], ref_sens[13])
        self.assertEqual(sens[12], ref_sens[14])
        self.assertEqual(sens[13], ref_sens[16])
        self.assertEqual(sens[14], ref_sens[17])
        self.assertEqual(sens[15], ref_sens[18])

    def test_get_id(self):
        # Test case I: Get parameter IDs
        ids = self.hierarchical_model.get_id()

        self.assertEqual(len(ids), 13)
        self.assertEqual(ids[0], 'Log-likelihood 1')
        self.assertEqual(ids[1], 'Log-likelihood 2')
        self.assertIsNone(ids[2])
        self.assertIsNone(ids[3])
        self.assertIsNone(ids[4])
        self.assertIsNone(ids[5])
        self.assertIsNone(ids[6])
        self.assertIsNone(ids[7])
        self.assertIsNone(ids[8])
        self.assertIsNone(ids[9])
        self.assertIsNone(ids[10])
        self.assertIsNone(ids[11])
        self.assertIsNone(ids[12])

        # Test case II: Get individual IDs
        ids = self.hierarchical_model.get_id(unique=True)

        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 'Log-likelihood 1')
        self.assertEqual(ids[1], 'Log-likelihood 2')

        # Test case III: Get IDs for fully pooled model
        # Create population models
        population_model = chi.PooledModel(n_dim=9)
        hierarchical_model = chi.HierarchicalLogLikelihood(
            self.log_likelihoods, population_model)
        ids = hierarchical_model.get_id(unique=True)

        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], 'Log-likelihood 1')
        self.assertEqual(ids[1], 'Log-likelihood 2')

        # Test case IV: covariate model
        ids = self.hierarchical_model3.get_id(unique=True)
        self.assertEqual(ids[0], 'Log-likelihood 1')
        self.assertEqual(ids[1], 'Log-likelihood 2')

    def test_get_parameter_names(self):
        # Test case I: without ids
        parameter_names = self.hierarchical_model.get_parameter_names()

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'central.size')
        self.assertEqual(parameter_names[1], 'central.size')
        self.assertEqual(parameter_names[2], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[3], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[4], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[5], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[6], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[7], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[8], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[11], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[12], 'Pooled Dim. 9')

        # Test case II: Exclude bottom-level
        parameter_names = self.hierarchical_model.get_parameter_names(
            exclude_bottom_level=True)

        self.assertEqual(len(parameter_names), 11)
        self.assertEqual(parameter_names[0], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[1], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[2], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[3], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[4], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[5], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[6], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[7], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[8], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 9')

        # Test case III: with ids
        parameter_names = self.hierarchical_model.get_parameter_names(
            include_ids=True)

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'Log-likelihood 1 central.size')
        self.assertEqual(parameter_names[1], 'Log-likelihood 2 central.size')
        self.assertEqual(parameter_names[2], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[3], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[4], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[5], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[6], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[7], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[8], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[11], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[12], 'Pooled Dim. 9')

        # Test case IV: Exclude bottom-level with IDs
        parameter_names = self.hierarchical_model.get_parameter_names(
            exclude_bottom_level=True, include_ids=True)

        self.assertEqual(parameter_names[0], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[1], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[2], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[3], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[4], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[5], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[6], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[7], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[8], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 9')

        # Test case V: with covariate model
        parameter_names = self.hierarchical_model3.get_parameter_names(
            include_ids=True)
        self.assertEqual(len(parameter_names), 19)
        self.assertEqual(
            parameter_names[0], 'Log-likelihood 1 dose.drug_amount')
        self.assertEqual(
            parameter_names[1],
            'Log-likelihood 1 dose.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[2], 'Log-likelihood 2 dose.drug_amount')
        self.assertEqual(
            parameter_names[3],
            'Log-likelihood 2 dose.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[4], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[5], 'Mean Dim. 2')
        self.assertEqual(parameter_names[6], 'Std. Dim. 2')
        self.assertEqual(parameter_names[7], 'Pooled Dim. 3')
        self.assertEqual(parameter_names[8], 'Pooled Dim. 4')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 5')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[11], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[12], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[13], 'Mean Dim. 9')
        self.assertEqual(
            parameter_names[14], 'Std. Dim. 9')
        self.assertEqual(
            parameter_names[15],
            'Mean Dim. 9 Cov. 1')
        self.assertEqual(
            parameter_names[16],
            'Mean Dim. 9 Cov. 2')

    def test_get_population_models(self):
        pop_model = self.hierarchical_model.get_population_model()
        self.assertIsInstance(pop_model, chi.PopulationModel)

    def test_n_log_likelihoods(self):
        n_ids = self.hierarchical_model.n_log_likelihoods()
        self.assertEqual(n_ids, 2)

    def test_n_parameters(self):
        # Test case I: All parameters
        # 9 individual parameters, from which 1 is modelled heterogeneously,
        # 1 log-normally and the rest is pooled
        # And there are 2 individuals
        n_parameters = 2 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.assertEqual(
            self.hierarchical_model.n_parameters(), n_parameters)

        # Test case II: Exclude bottom parameters
        n_parameters = self.hierarchical_model.n_parameters(
            exclude_bottom_level=True)
        self.assertEqual(n_parameters, 11)

    def test_n_observations(self):
        n_obs = self.hierarchical_model.n_observations()

        self.assertEqual(len(n_obs), 2)
        self.assertEqual(n_obs[0], 7)
        self.assertEqual(n_obs[1], 7)


class TestHierarchicalLogPosterior(unittest.TestCase):
    """
    Tests the chi.HierarchicalLogPosterior class.
    """

    @classmethod
    def setUpClass(cls):
        # Create data
        obs_1 = [1, 1.1, 1.2, 1.3]
        times_1 = [1, 2, 3, 4]
        obs_2 = [2, 2.1, 2.2]
        times_2 = [2, 5, 6]
        observations = [obs_1, obs_2]
        times = [times_1, times_2]

        # Set up mechanistic and error models
        model = ModelLibrary().one_compartment_pk_model()
        model.set_administration('central', direct=False)
        model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel()] * 2

        # Create log-likelihoods
        log_likelihoods = [
            chi.LogLikelihood(
                model, error_models, observations, times),
            chi.LogLikelihood(
                model, error_models, observations, times)]

        # Create population models
        population_model = chi.ComposedPopulationModel([
            chi.PooledModel(n_dim=2),
            chi.LogNormalModel(),
            chi.PooledModel(),
            chi.HeterogeneousModel(),
            chi.PooledModel(n_dim=4)])

        # Create hierarchical log-likelihood
        cls.hierarch_log_likelihood = chi.HierarchicalLogLikelihood(
            log_likelihoods, population_model)

        # Define log-prior
        cls.log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1))

        # Create log-posterior
        cls.log_posterior = chi.HierarchicalLogPosterior(
            cls.hierarch_log_likelihood,
            cls.log_prior)

    def test_bad_instantiation(self):
        # Log-likelihood has bad type
        log_likelihood = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The log-likelihood has'):
            chi.HierarchicalLogPosterior(
                log_likelihood, self.log_prior)

        # Log-prior has bad type
        log_prior = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The log-prior has to be'):
            chi.HierarchicalLogPosterior(
                self.hierarch_log_likelihood, log_prior)

        # The dimension of the log-prior does not match number of top-level
        # parameters
        log_prior = pints.LogNormalLogPrior(0, 1)
        with self.assertRaisesRegex(ValueError, 'The log-prior has to have'):
            chi.HierarchicalLogPosterior(
                self.hierarch_log_likelihood, log_prior)

    def test_call(self):
        # Test case I: Check score contributions add appropriately
        all_params = np.arange(start=1, stop=14, step=1)
        top_params = all_params[2:]
        ref_score = self.hierarch_log_likelihood(all_params) + \
            self.log_prior(top_params)
        score = self.log_posterior(all_params)

        self.assertNotEqual(score, -np.inf)
        self.assertEqual(score, ref_score)

        # Test case II: Check exception for inf prior score
        parameters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1]
        self.assertEqual(self.log_posterior(parameters), -np.inf)

    def test_evaluateS1(self):
        # Test case I: Check score contributions add appropriately
        all_params = np.arange(start=1, stop=14, step=1)
        top_params = all_params[2:]
        ref_score1, ref_sens1 = self.hierarch_log_likelihood.evaluateS1(
            all_params)
        ref_score2, ref_sens2 = self.log_prior.evaluateS1(top_params)
        ref_score = ref_score1 + ref_score2
        ref_sens = [
            ref_sens1[0],
            ref_sens1[1],
            ref_sens1[2] + ref_sens2[0],
            ref_sens1[3] + ref_sens2[1],
            ref_sens1[4] + ref_sens2[2],
            ref_sens1[5] + ref_sens2[3],
            ref_sens1[6] + ref_sens2[4],
            ref_sens1[7] + ref_sens2[5],
            ref_sens1[8] + ref_sens2[6],
            ref_sens1[9] + ref_sens2[7],
            ref_sens1[10] + ref_sens2[8],
            ref_sens1[11] + ref_sens2[9],
            ref_sens1[12] + ref_sens2[10]]

        score, sens = self.log_posterior.evaluateS1(all_params)

        self.assertNotEqual(score, -np.inf)
        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 13)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])
        self.assertEqual(sens[9], ref_sens[9])
        self.assertEqual(sens[10], ref_sens[10])
        self.assertEqual(sens[11], ref_sens[11])
        self.assertEqual(sens[12], ref_sens[12])

        # Test case II: Check exception for inf prior score
        parameters = [0, 1, 2, 3, 4, 5, np.inf, 7, 8, 9, 10, 11, 12]
        score, _ = self.log_posterior.evaluateS1(parameters)
        self.assertEqual(score, -np.inf)

    def test_get_log_likelihood(self):
        log_likelihood = self.log_posterior.get_log_likelihood()
        self.assertIsInstance(log_likelihood, chi.HierarchicalLogLikelihood)

    def test_get_log_prior(self):
        log_prior = self.log_posterior.get_log_prior()
        self.assertIsInstance(log_prior, pints.LogPrior)

    def test_get_id(self):
        ids = self.log_posterior.get_id()

        self.assertEqual(len(ids), 13)
        self.assertEqual(ids[0], 'Log-likelihood 1')
        self.assertEqual(ids[1], 'Log-likelihood 2')
        self.assertIsNone(ids[2])
        self.assertIsNone(ids[3])
        self.assertIsNone(ids[4])
        self.assertIsNone(ids[5])
        self.assertIsNone(ids[6])
        self.assertIsNone(ids[7])
        self.assertIsNone(ids[8])
        self.assertIsNone(ids[9])
        self.assertIsNone(ids[10])
        self.assertIsNone(ids[11])
        self.assertIsNone(ids[12])

    def test_get_parameter_names(self):
        # Test case I: without ids
        parameter_names = self.log_posterior.get_parameter_names()

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'central.size')
        self.assertEqual(parameter_names[1], 'central.size')
        self.assertEqual(parameter_names[2], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[3], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[4], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[5], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[6], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[7], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[8], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[11], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[12], 'Pooled Dim. 9')

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'central.size')
        self.assertEqual(parameter_names[1], 'central.size')
        self.assertEqual(parameter_names[2], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[3], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[4], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[5], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[6], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[7], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[8], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[11], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[12], 'Pooled Dim. 9')

        # Test case II: Exclude bottom-level
        parameter_names = self.log_posterior.get_parameter_names(
            exclude_bottom_level=True)

        self.assertEqual(len(parameter_names), 11)
        self.assertEqual(parameter_names[0], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[1], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[2], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[3], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[4], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[5], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[6], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[7], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[8], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 9')

        # Test case III: with ids
        parameter_names = self.log_posterior.get_parameter_names(
            include_ids=True)

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'Log-likelihood 1 central.size')
        self.assertEqual(parameter_names[1], 'Log-likelihood 2 central.size')
        self.assertEqual(parameter_names[2], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[3], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[4], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[5], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[6], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[7], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[8], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[11], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[12], 'Pooled Dim. 9')

        # Test case IV: Exclude bottom-level with IDs
        parameter_names = self.log_posterior.get_parameter_names(
            exclude_bottom_level=True, include_ids=True)

        self.assertEqual(parameter_names[0], 'Pooled Dim. 1')
        self.assertEqual(parameter_names[1], 'Pooled Dim. 2')
        self.assertEqual(parameter_names[2], 'Log mean Dim. 3')
        self.assertEqual(parameter_names[3], 'Log std. Dim. 3')
        self.assertEqual(parameter_names[4], 'Pooled Dim. 4')
        self.assertEqual(parameter_names[5], 'ID 1 Dim. 5')
        self.assertEqual(parameter_names[6], 'ID 2 Dim. 5')
        self.assertEqual(
            parameter_names[7], 'Pooled Dim. 6')
        self.assertEqual(
            parameter_names[8], 'Pooled Dim. 7')
        self.assertEqual(
            parameter_names[9], 'Pooled Dim. 8')
        self.assertEqual(
            parameter_names[10], 'Pooled Dim. 9')

    def test_get_population_model(self):
        population_model = self.log_posterior.get_population_model()
        self.assertIsInstance(population_model, chi.PopulationModel)

    def test_n_ids(self):
        n_ids = self.log_posterior.n_ids()
        self.assertEqual(n_ids, 2)

    def test_n_parameters(self):
        # Test case I: All parameters
        # 9 individual parameters, from which 1 is modelled heterogeneously,
        # 1 log-normally and the rest is pooled
        # And there are 2 individuals
        n_parameters = 2 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.assertEqual(
            self.log_posterior.n_parameters(), n_parameters)

        # Test case II: Exclude bottom parameters
        n_parameters = self.log_posterior.n_parameters(
            exclude_bottom_level=True)
        self.assertEqual(n_parameters, 11)

    def test_sample_initial_parameters(self):
        # Bad input
        n_samples = 0
        with self.assertRaisesRegex(ValueError, 'The number of samples has'):
            self.log_posterior.sample_initial_parameters(n_samples=n_samples)

        samples = self.log_posterior.sample_initial_parameters()
        self.assertEqual(samples.shape, (1, 13))

        n_samples = 10
        samples = self.log_posterior.sample_initial_parameters(
            n_samples=n_samples)
        self.assertEqual(samples.shape, (10, 13))

        seed = 3
        samples = self.log_posterior.sample_initial_parameters(seed=seed)
        self.assertEqual(samples.shape, (1, 13))

        # Test simple population model
        # Create data
        obs_1 = [1, 1.1, 1.2, 1.3]
        times_1 = [1, 2, 3, 4]
        observations = [obs_1]
        times = [times_1]

        # Set up mechanistic and error models
        model = ToyExponentialModel()
        error_model = chi.GaussianErrorModel()

        # Create log-likelihoods
        log_likelihoods = [
            chi.LogLikelihood(
                model, error_model, observations, times),
            chi.LogLikelihood(
                model, error_model, observations, times)]

        # Create population models
        population_model = chi.GaussianModel(n_dim=3)

        # Create hierarchical log-likelihood
        log_likelihood = chi.HierarchicalLogLikelihood(
            log_likelihoods, population_model)

        # Define log-prior
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1))

        # Create log-posterior
        log_posterior = chi.HierarchicalLogPosterior(
            log_likelihood, log_prior)

        n_samples = 10
        samples = log_posterior.sample_initial_parameters(
            n_samples=n_samples)
        self.assertEqual(samples.shape, (10, 12))

        # Test sampling for covariate population model
        # Create data
        obs_1 = [1, 1.1, 1.2, 1.3]
        times_1 = [1, 2, 3, 4]
        observations = [obs_1]
        times = [times_1]
        covariates = np.arange(2).reshape(2, 1) + 1000

        # Set up mechanistic and error models
        model = ToyExponentialModel()
        error_model = chi.GaussianErrorModel()

        # Create log-likelihoods
        log_likelihoods = [
            chi.LogLikelihood(
                model, error_model, observations, times),
            chi.LogLikelihood(
                model, error_model, observations, times)]

        # Create population models
        population_model = chi.ComposedPopulationModel([
            chi.CovariatePopulationModel(
                chi.GaussianModel(), chi.LinearCovariateModel()),
            chi.PooledModel(n_dim=2)])

        # Create hierarchical log-likelihood
        log_likelihood = chi.HierarchicalLogLikelihood(
            log_likelihoods, population_model, covariates=covariates)

        # Define log-prior
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1))

        # Create log-posterior
        log_posterior = chi.HierarchicalLogPosterior(
            log_likelihood, log_prior)

        n_samples = 10
        samples = log_posterior.sample_initial_parameters(
            n_samples=n_samples)
        self.assertEqual(samples.shape, (10, 8))

        # Test no bottom level parameters
        # Create population models
        population_model = chi.PooledModel(n_dim=3)

        # Create hierarchical log-likelihood
        log_likelihood = chi.HierarchicalLogLikelihood(
            log_likelihoods, population_model, covariates=covariates)

        # Define log-prior
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1))

        # Create log-posterior
        log_posterior = chi.HierarchicalLogPosterior(
            log_likelihood, log_prior)

        n_samples = 10
        samples = log_posterior.sample_initial_parameters(
            n_samples=n_samples)
        self.assertEqual(samples.shape, (10, 3))

        # Test fixed population parameters
        # Fix population parameters
        population_model = chi.ComposedPopulationModel([
            chi.CovariatePopulationModel(
                chi.GaussianModel(), chi.LinearCovariateModel()),
            chi.PooledModel(n_dim=2)])
        population_model = chi.ReducedPopulationModel(population_model)
        population_model.fix_parameters({'Std. Dim. 1': 1})

        # Create hierarchical log-likelihood
        log_likelihood = chi.HierarchicalLogLikelihood(
            log_likelihoods, population_model, covariates=covariates)

        # Define log-prior
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1))

        # Create log-posterior
        log_posterior = chi.HierarchicalLogPosterior(
            log_likelihood, log_prior)

        n_samples = 10
        samples = log_posterior.sample_initial_parameters(
            n_samples=n_samples)
        self.assertEqual(samples.shape, (10, 7))

        # Fix all parameters but one
        population_model.fix_parameters({
            'Mean Dim. 1': 1,
            'Mean Dim. 1 Cov. 1': 1,
            'Std. Dim. 1 Cov. 1': 1,
            'Pooled Dim. 1': 1})

        # Create hierarchical log-likelihood
        log_likelihood = chi.HierarchicalLogLikelihood(
            log_likelihoods, population_model, covariates=covariates)

        # Define log-prior
        log_prior = pints.LogNormalLogPrior(1, 1)

        # Create log-posterior
        log_posterior = chi.HierarchicalLogPosterior(
            log_likelihood, log_prior)

        n_samples = 10
        samples = log_posterior.sample_initial_parameters(
            n_samples=n_samples)
        self.assertEqual(samples.shape, (10, 3))


class TestLogLikelihood(unittest.TestCase):
    """
    Tests the chi.LogLikelihood class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        obs_1 = [1, 1.1, 1.2, 1.3]
        times_1 = [1, 2, 3, 4]
        obs_2 = [2, 2.1, 2.2]
        times_2 = [2, 5, 6]

        cls.observations = [obs_1, obs_2]
        cls.times = [times_1, times_2]

        # Set up mechanistic and error models
        cls.model = ModelLibrary().one_compartment_pk_model()
        cls.model.set_administration('central', direct=False)
        cls.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        cls.error_models = [
            chi.ConstantAndMultiplicativeGaussianErrorModel()] * 2

        # Create log-likelihood
        cls.log_likelihood = chi.LogLikelihood(
            cls.model, cls.error_models, cls.observations, cls.times)

        error_models = [chi.GaussianErrorModel()] * 2
        cls.log_likelihood2 = chi.LogLikelihood(
            cls.model, error_models, cls.observations, cls.times)

    def test_instantiation(self):
        # Check whether changing the model changes the log-likelihood
        obs = [1, 1.1, 1.2, 1.3]
        times = [1, 2, 3, 4]
        model = ModelLibrary().one_compartment_pk_model()
        error_model = [
            chi.ConstantAndMultiplicativeGaussianErrorModel()]
        log_likelihood = chi.LogLikelihood(
            model, error_model, obs, times)
        m = log_likelihood.get_submodels()['Mechanistic model']
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(m.n_parameters(), 3)

        model.set_administration('central', direct=False)
        m = log_likelihood.get_submodels()['Mechanistic model']
        self.assertEqual(model.n_parameters(), 5)
        self.assertEqual(m.n_parameters(), 3)

    def test_bad_instantiation(self):
        # Mechantic model has wrong type
        mechanistic_model = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The mechanistic model'):
            chi.LogLikelihood(
                mechanistic_model, self.error_models, self.observations,
                self.times)

        # Wrong number of error models
        outputs = ['central.drug_amount']
        with self.assertRaisesRegex(ValueError, 'One error model has'):
            chi.LogLikelihood(
                self.model, self.error_models, self.observations, self.times,
                outputs)

        # Wrong number of error models
        error_models = ['Wrong', 'type']
        with self.assertRaisesRegex(TypeError, 'The error models have to'):
            chi.LogLikelihood(
                self.model, error_models, self.observations, self.times)

        # Wrong length of observations
        observations = [['There'], ['are'], ['only two outputs']]
        with self.assertRaisesRegex(ValueError, 'The observations have'):
            chi.LogLikelihood(
                self.model, self.error_models, observations, self.times)

        # Wrong length of times
        times = [['There'], ['are'], ['only two outputs']]
        with self.assertRaisesRegex(ValueError, 'The times have the wrong'):
            chi.LogLikelihood(
                self.model, self.error_models, self.observations, times)

        # Negative times
        observations = [[1, 2], [1, 2]]
        times = [[-1, 2], [1, 2]]
        with self.assertRaisesRegex(ValueError, 'Times cannot be negative'):
            chi.LogLikelihood(
                self.model, self.error_models, observations, times)

        # Not strictly increasing times
        observations = [[1, 2], [1, 2]]
        times = [[2, 1], [1, 2]]
        with self.assertRaisesRegex(ValueError, 'Times must be increasing.'):
            chi.LogLikelihood(
                self.model, self.error_models, observations, times)

        # Observations and times don't match
        observations = [[1, 2], [1, 2]]  # Times have 4 and 3
        with self.assertRaisesRegex(ValueError, 'The observations and times'):
            chi.LogLikelihood(
                self.model, self.error_models, observations, self.times)

    def test_call_and_compute_pointwise_ll(self):
        # Test case I: Compute reference score manually
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        times = self.times[0]
        observations = self.observations[0]
        model_output = self.model.simulate(parameters[:5], times)
        model_output = model_output[0]
        error_model = self.error_models[0]
        ref_score_1 = error_model.compute_log_likelihood(
            parameters[5:7], model_output, observations)

        times = self.times[1]
        observations = self.observations[1]
        model_output = self.model.simulate(parameters[:5], times)
        model_output = model_output[1]
        error_model = self.error_models[1]
        ref_score_2 = error_model.compute_log_likelihood(
            parameters[7:9], model_output, observations)

        ref_score = ref_score_1 + ref_score_2
        score = self.log_likelihood(parameters)
        pw_score = self.log_likelihood.compute_pointwise_ll(
            parameters)

        self.assertAlmostEqual(score, ref_score)
        n_obs = 7
        self.assertEqual(pw_score.shape, (n_obs,))
        self.assertEqual(np.sum(pw_score), score)

        # Test case II: Compute reference score with two likelihoods
        parameters = [9, 8, 7, 6, 5, 4, 3, 2, 1]

        times = self.times[0]
        observations = self.observations[0]
        self.model.set_outputs(['central.drug_amount'])
        error_model = self.error_models[0]
        log_likelihood = chi.LogLikelihood(
            self.model, error_model, observations, times)
        ref_score_1 = log_likelihood(parameters[:7])

        times = self.times[1]
        observations = self.observations[1]
        self.model.set_outputs(['dose.drug_amount'])
        error_model = self.error_models[1]
        log_likelihood = chi.LogLikelihood(
            self.model, error_model, observations, times)
        ref_score_2 = log_likelihood(parameters[:5] + parameters[7:9])

        ref_score = ref_score_1 + ref_score_2
        score = self.log_likelihood(parameters)
        pw_score = self.log_likelihood.compute_pointwise_ll(
            parameters)

        self.assertAlmostEqual(score, ref_score)
        n_obs = 7
        self.assertEqual(pw_score.shape, (n_obs,))
        self.assertEqual(np.sum(pw_score), score)

        # Reset number of outputs
        self.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])

        # Make sure that call works even when sensitivities were initially
        # switched on
        m = self.log_likelihood.get_submodels()['Mechanistic model']
        m.enable_sensitivities(True)
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.log_likelihood(parameters)
        m.enable_sensitivities(True)
        self.log_likelihood.compute_pointwise_ll(parameters)

        # Leave observations for one outputs empty
        obs = [[], self.observations[1]]
        times = [[], self.times[1]]
        ll = chi.LogLikelihood(
            self.model, self.error_models, obs, times)

        parameters = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        score = ll(parameters)
        pw_score = ll.compute_pointwise_ll(parameters)
        self.assertEqual(score, ref_score_2)
        n_obs = 3
        self.assertEqual(pw_score.shape, (n_obs,))
        self.assertEqual(np.sum(pw_score), score)

    def test_evaluateS1(self):
        # Test case I: Compute reference score manually
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        times = self.times[0]
        observations = self.observations[0]
        self.model.enable_sensitivities(True)
        model_output, model_sens = self.model.simulate(
            parameters[:5], times)
        model_output = model_output[0]
        model_sens = model_sens[:, 0, :]
        error_model = self.error_models[0]
        ref_score_1, ref_sens_1 = error_model.compute_sensitivities(
            parameters[5:7], model_output, model_sens, observations)
        times = self.times[1]
        observations = self.observations[1]
        model_output, model_sens = self.model.simulate(
            parameters[:5], times)
        model_output = model_output[1]
        model_sens = model_sens[:, 1, :]
        error_model = self.error_models[1]
        ref_score_2, ref_sens_2 = error_model.compute_sensitivities(
            parameters[7:9], model_output, model_sens, observations)

        ref_score = ref_score_1 + ref_score_2
        score, sens = self.log_likelihood.evaluateS1(parameters)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 9)
        ref_dpsi = ref_sens_1[:5] + ref_sens_2[:5]
        self.assertAlmostEqual(sens[0], ref_dpsi[0])
        self.assertAlmostEqual(sens[1], ref_dpsi[1])
        self.assertAlmostEqual(sens[2], ref_dpsi[2])
        self.assertAlmostEqual(sens[3], ref_dpsi[3])
        self.assertAlmostEqual(sens[4], ref_dpsi[4])
        self.assertAlmostEqual(sens[5], ref_sens_1[5])
        self.assertAlmostEqual(sens[6], ref_sens_1[6])
        self.assertAlmostEqual(sens[7], ref_sens_2[5])
        self.assertAlmostEqual(sens[8], ref_sens_2[6])

        # # TODO: For now this remains a myokit problem!
        # # (can investigate further when that is fixed!!)
        # # Test case II: Comparison against numpy gradients
        # # Test case II.1: ConstantAndMultiplicativeError
        # self.log_likelihood._mechanistic_model.simulator.set_tolerance(
        #     abs_tol=1E-10, rel_tol=1E-10
        # )
        # epsilon = 0.00001
        # n_parameters = self.log_likelihood.n_parameters()
        # parameters = np.full(shape=n_parameters, fill_value=0.1)
        # ref_sens = []
        # for index in range(n_parameters):
        #     # Construct parameter grid
        #     low = parameters.copy()
        #     low[index] -= epsilon
        #     high = parameters.copy()
        #     high[index] += epsilon

        #     # Compute reference using numpy.gradient
        #     sens = np.gradient(
        #         [
        #             self.log_likelihood(low),
        #             self.log_likelihood(parameters),
        #             self.log_likelihood(high)],
        #         (epsilon))
        #     print('%d :' % index, sens)
        #     ref_sens.append(sens[1])

        # # Compute sensitivities with hierarchical model
        # _, sens = self.log_likelihood.evaluateS1(parameters)

        # self.assertEqual(len(sens), 9)
        # self.assertAlmostEqual(sens[0], ref_sens[0])
        # self.assertAlmostEqual(sens[1], ref_sens[1])
        # self.assertAlmostEqual(sens[2], ref_sens[2])
        # self.assertAlmostEqual(sens[3], ref_sens[3])
        # self.assertAlmostEqual(sens[4], ref_sens[4])
        # self.assertAlmostEqual(sens[5], ref_sens[5])
        # self.assertAlmostEqual(sens[6], ref_sens[6])
        # self.assertAlmostEqual(sens[7], ref_sens[7])
        # self.assertEqual(sens[8], ref_sens[8])

        # #TODO: Looks like the problem is the approximation of sensitivity
        # ODEs. We can improve by tuning sensitivity of adaptivity, but it
        # doesn't help much.
        # # Test case II.1: GaussianError
        # epsilon = 0.000001
        # print(self.log_likelihood2.get_parameter_names())
        # n_parameters = self.log_likelihood2.n_parameters()
        # parameters = np.full(shape=n_parameters, fill_value=0.1)
        # ref_sens = []
        # for index in range(n_parameters):
        #     # Construct parameter grid
        #     low = np.full(shape=n_parameters, fill_value=0.1)
        #     low[index] -= epsilon
        #     high = np.full(shape=n_parameters, fill_value=0.1)
        #     high[index] += epsilon

        #     # Compute reference using numpy.gradient
        #     sens = np.gradient(
        #         [
        #             self.log_likelihood2(low),
        #             self.log_likelihood2(parameters),
        #             self.log_likelihood2(high)],
        #         (epsilon))
        #     ref_sens.append(sens[1])

        # # Compute sensitivities with hierarchical model
        # _, sens = self.log_likelihood2.evaluateS1(parameters)
        # self.log_likelihood2._mechanistic_model.simulator.set_tolerance(
        #     abs_tol=1E-10, rel_tol=1E-10
        # )
        # _, sens = self.log_likelihood2.evaluateS1(parameters)

        # self.assertEqual(len(sens), 7)
        # self.assertAlmostEqual(sens[0], ref_sens[0], 5)
        # self.assertAlmostEqual(sens[1], ref_sens[1], 5)
        # self.assertAlmostEqual(sens[2], ref_sens[2], 5)
        # self.assertAlmostEqual(sens[3], ref_sens[3], 5)
        # self.assertAlmostEqual(sens[4], ref_sens[4], 5)
        # self.assertAlmostEqual(sens[5], ref_sens[5], 5)
        # self.assertAlmostEqual(sens[6], ref_sens[6], 5)

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'central.drug_amount': 1,
            'dose.absorption_rate': 1})

        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 7)

        parameter_names = self.log_likelihood.get_parameter_names()
        self.assertEqual(len(parameter_names), 7)
        self.assertEqual(parameter_names[0], 'dose.drug_amount')
        self.assertEqual(parameter_names[1], 'central.size')
        self.assertEqual(parameter_names[2], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[3], 'central.drug_amount Sigma base')
        self.assertEqual(parameter_names[4], 'central.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[5], 'dose.drug_amount Sigma base')
        self.assertEqual(parameter_names[6], 'dose.drug_amount Sigma rel.')

        # Test case II: fix overlapping set of parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'dose.absorption_rate': None,
            'dose.drug_amount Sigma base': 0.5,
            'myokit.elimination_rate': 0.3})

        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 6)

        parameter_names = self.log_likelihood.get_parameter_names()
        self.assertEqual(len(parameter_names), 6)
        self.assertEqual(parameter_names[0], 'dose.drug_amount')
        self.assertEqual(parameter_names[1], 'central.size')
        self.assertEqual(parameter_names[2], 'dose.absorption_rate')
        self.assertEqual(parameter_names[3], 'central.drug_amount Sigma base')
        self.assertEqual(parameter_names[4], 'central.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[5], 'dose.drug_amount Sigma rel.')

        # Test case III: unfix all parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'central.drug_amount': None,
            'dose.drug_amount Sigma base': None,
            'myokit.elimination_rate': None})

        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 9)

        parameter_names = self.log_likelihood.get_parameter_names()
        self.assertEqual(len(parameter_names), 9)
        self.assertEqual(parameter_names[0], 'central.drug_amount')
        self.assertEqual(parameter_names[1], 'dose.drug_amount')
        self.assertEqual(parameter_names[2], 'central.size')
        self.assertEqual(parameter_names[3], 'dose.absorption_rate')
        self.assertEqual(parameter_names[4], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[5], 'central.drug_amount Sigma base')
        self.assertEqual(parameter_names[6], 'central.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[7], 'dose.drug_amount Sigma base')
        self.assertEqual(parameter_names[8], 'dose.drug_amount Sigma rel.')

    def test_fix_parameters_bad_input(self):
        name_value_dict = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The name-value dictionary'):
            self.log_likelihood.fix_parameters(name_value_dict)

    def test_get_parameter_names(self):
        # Test case I: Single output problem
        parameter_names = self.log_likelihood.get_parameter_names()

        self.assertEqual(len(parameter_names), 9)
        self.assertEqual(parameter_names[0], 'central.drug_amount')
        self.assertEqual(parameter_names[1], 'dose.drug_amount')
        self.assertEqual(parameter_names[2], 'central.size')
        self.assertEqual(parameter_names[3], 'dose.absorption_rate')
        self.assertEqual(parameter_names[4], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[5], 'central.drug_amount Sigma base')
        self.assertEqual(parameter_names[6], 'central.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[7], 'dose.drug_amount Sigma base')
        self.assertEqual(parameter_names[8], 'dose.drug_amount Sigma rel.')

    def test_get_set_id(self):
        # Test case I: Check default
        self.assertIsNone(self.log_likelihood.get_id())

        # Test case II: Set ID
        _id = '123'
        self.log_likelihood.set_id(_id)
        self.assertEqual(self.log_likelihood.get_id(), _id)

    def test_get_submodels(self):
        # Test case I: no fixed parameters
        submodels = self.log_likelihood.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, chi.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 2)
        self.assertIsInstance(error_models[0], chi.ErrorModel)
        self.assertIsInstance(error_models[1], chi.ErrorModel)

        # Test case II: some fixed parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'central.drug_amount': 1,
            'dose.drug_amount Sigma base': 1})
        submodels = self.log_likelihood.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, chi.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 2)
        self.assertIsInstance(error_models[0], chi.ErrorModel)
        self.assertIsInstance(error_models[1], chi.ErrorModel)

        # Unfix parameter
        self.log_likelihood.fix_parameters({
            'central.drug_amount': None,
            'dose.drug_amount Sigma base': None})

    def test_n_parameters(self):
        # Test case I:
        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 9)

        # Test case II:
        times = self.times[0]
        observations = self.observations[0]
        self.model.set_outputs(['central.drug_amount'])
        error_model = self.error_models[0]
        log_likelihood = chi.LogLikelihood(
            self.model, error_model, observations, times)

        n_parameters = log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 7)

        # Reset number of outputs
        self.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])

    def test_n_observations(self):
        # Test case I:
        n_obs = self.log_likelihood.n_observations()

        self.assertEqual(len(n_obs), 2)
        self.assertEqual(n_obs[0], 4)
        self.assertEqual(n_obs[1], 3)

    def test_set_id(self):
        self.log_likelihood.set_id(1.12)
        self.assertEqual(self.log_likelihood.get_id(), '1')


class TestLogPosterior(unittest.TestCase):
    """
    Tests the chi.LogPosterior class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test dataset
        times = [0, 1, 2, 3]
        values = [10, 11, 12, 13]

        # Create test model
        model = ModelLibrary().tumour_growth_inhibition_model_koch()
        error_model = chi.ConstantAndMultiplicativeGaussianErrorModel()
        cls.log_likelihood = chi.LogLikelihood(
            model, error_model, values, times)
        cls.log_likelihood.set_id('42')
        cls.log_prior = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1))
        cls.log_posterior = chi.LogPosterior(
            cls.log_likelihood, cls.log_prior)

    def test_bad_instantiation(self):
        # Log-likelihood has bad type
        log_likelihood = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The log-likelihood has to'):
            chi.LogPosterior(log_likelihood, self.log_prior)

        # Log-prior has bad type
        log_prior = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The log-prior has to'):
            chi.LogPosterior(self.log_likelihood, log_prior)

        # The dimensionality of likelihood and prior don't match
        log_prior = pints.UniformLogPrior(0, 1)
        with self.assertRaisesRegex(ValueError, 'The log-prior and the'):
            chi.LogPosterior(self.log_likelihood, log_prior)

    def test_call(self):
        parameters = [1, 2, 3, 4, 5, 6, 7]
        ref_score = \
            self.log_likelihood(parameters) + self.log_prior(parameters)
        score = self.log_posterior(parameters)
        self.assertEqual(score, ref_score)

    def test_evaluateS1(self):
        parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ref_score_1, ref_sens_1 = self.log_likelihood.evaluateS1(parameters)
        ref_score_2, ref_sens_2 = self.log_prior.evaluateS1(parameters)
        ref_score = ref_score_1 + ref_score_2
        ref_sens = ref_sens_1 + ref_sens_2
        score, sens = self.log_posterior.evaluateS1(parameters)
        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 7)
        self.assertEqual(len(ref_sens), 7)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])

        # Test that it returns inf when parameters outside
        # priors
        parameters = [-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        score, _ = self.log_posterior.evaluateS1(parameters)
        self.assertEqual(score, -np.inf)

    def test_get_id(self):
        # Test case I: Non-trivial IDs
        _id = self.log_posterior.get_id()
        self.assertEqual(_id, '42')

    def test_get_log_likelihood(self):
        log_likelihood = self.log_posterior.get_log_likelihood()
        self.assertIsInstance(log_likelihood, chi.LogLikelihood)

    def test_get_log_prior(self):
        log_prior = self.log_posterior.get_log_prior()
        self.assertIsInstance(log_prior, pints.LogPrior)

    def test_get_parameter_names(self):
        # Test case I: Non-trivial parameters
        parameter_names = self.log_posterior.get_parameter_names()

        self.assertEqual(len(parameter_names), 7)
        self.assertEqual(parameter_names[0], 'myokit.tumour_volume')
        self.assertEqual(parameter_names[1], 'myokit.drug_concentration')
        self.assertEqual(parameter_names[2], 'myokit.kappa')
        self.assertEqual(parameter_names[3], 'myokit.lambda_0')
        self.assertEqual(parameter_names[4], 'myokit.lambda_1')
        self.assertEqual(parameter_names[5], 'Sigma base')
        self.assertEqual(parameter_names[6], 'Sigma rel.')

    def test_sample_initial_parameters(self):
        samples = self.log_posterior.sample_initial_parameters()
        self.assertEqual(samples.shape, (1, 7))

        n_samples = 10
        samples = self.log_posterior.sample_initial_parameters(
            n_samples=n_samples)
        self.assertEqual(samples.shape, (10, 7))

        seed = 3
        samples = self.log_posterior.sample_initial_parameters(seed=seed)
        self.assertEqual(samples.shape, (1, 7))


class TestPopulationFilterLogPosterior(unittest.TestCase):
    """
    Tests the chi.PopulationFilterLogPosterior class.
    """
    @classmethod
    def setUpClass(cls):
        observations = np.array([
            [[1, 2, np.nan, 5]],
            [[0.1, 2, 4, 3]],
            [[np.nan, 3, 2, np.nan]],
            [[0, 20, 13, -4]],
            [[21, 0.2, 8, 4]],
            [[0.1, 0.2, 0.3, 0.4]]])
        population_filter = chi.GaussianFilter(observations)
        times = np.array([1, 2, 3, 4])
        mechanistic_model = ToyExponentialModel()
        population_model = chi.ComposedPopulationModel([
            chi.LogNormalModel(n_dim=1),
            chi.LogNormalModel(n_dim=1, centered=False)])
        n_samples = 3

        # Test case I: Fixed sigma, Gaussian error
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.1, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.1, 0.1)
        )
        sigma = 1
        error_on_log_scale = False
        cls.log_posterior1 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples
        )

        # Test case II: Fixed sigma, log-normal error
        sigma = 1
        error_on_log_scale = True
        cls.log_posterior2 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples
        )

        # Test case III: free sigma, Gaussian error
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.1, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.1, 0.1),
            pints.LogNormalLogPrior(0.1, 0.1)
        )
        sigma = None
        error_on_log_scale = False
        cls.log_posterior3 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples
        )

        # Test case IV: free sigma, log-normal error
        sigma = None
        error_on_log_scale = True
        cls.log_posterior4 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples
        )

        # Test case V: all pooled
        population_model = chi.PooledModel(n_dim=2)
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1)
        )
        sigma = 1
        error_on_log_scale = False
        cls.log_posterior5 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples
        )

        # Test case VI: all heterogen.
        population_model = chi.HeterogeneousModel(n_dim=2)
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.1, 0.1)
        )
        sigma = 1
        error_on_log_scale = False
        cls.log_posterior6 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples
        )

        # Test case VII: mix
        population_model = chi.ComposedPopulationModel([
            chi.HeterogeneousModel(),
            chi.PooledModel()])
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
        )
        sigma = 1
        error_on_log_scale = False
        cls.log_posterior7 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples
        )

        # Test case VIII: Covariate model
        # Test case VII: mix
        cov_model = chi.CovariatePopulationModel(
            chi.GaussianModel(),
            chi.LinearCovariateModel()
        )
        population_model = chi.ComposedPopulationModel([
            cov_model,
            chi.PooledModel()])
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
        )
        sigma = 1
        covariates = [2]
        error_on_log_scale = False
        cls.log_posterior8 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples,
            covariates=covariates
        )

        # Test case IX: all pooled
        population_model = chi.PooledModel(n_dim=2)
        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1)
        )
        sigma = 1
        error_on_log_scale = False
        cls.log_posterior9 = chi.PopulationFilterLogPosterior(
            population_filter=population_filter,
            times=times,
            mechanistic_model=mechanistic_model,
            population_model=population_model,
            log_prior=log_prior,
            sigma=sigma,
            error_on_log_scale=error_on_log_scale,
            n_samples=n_samples
        )

    def test_bad_instantiation(self):
        observations = np.array([
            [[1, 2, np.nan, 5]],
            [[0.1, 2, 4, 3]],
            [[np.nan, 3, 2, np.nan]],
            [[0, 20, 13, -4]],
            [[21, 0.2, 8, 4]],
            [[0.1, 0.2, 0.3, 0.4]]])
        population_filter = chi.GaussianFilter(observations)
        times = np.array([1, 2, 3, 4])
        mechanistic_model = ToyExponentialModel()
        population_model = chi.LogNormalModel(n_dim=2)

        log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.1, 0.1),
            pints.LogNormalLogPrior(0.3, 0.1),
            pints.LogNormalLogPrior(0.1, 0.1)
        )

        # Population filter has the wrong type
        p = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The population filter has to'):
            chi.PopulationFilterLogPosterior(
                p, times, mechanistic_model,
                population_model, log_prior)

        # Times are not unique
        t = np.array([1, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, 'The measurement times in'):
            chi.PopulationFilterLogPosterior(
                population_filter, t, mechanistic_model,
                population_model, log_prior)

        # Times and observations do not match
        t = np.array([1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, 'The length of times does'):
            chi.PopulationFilterLogPosterior(
                population_filter, t, mechanistic_model,
                population_model, log_prior)

        # Mechanistic model has the wrong type
        m = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The mechanistic model has to'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, m,
                population_model, log_prior)

        # Mechanistic model has the wrong number of outputs
        p = chi.GaussianFilter(np.ones(shape=(1, 2, 4)))
        with self.assertRaisesRegex(ValueError, 'The number of mechanistic'):
            chi.PopulationFilterLogPosterior(
                p, times, mechanistic_model,
                population_model, log_prior)

        # Population model has the wrong type
        p = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The population model has to'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                p, log_prior)

        # Population model has the wrong dimensionality
        p = chi.PooledModel(n_dim=10)
        with self.assertRaisesRegex(ValueError, 'The number of population'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                p, log_prior)

        # The number of covariates does not match the number of covariates
        # required by the population model
        p = chi.ComposedPopulationModel([
            chi.CovariatePopulationModel(
                chi.PooledModel(), chi.LinearCovariateModel()),
            chi.CovariatePopulationModel(
                chi.PooledModel(), chi.LinearCovariateModel())
        ])
        n_samples, n_cov = 10, 3
        c = np.ones(shape=(n_samples, n_cov))
        with self.assertRaisesRegex(ValueError, 'Invalid covariates.'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                p, log_prior, covariates=c)

        # The covariates cannot be broadcasted to (n_samples, n_cov)
        n_samples, n_cov = 10, 2
        c = np.ones(shape=(n_samples, n_cov))
        with self.assertRaisesRegex(ValueError, 'Invalid covariates.'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                p, log_prior, covariates=c)

        # The number of sigmas does not match the number of model outputs
        sigma = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'One sigma for each obs'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                population_model, log_prior, sigma=sigma)

        # Sigma has negative elements
        sigma = -1
        with self.assertRaisesRegex(ValueError, 'The elements of sigma'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                population_model, log_prior, sigma=sigma)

        # Log-prior has the wrong type
        p = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The log-prior has to'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                population_model, p)

        # Log-prior has the wrong dimensionality
        with self.assertRaisesRegex(ValueError, 'The dimensionality of'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                population_model, log_prior)

        # Number of samples is zero or negative
        with self.assertRaisesRegex(ValueError, 'The number of samples of'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                population_model, log_prior, sigma=1, n_samples=0)

        # Number of samples is zero or negative
        with self.assertRaisesRegex(ValueError, 'The number of samples of'):
            chi.PopulationFilterLogPosterior(
                population_filter, times, mechanistic_model,
                population_model, log_prior, sigma=1, n_samples=-10)

    def test_call(self):
        # Test case I: fixed sigma and Gaussian error model
        # Test case I.1: log-prior returns infinity
        parameters = np.ones(22) * -1
        score = self.log_posterior1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case I.2: population model returns infinity
        parameters[:4] = 1
        score = self.log_posterior1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case I.3: finite score for valid parameters
        parameters = np.linspace(0.1, 1, num=22)
        score = self.log_posterior1(parameters)
        self.assertFalse(np.isinf(score))

        # Test case II: fixed sigma and log-normal error model
        # Test case II.1: log-prior returns infinity
        parameters = np.ones(22) * -1
        score = self.log_posterior2(parameters)
        self.assertTrue(np.isinf(score))

        # Test case II.2: population model returns infinity
        parameters[:4] = 1
        score = self.log_posterior2(parameters)
        self.assertTrue(np.isinf(score))

        # Test case II.3: finite score for valid parameters
        parameters = np.linspace(0.1, 1, num=22)
        score = self.log_posterior2(parameters)
        self.assertFalse(np.isinf(score))

        # Test case III: free sigma and Gaussian error model
        # Test case III.1: log-prior returns infinity
        parameters = np.ones(23) * -1
        score = self.log_posterior3(parameters)
        self.assertTrue(np.isinf(score))

        # Test case III.2: population model returns infinity
        parameters[:4] = 1
        score = self.log_posterior3(parameters)
        self.assertTrue(np.isinf(score))

        # Test case III.3: finite score for valid parameters
        parameters = np.linspace(0.1, 1, num=23)
        score = self.log_posterior3(parameters)
        self.assertFalse(np.isinf(score))

        # Test case IV: free sigma and log-normal error model
        # Test case IV.1: log-prior returns infinity
        parameters = np.ones(23) * -1
        score = self.log_posterior4(parameters)
        self.assertTrue(np.isinf(score))

        # Test case IV.2: population model returns infinity
        parameters[:4] = 1
        score = self.log_posterior4(parameters)
        self.assertTrue(np.isinf(score))

        # Test case IV.3: finite score for valid parameters
        parameters = np.linspace(0.1, 1, num=23)
        score = self.log_posterior4(parameters)
        self.assertFalse(np.isinf(score))

        # Test case V: Fully pooled never works with Gaussian population filter
        parameters = np.ones(14)
        score = self.log_posterior5(parameters)
        self.assertTrue(np.isinf(score))

        # Test case VI: All heterogen.
        parameters = np.ones(18)
        parameters[2:4] = 2
        parameters[4:6] = 3
        score = self.log_posterior6(parameters)
        self.assertFalse(np.isinf(score))

        # Test case VII: mixed.
        parameters = np.ones(16)
        parameters[:3] = np.arange(3, 6)
        score = self.log_posterior7(parameters)
        self.assertFalse(np.isinf(score))

        # Test case VIII: covariate model.
        parameters = np.ones(20)
        parameters[:10] = np.arange(3, 13)
        score = self.log_posterior8(parameters)
        self.assertFalse(np.isinf(score))

    def test_sensitivities(self):
        # Test case I: fixed sigma and Gaussian error model
        # Test case I.1: log-prior returns infinity
        parameters = np.ones(22) * -1
        score, _ = self.log_posterior1.evaluateS1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case I.2: population model returns infinity
        parameters[:4] = 1
        score, _ = self.log_posterior1.evaluateS1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case I.3: finite difference check for valid parameters
        parameters = np.linspace(0.1, 1, num=22)
        epsilon = 0.00001
        ref_sens = []
        ref_score = self.log_posterior1(parameters)
        for index in range(len(parameters)):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.log_posterior1(low),
                    ref_score,
                    self.log_posterior1(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.log_posterior1.evaluateS1(parameters)

        self.assertAlmostEqual(score, ref_score)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertEqual(len(sens), 22)
        self.assertAlmostEqual(sens[0], ref_sens[0], places=4)
        self.assertAlmostEqual(sens[1], ref_sens[1], places=4)
        self.assertAlmostEqual(sens[2], ref_sens[2], places=4)
        self.assertAlmostEqual(sens[3], ref_sens[3], places=4)
        self.assertAlmostEqual(sens[4], ref_sens[4], places=4)
        self.assertAlmostEqual(sens[5], ref_sens[5], places=4)
        self.assertAlmostEqual(sens[6], ref_sens[6], places=4)
        self.assertAlmostEqual(sens[7], ref_sens[7], places=4)
        self.assertAlmostEqual(sens[8], ref_sens[8], places=4)
        self.assertAlmostEqual(sens[9], ref_sens[9], places=4)
        self.assertAlmostEqual(sens[10], ref_sens[10], places=4)
        self.assertAlmostEqual(sens[11], ref_sens[11], places=4)
        self.assertAlmostEqual(sens[12], ref_sens[12], places=4)
        self.assertAlmostEqual(sens[13], ref_sens[13], places=4)
        self.assertAlmostEqual(sens[14], ref_sens[14], places=4)
        self.assertAlmostEqual(sens[15], ref_sens[15], places=4)
        self.assertAlmostEqual(sens[16], ref_sens[16], places=4)
        self.assertAlmostEqual(sens[17], ref_sens[17], places=4)
        self.assertAlmostEqual(sens[18], ref_sens[18], places=4)
        self.assertAlmostEqual(sens[19], ref_sens[19], places=4)
        self.assertAlmostEqual(sens[20], ref_sens[20], places=4)
        self.assertAlmostEqual(sens[21], ref_sens[21], places=4)

        # Test case II: fixed sigma and log-normal error model
        # Test case II.1: log-prior returns infinity
        parameters = np.ones(22) * -1
        score, _ = self.log_posterior2.evaluateS1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case II.2: population model returns infinity
        parameters[:4] = 1
        score, _ = self.log_posterior2.evaluateS1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case II.3: finite difference check for valid parameters
        parameters = np.linspace(0.1, 1, num=22)
        epsilon = 0.00001
        ref_sens = []
        ref_score = self.log_posterior2(parameters)
        for index in range(len(parameters)):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.log_posterior2(low),
                    ref_score,
                    self.log_posterior2(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.log_posterior2.evaluateS1(parameters)

        self.assertAlmostEqual(score, ref_score)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertEqual(len(sens), 22)
        self.assertAlmostEqual(sens[0], ref_sens[0], places=4)
        self.assertAlmostEqual(sens[1], ref_sens[1], places=4)
        self.assertAlmostEqual(sens[2], ref_sens[2], places=4)
        self.assertAlmostEqual(sens[3], ref_sens[3], places=4)
        self.assertAlmostEqual(sens[4], ref_sens[4], places=4)
        self.assertAlmostEqual(sens[5], ref_sens[5], places=4)
        self.assertAlmostEqual(sens[6], ref_sens[6], places=4)
        self.assertAlmostEqual(sens[7], ref_sens[7], places=4)
        self.assertAlmostEqual(sens[8], ref_sens[8], places=4)
        self.assertAlmostEqual(sens[9], ref_sens[9], places=4)
        self.assertAlmostEqual(sens[10], ref_sens[10], places=4)
        self.assertAlmostEqual(sens[11], ref_sens[11], places=4)
        self.assertAlmostEqual(sens[12], ref_sens[12], places=4)
        self.assertAlmostEqual(sens[13], ref_sens[13], places=4)
        self.assertAlmostEqual(sens[14], ref_sens[14], places=4)
        self.assertAlmostEqual(sens[15], ref_sens[15], places=4)
        self.assertAlmostEqual(sens[16], ref_sens[16], places=4)
        self.assertAlmostEqual(sens[17], ref_sens[17], places=4)
        self.assertAlmostEqual(sens[18], ref_sens[18], places=4)
        self.assertAlmostEqual(sens[19], ref_sens[19], places=4)
        self.assertAlmostEqual(sens[20], ref_sens[20], places=4)
        self.assertAlmostEqual(sens[21], ref_sens[21], places=4)

        # Test case III: free sigma and Gaussian error model
        # Test case III.1: log-prior returns infinity
        parameters = np.ones(23) * -1
        score, _ = self.log_posterior3.evaluateS1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case III.2: population model returns infinity
        parameters[:4] = 1
        score, _ = self.log_posterior3.evaluateS1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case III.3: finite difference check for valid parameters
        parameters = np.linspace(0.1, 1, num=23)
        epsilon = 0.000001
        ref_sens = []
        ref_score = self.log_posterior3(parameters)
        for index in range(len(parameters)):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.log_posterior3(low),
                    ref_score,
                    self.log_posterior3(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.log_posterior3.evaluateS1(parameters)

        self.assertAlmostEqual(score, ref_score)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertEqual(len(sens), 23)
        self.assertAlmostEqual(sens[0], ref_sens[0], places=4)
        self.assertAlmostEqual(sens[1], ref_sens[1], places=4)
        self.assertAlmostEqual(sens[2], ref_sens[2], places=4)
        self.assertAlmostEqual(sens[3], ref_sens[3], places=4)
        self.assertAlmostEqual(sens[4], ref_sens[4], places=4)
        self.assertAlmostEqual(sens[5], ref_sens[5], places=4)
        self.assertAlmostEqual(sens[6], ref_sens[6], places=4)
        self.assertAlmostEqual(sens[7], ref_sens[7], places=4)
        self.assertAlmostEqual(sens[8], ref_sens[8], places=4)
        self.assertAlmostEqual(sens[9], ref_sens[9], places=4)
        self.assertAlmostEqual(sens[10], ref_sens[10], places=4)
        self.assertAlmostEqual(sens[11], ref_sens[11], places=4)
        self.assertAlmostEqual(sens[12], ref_sens[12], places=4)
        self.assertAlmostEqual(sens[13], ref_sens[13], places=4)
        self.assertAlmostEqual(sens[14], ref_sens[14], places=4)
        self.assertAlmostEqual(sens[15], ref_sens[15], places=4)
        self.assertAlmostEqual(sens[16], ref_sens[16], places=4)
        self.assertAlmostEqual(sens[17], ref_sens[17], places=4)
        self.assertAlmostEqual(sens[18], ref_sens[18], places=4)
        self.assertAlmostEqual(sens[19], ref_sens[19], places=4)
        self.assertAlmostEqual(sens[20], ref_sens[20], places=4)
        self.assertAlmostEqual(sens[21], ref_sens[21], places=4)
        self.assertAlmostEqual(sens[22], ref_sens[22], places=4)

        # Test case IV: free sigma and log-normal error model
        # Test case IV.1: log-prior returns infinity
        parameters = np.ones(23) * -1
        score, _ = self.log_posterior4.evaluateS1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case IV.2: population model returns infinity
        parameters[:4] = 1
        score, _ = self.log_posterior4.evaluateS1(parameters)
        self.assertTrue(np.isinf(score))

        # Test case IV.3: finite difference check for valid parameters
        parameters = np.linspace(0.1, 1, num=23)
        epsilon = 0.000001
        ref_sens = []
        ref_score = self.log_posterior4(parameters)
        for index in range(len(parameters)):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.log_posterior4(low),
                    ref_score,
                    self.log_posterior4(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.log_posterior4.evaluateS1(parameters)

        self.assertAlmostEqual(score, ref_score)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertEqual(len(sens), 23)
        self.assertAlmostEqual(sens[0], ref_sens[0], places=4)
        self.assertAlmostEqual(sens[1], ref_sens[1], places=4)
        self.assertAlmostEqual(sens[2], ref_sens[2], places=4)
        self.assertAlmostEqual(sens[3], ref_sens[3], places=4)
        self.assertAlmostEqual(sens[4], ref_sens[4], places=4)
        self.assertAlmostEqual(sens[5], ref_sens[5], places=4)
        self.assertAlmostEqual(sens[6], ref_sens[6], places=4)
        self.assertAlmostEqual(sens[7], ref_sens[7], places=4)
        self.assertAlmostEqual(sens[8], ref_sens[8], places=4)
        self.assertAlmostEqual(sens[9], ref_sens[9], places=4)
        self.assertAlmostEqual(sens[10], ref_sens[10], places=4)
        self.assertAlmostEqual(sens[11], ref_sens[11], places=4)
        self.assertAlmostEqual(sens[12], ref_sens[12], places=4)
        self.assertAlmostEqual(sens[13], ref_sens[13], places=4)
        self.assertAlmostEqual(sens[14], ref_sens[14], places=4)
        self.assertAlmostEqual(sens[15], ref_sens[15], places=4)
        self.assertAlmostEqual(sens[16], ref_sens[16], places=4)
        self.assertAlmostEqual(sens[17], ref_sens[17], places=4)
        self.assertAlmostEqual(sens[18], ref_sens[18], places=4)
        self.assertAlmostEqual(sens[19], ref_sens[19], places=4)
        self.assertAlmostEqual(sens[20], ref_sens[20], places=4)
        self.assertAlmostEqual(sens[21], ref_sens[21], places=4)
        self.assertAlmostEqual(sens[22], ref_sens[22], places=4)

        # Test case V: Fully pooled never works with Gaussian population filter
        parameters = np.ones(14)
        score, sens = self.log_posterior5.evaluateS1(parameters)

        self.assertTrue(np.isinf(score))
        self.assertEqual(len(sens), 14)

        # Test case VI: All heterogen.
        parameters = np.ones(18)
        parameters[2:4] = 2
        parameters[4:6] = 3
        epsilon = 0.00001
        ref_sens = []
        ref_score = self.log_posterior6(parameters)
        for index in range(len(parameters)):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.log_posterior6(low),
                    ref_score,
                    self.log_posterior6(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.log_posterior6.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertEqual(len(sens), 18)
        self.assertAlmostEqual(sens[0], ref_sens[0], places=4)
        self.assertAlmostEqual(sens[1], ref_sens[1], places=4)
        self.assertAlmostEqual(sens[2], ref_sens[2], places=4)
        self.assertAlmostEqual(sens[3], ref_sens[3], places=4)
        self.assertAlmostEqual(sens[4], ref_sens[4], places=4)
        self.assertAlmostEqual(sens[5], ref_sens[5], places=4)
        self.assertAlmostEqual(sens[6], ref_sens[6], places=4)
        self.assertAlmostEqual(sens[7], ref_sens[7], places=4)
        self.assertAlmostEqual(sens[8], ref_sens[8], places=4)
        self.assertAlmostEqual(sens[9], ref_sens[9], places=4)
        self.assertAlmostEqual(sens[10], ref_sens[10], places=4)
        self.assertAlmostEqual(sens[11], ref_sens[11], places=4)
        self.assertAlmostEqual(sens[12], ref_sens[12], places=4)
        self.assertAlmostEqual(sens[13], ref_sens[13], places=4)
        self.assertAlmostEqual(sens[14], ref_sens[14], places=4)
        self.assertAlmostEqual(sens[15], ref_sens[15], places=4)
        self.assertAlmostEqual(sens[16], ref_sens[16], places=4)
        self.assertAlmostEqual(sens[17], ref_sens[17], places=4)

        # Test case VII: mixed
        parameters = np.ones(16)
        parameters[:3] = np.arange(3, 6)
        epsilon = 0.00001
        ref_sens = []
        ref_score = self.log_posterior7(parameters)
        for index in range(len(parameters)):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.log_posterior7(low),
                    ref_score,
                    self.log_posterior7(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.log_posterior7.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertEqual(len(sens), 16)
        self.assertAlmostEqual(sens[0], ref_sens[0], places=4)
        self.assertAlmostEqual(sens[1], ref_sens[1], places=4)
        self.assertAlmostEqual(sens[2], ref_sens[2], places=4)
        self.assertAlmostEqual(sens[3], ref_sens[3], places=4)
        self.assertAlmostEqual(sens[4], ref_sens[4], places=4)
        self.assertAlmostEqual(sens[5], ref_sens[5], places=4)
        self.assertAlmostEqual(sens[6], ref_sens[6], places=4)
        self.assertAlmostEqual(sens[7], ref_sens[7], places=4)
        self.assertAlmostEqual(sens[8], ref_sens[8], places=4)
        self.assertAlmostEqual(sens[9], ref_sens[9], places=4)
        self.assertAlmostEqual(sens[10], ref_sens[10], places=4)
        self.assertAlmostEqual(sens[11], ref_sens[11], places=4)
        self.assertAlmostEqual(sens[12], ref_sens[12], places=4)
        self.assertAlmostEqual(sens[13], ref_sens[13], places=4)
        self.assertAlmostEqual(sens[14], ref_sens[14], places=4)
        self.assertAlmostEqual(sens[15], ref_sens[15], places=4)

        # Test case VIII: covariate model
        parameters = np.ones(20)
        parameters[:10] = np.arange(3, 13)
        epsilon = 0.00001
        ref_sens = []
        ref_score = self.log_posterior8(parameters)
        for index in range(len(parameters)):
            # Construct parameter grid
            low = parameters.copy()
            low[index] -= epsilon
            high = parameters.copy()
            high[index] += epsilon

            # Compute reference using numpy.gradient
            sens = np.gradient(
                [
                    self.log_posterior8(low),
                    ref_score,
                    self.log_posterior8(high)],
                (epsilon))
            ref_sens.append(sens[1])

        # Compute sensitivities from filter
        score, sens = self.log_posterior8.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertEqual(len(sens), 20)
        self.assertAlmostEqual(sens[0], ref_sens[0], places=4)
        self.assertAlmostEqual(sens[1], ref_sens[1], places=4)
        self.assertAlmostEqual(sens[2], ref_sens[2], places=4)
        self.assertAlmostEqual(sens[3], ref_sens[3], places=4)
        self.assertAlmostEqual(sens[4], ref_sens[4], places=4)
        self.assertAlmostEqual(sens[5], ref_sens[5], places=4)
        self.assertAlmostEqual(sens[6], ref_sens[6], places=4)
        self.assertAlmostEqual(sens[7], ref_sens[7], places=4)
        self.assertAlmostEqual(sens[8], ref_sens[8], places=4)
        self.assertAlmostEqual(sens[9], ref_sens[9], places=4)
        self.assertAlmostEqual(sens[10], ref_sens[10], places=4)
        self.assertAlmostEqual(sens[11], ref_sens[11], places=4)
        self.assertAlmostEqual(sens[12], ref_sens[12], places=4)
        self.assertAlmostEqual(sens[13], ref_sens[13], places=4)
        self.assertAlmostEqual(sens[14], ref_sens[14], places=4)
        self.assertAlmostEqual(sens[15], ref_sens[15], places=4)
        self.assertAlmostEqual(sens[16], ref_sens[16], places=4)

        # Test all pooled
        parameters = np.arange(1, 15) * 0.01
        score, sens = self.log_posterior9.evaluateS1(parameters)

        self.assertFalse(np.isinf(score))
        self.assertFalse(np.any(np.isinf(sens)))

    def test_get_log_likelihood(self):
        self.assertIsInstance(
            self.log_posterior1.get_log_likelihood(), chi.PopulationFilter)

    def test_get_log_prior(self):
        self.assertIsInstance(
            self.log_posterior1.get_log_prior(), pints.LogPrior)

    def test_get_id(self):
        ids = self.log_posterior1.get_id()
        self.assertEqual(len(ids), 22)
        self.assertIsNone(ids[0])
        self.assertIsNone(ids[1])
        self.assertIsNone(ids[2])
        self.assertIsNone(ids[3])
        self.assertEqual(ids[4], 'Sim. 1')
        self.assertEqual(ids[5], 'Sim. 1')
        self.assertEqual(ids[6], 'Sim. 2')
        self.assertEqual(ids[7], 'Sim. 2')
        self.assertEqual(ids[8], 'Sim. 3')
        self.assertEqual(ids[9], 'Sim. 3')
        self.assertEqual(ids[10], 'Sim. 1')
        self.assertEqual(ids[11], 'Sim. 1')
        self.assertEqual(ids[12], 'Sim. 1')
        self.assertEqual(ids[13], 'Sim. 1')
        self.assertEqual(ids[14], 'Sim. 2')
        self.assertEqual(ids[15], 'Sim. 2')
        self.assertEqual(ids[16], 'Sim. 2')
        self.assertEqual(ids[17], 'Sim. 2')
        self.assertEqual(ids[18], 'Sim. 3')
        self.assertEqual(ids[19], 'Sim. 3')
        self.assertEqual(ids[20], 'Sim. 3')
        self.assertEqual(ids[21], 'Sim. 3')

        ids = self.log_posterior1.get_id(unique=True)
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[0], 'Sim. 1')
        self.assertEqual(ids[1], 'Sim. 2')
        self.assertEqual(ids[2], 'Sim. 3')

        ids = self.log_posterior3.get_id()
        self.assertEqual(len(ids), 23)
        self.assertIsNone(ids[0])
        self.assertIsNone(ids[1])
        self.assertIsNone(ids[2])
        self.assertIsNone(ids[3])
        self.assertIsNone(ids[4])
        self.assertEqual(ids[5], 'Sim. 1')
        self.assertEqual(ids[6], 'Sim. 1')
        self.assertEqual(ids[7], 'Sim. 2')
        self.assertEqual(ids[8], 'Sim. 2')
        self.assertEqual(ids[9], 'Sim. 3')
        self.assertEqual(ids[10], 'Sim. 3')
        self.assertEqual(ids[11], 'Sim. 1')
        self.assertEqual(ids[12], 'Sim. 1')
        self.assertEqual(ids[13], 'Sim. 1')
        self.assertEqual(ids[14], 'Sim. 1')
        self.assertEqual(ids[15], 'Sim. 2')
        self.assertEqual(ids[16], 'Sim. 2')
        self.assertEqual(ids[17], 'Sim. 2')
        self.assertEqual(ids[18], 'Sim. 2')
        self.assertEqual(ids[19], 'Sim. 3')
        self.assertEqual(ids[20], 'Sim. 3')
        self.assertEqual(ids[21], 'Sim. 3')
        self.assertEqual(ids[22], 'Sim. 3')

        ids = self.log_posterior3.get_id(unique=True)
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[0], 'Sim. 1')
        self.assertEqual(ids[1], 'Sim. 2')
        self.assertEqual(ids[2], 'Sim. 3')

    def test_get_parameter_names(self):
        names = self.log_posterior1.get_parameter_names()
        self.assertEqual(len(names), 22)
        self.assertEqual(names[0], 'Log mean Initial count')
        self.assertEqual(names[1], 'Log std. Initial count')
        self.assertEqual(names[2], 'Log mean Growth rate')
        self.assertEqual(names[3], 'Log std. Growth rate')
        self.assertEqual(names[4], 'Initial count')
        self.assertEqual(names[5], 'Growth rate')
        self.assertEqual(names[6], 'Initial count')
        self.assertEqual(names[7], 'Growth rate')
        self.assertEqual(names[8], 'Initial count')
        self.assertEqual(names[9], 'Growth rate')
        self.assertEqual(names[10], 'Count Epsilon time 1')
        self.assertEqual(names[11], 'Count Epsilon time 2')
        self.assertEqual(names[12], 'Count Epsilon time 3')
        self.assertEqual(names[13], 'Count Epsilon time 4')
        self.assertEqual(names[14], 'Count Epsilon time 1')
        self.assertEqual(names[15], 'Count Epsilon time 2')
        self.assertEqual(names[16], 'Count Epsilon time 3')
        self.assertEqual(names[17], 'Count Epsilon time 4')
        self.assertEqual(names[18], 'Count Epsilon time 1')
        self.assertEqual(names[19], 'Count Epsilon time 2')
        self.assertEqual(names[20], 'Count Epsilon time 3')
        self.assertEqual(names[21], 'Count Epsilon time 4')

        names = self.log_posterior1.get_parameter_names(
            exclude_bottom_level=True)
        self.assertEqual(len(names), 4)
        self.assertEqual(names[0], 'Log mean Initial count')
        self.assertEqual(names[1], 'Log std. Initial count')
        self.assertEqual(names[2], 'Log mean Growth rate')
        self.assertEqual(names[3], 'Log std. Growth rate')

        names = self.log_posterior1.get_parameter_names(
            include_ids=True)
        self.assertEqual(len(names), 22)
        self.assertEqual(names[0], 'Log mean Initial count')
        self.assertEqual(names[1], 'Log std. Initial count')
        self.assertEqual(names[2], 'Log mean Growth rate')
        self.assertEqual(names[3], 'Log std. Growth rate')
        self.assertEqual(names[4], 'Sim. 1 Initial count')
        self.assertEqual(names[5], 'Sim. 1 Growth rate')
        self.assertEqual(names[6], 'Sim. 2 Initial count')
        self.assertEqual(names[7], 'Sim. 2 Growth rate')
        self.assertEqual(names[8], 'Sim. 3 Initial count')
        self.assertEqual(names[9], 'Sim. 3 Growth rate')
        self.assertEqual(names[10], 'Sim. 1 Count Epsilon time 1')
        self.assertEqual(names[11], 'Sim. 1 Count Epsilon time 2')
        self.assertEqual(names[12], 'Sim. 1 Count Epsilon time 3')
        self.assertEqual(names[13], 'Sim. 1 Count Epsilon time 4')
        self.assertEqual(names[14], 'Sim. 2 Count Epsilon time 1')
        self.assertEqual(names[15], 'Sim. 2 Count Epsilon time 2')
        self.assertEqual(names[16], 'Sim. 2 Count Epsilon time 3')
        self.assertEqual(names[17], 'Sim. 2 Count Epsilon time 4')
        self.assertEqual(names[18], 'Sim. 3 Count Epsilon time 1')
        self.assertEqual(names[19], 'Sim. 3 Count Epsilon time 2')
        self.assertEqual(names[20], 'Sim. 3 Count Epsilon time 3')
        self.assertEqual(names[21], 'Sim. 3 Count Epsilon time 4')

        names = self.log_posterior1.get_parameter_names(
            exclude_bottom_level=True, include_ids=True)
        self.assertEqual(len(names), 4)
        self.assertEqual(names[0], 'Log mean Initial count')
        self.assertEqual(names[1], 'Log std. Initial count')
        self.assertEqual(names[2], 'Log mean Growth rate')
        self.assertEqual(names[3], 'Log std. Growth rate')

        names = self.log_posterior3.get_parameter_names()
        self.assertEqual(len(names), 23)
        self.assertEqual(names[0], 'Log mean Initial count')
        self.assertEqual(names[1], 'Log std. Initial count')
        self.assertEqual(names[2], 'Log mean Growth rate')
        self.assertEqual(names[3], 'Log std. Growth rate')
        self.assertEqual(names[4], 'Sigma Count')
        self.assertEqual(names[5], 'Initial count')
        self.assertEqual(names[6], 'Growth rate')
        self.assertEqual(names[7], 'Initial count')
        self.assertEqual(names[8], 'Growth rate')
        self.assertEqual(names[9], 'Initial count')
        self.assertEqual(names[10], 'Growth rate')
        self.assertEqual(names[11], 'Count Epsilon time 1')
        self.assertEqual(names[12], 'Count Epsilon time 2')
        self.assertEqual(names[13], 'Count Epsilon time 3')
        self.assertEqual(names[14], 'Count Epsilon time 4')
        self.assertEqual(names[15], 'Count Epsilon time 1')
        self.assertEqual(names[16], 'Count Epsilon time 2')
        self.assertEqual(names[17], 'Count Epsilon time 3')
        self.assertEqual(names[18], 'Count Epsilon time 4')
        self.assertEqual(names[19], 'Count Epsilon time 1')
        self.assertEqual(names[20], 'Count Epsilon time 2')
        self.assertEqual(names[21], 'Count Epsilon time 3')
        self.assertEqual(names[22], 'Count Epsilon time 4')

        names = self.log_posterior3.get_parameter_names(
            exclude_bottom_level=True)
        self.assertEqual(len(names), 5)
        self.assertEqual(names[0], 'Log mean Initial count')
        self.assertEqual(names[1], 'Log std. Initial count')
        self.assertEqual(names[2], 'Log mean Growth rate')
        self.assertEqual(names[3], 'Log std. Growth rate')
        self.assertEqual(names[4], 'Sigma Count')

        names = self.log_posterior3.get_parameter_names(
            include_ids=True)
        self.assertEqual(len(names), 23)
        self.assertEqual(names[0], 'Log mean Initial count')
        self.assertEqual(names[1], 'Log std. Initial count')
        self.assertEqual(names[2], 'Log mean Growth rate')
        self.assertEqual(names[3], 'Log std. Growth rate')
        self.assertEqual(names[4], 'Sigma Count')
        self.assertEqual(names[5], 'Sim. 1 Initial count')
        self.assertEqual(names[6], 'Sim. 1 Growth rate')
        self.assertEqual(names[7], 'Sim. 2 Initial count')
        self.assertEqual(names[8], 'Sim. 2 Growth rate')
        self.assertEqual(names[9], 'Sim. 3 Initial count')
        self.assertEqual(names[10], 'Sim. 3 Growth rate')
        self.assertEqual(names[11], 'Sim. 1 Count Epsilon time 1')
        self.assertEqual(names[12], 'Sim. 1 Count Epsilon time 2')
        self.assertEqual(names[13], 'Sim. 1 Count Epsilon time 3')
        self.assertEqual(names[14], 'Sim. 1 Count Epsilon time 4')
        self.assertEqual(names[15], 'Sim. 2 Count Epsilon time 1')
        self.assertEqual(names[16], 'Sim. 2 Count Epsilon time 2')
        self.assertEqual(names[17], 'Sim. 2 Count Epsilon time 3')
        self.assertEqual(names[18], 'Sim. 2 Count Epsilon time 4')
        self.assertEqual(names[19], 'Sim. 3 Count Epsilon time 1')
        self.assertEqual(names[20], 'Sim. 3 Count Epsilon time 2')
        self.assertEqual(names[21], 'Sim. 3 Count Epsilon time 3')
        self.assertEqual(names[22], 'Sim. 3 Count Epsilon time 4')

        names = self.log_posterior3.get_parameter_names(
            exclude_bottom_level=True, include_ids=True)
        self.assertEqual(len(names), 5)
        self.assertEqual(names[0], 'Log mean Initial count')
        self.assertEqual(names[1], 'Log std. Initial count')
        self.assertEqual(names[2], 'Log mean Growth rate')
        self.assertEqual(names[3], 'Log std. Growth rate')
        self.assertEqual(names[4], 'Sigma Count')

        names = self.log_posterior9.get_parameter_names(
            exclude_bottom_level=True)
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], 'Pooled Initial count')
        self.assertEqual(names[1], 'Pooled Growth rate')

    def test_get_population_model(self):
        pop_model = self.log_posterior1.get_population_model()
        self.assertIsInstance(pop_model, chi.PopulationModel)

    def test_n_parameters(self):
        self.assertEqual(self.log_posterior1.n_parameters(), 22)
        self.assertEqual(
            self.log_posterior1.n_parameters(exclude_bottom_level=True), 4)
        self.assertEqual(self.log_posterior3.n_parameters(), 23)
        self.assertEqual(
            self.log_posterior3.n_parameters(exclude_bottom_level=True), 5)

    def test_n_samples(self):
        self.assertEqual(self.log_posterior1.n_samples(), 3)

    def test_sample_initial_parameters(self):
        # Bad input
        n_samples = 0
        with self.assertRaisesRegex(ValueError, 'The number of samples has'):
            self.log_posterior1.sample_initial_parameters(n_samples=n_samples)

        samples = self.log_posterior1.sample_initial_parameters()
        self.assertEqual(samples.shape, (1, 22))

        n_samples = 10
        samples = self.log_posterior1.sample_initial_parameters(
            n_samples=n_samples)
        self.assertEqual(samples.shape, (10, 22))

        seed = 3
        samples = self.log_posterior1.sample_initial_parameters(seed=seed)
        self.assertEqual(samples.shape, (1, 22))

        samples = self.log_posterior9.sample_initial_parameters(seed=seed)
        self.assertEqual(samples.shape, (1, 14))


if __name__ == '__main__':
    unittest.main()
