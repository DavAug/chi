#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pints
import pints.toy

import erlotinib as erlo


class TestHierarchicalLogLikelihood(unittest.TestCase):
    """
    Tests the erlotinib.HierarchicalLogLikelihood class.
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
        path = erlo.ModelLibrary().one_compartment_pk_model()
        cls.model = erlo.PharmacokineticModel(path)
        cls.model.set_administration('central', direct=False)
        cls.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        cls.error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2

        # Create log-likelihoods
        cls.log_likelihoods = [
            erlo.LogLikelihood(
                cls.model, cls.error_models, cls.observations, cls.times),
            erlo.LogLikelihood(
                cls.model, cls.error_models, cls.observations, cls.times)]

        # Create population models
        cls.population_models = [
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.LogNormalModel(),
            erlo.PooledModel(),
            erlo.HeterogeneousModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel()]

        cls.hierarchical_model = erlo.HierarchicalLogLikelihood(
            cls.log_likelihoods, cls.population_models)

    def test_bad_instantiation(self):
        # Log-likelihoods are not pints.LogPDF
        log_likelihoods = ['bad', 'type']
        with self.assertRaisesRegex(ValueError, 'The log-likelihoods have'):
            erlo.HierarchicalLogLikelihood(
                log_likelihoods, self.population_models)

        # Log-likelihoods are defined on different parameter spaces
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        log_likelihoods = [
            self.log_likelihoods[0],
            erlo.LogLikelihood(
                model, error_models, self.observations[0], self.times[0])]

        with self.assertRaisesRegex(ValueError, 'The number of parameters'):
            erlo.HierarchicalLogLikelihood(
                log_likelihoods, self.population_models)

        # The log-likelihood parameter names differ
        model.set_outputs(['central.drug_concentration', 'dose.drug_amount'])
        error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2
        log_likelihoods = [
            self.log_likelihoods[0],
            erlo.LogLikelihood(
                model, error_models, self.observations, self.times)]

        with self.assertRaisesRegex(ValueError, 'The parameter names'):
            erlo.HierarchicalLogLikelihood(
                log_likelihoods, self.population_models)

        # Population models are not erlotinib.PopulationModel
        population_models = ['bad', 'type'] + ['match dimension'] * 7
        with self.assertRaisesRegex(ValueError, 'The population models have'):
            erlo.HierarchicalLogLikelihood(
                self.log_likelihoods, population_models)

        # Not all parameters of the likelihoods are assigned to a pop model
        population_models = [
            erlo.PooledModel(),
            erlo.PooledModel()]
        with self.assertRaisesRegex(ValueError, 'Wrong number of population'):
            erlo.HierarchicalLogLikelihood(
                self.log_likelihoods, population_models)

    # def test_call(self):
    #     # Create reference model
    #     pooled_log_pdf = pints.PooledLogPDF(
    #         self.log_likelihoods, pooled=[True]*6)

    #     # Test case I.1
    #     parameters = [1, 1, 1, 1, 1, 1]
    #     score = pooled_log_pdf(parameters)

    #     self.assertEqual(self.hierarchical_model(parameters), score)

    #     # Test case I.2
    #     parameters = [10, 1, 0.1, 1, 3, 1]
    #     score = pooled_log_pdf(parameters)

    #     self.assertEqual(self.hierarchical_model(parameters), score)

    #     # Test case II.1: non-pooled model
    #     pop_models = [
    #         erlo.HeterogeneousModel()] \
    #         * self.n_individual_params
    #     likelihood = erlo.HierarchicalLogLikelihood(
    #         self.log_likelihoods, pop_models)

    #     # Compute score from individual likelihoods
    #     parameters = [1, 1, 1, 1, 1, 1]
    #     score = 0
    #     for ll in self.log_likelihoods:
    #         score += ll(parameters)

    #     n_parameters = 6
    #     n_ids = 8
    #     parameters = [1] * n_parameters * n_ids
    #     self.assertEqual(likelihood(parameters), score)

    #     # Test case II.2
    #     # Compute score from individual likelihoods
    #     parameters = [10, 1, 0.1, 1, 3, 1]
    #     score = 0
    #     for ll in self.log_likelihoods:
    #         score += ll(parameters)

    #     n_ids = 8
    #     parameters = \
    #         [parameters[0]] * n_ids + \
    #         [parameters[1]] * n_ids + \
    #         [parameters[2]] * n_ids + \
    #         [parameters[3]] * n_ids + \
    #         [parameters[4]] * n_ids + \
    #         [parameters[5]] * n_ids
    #     self.assertEqual(likelihood(parameters), score)

    #     # Test case III.1: Non-trivial population model.
    #     pop_models = \
    #         [erlo.LogNormalModel()] + \
    #         [erlo.PooledModel()] \
    #         * (self.n_individual_params - 1)
    #     likelihood = erlo.HierarchicalLogLikelihood(
    #         self.log_likelihoods, pop_models)

    #     ref_likelihood_part_one = erlo.LogNormalModel()
    #     ref_likelihood_part_two = pints.PooledLogPDF(
    #         self.log_likelihoods, pooled=[False] + [True]*5)

    #     parameters = [10, 1, 0.1, 1, 3, 1]
    #     pop_params = [1, 1]

    #     n_ids = 8
    #     parameters = \
    #         [parameters[0]] * n_ids + \
    #         pop_params + \
    #         [parameters[1]] + \
    #         [parameters[2]] + \
    #         [parameters[3]] + \
    #         [parameters[4]] + \
    #         [parameters[5]]

    #     score = \
    #         ref_likelihood_part_one.compute_log_likelihood(
    #             pop_params, parameters[:n_ids]) + \
    #         ref_likelihood_part_two(parameters[:n_ids] + parameters[n_ids+2:])

    #     self.assertNotEqual(score, -np.inf)
    #     self.assertAlmostEqual(likelihood(parameters), score)

    #     # Test case III.2: Returns -np.inf if individuals are far away from
    #     # pop distribution
    #     parameters = [100000, 1, 0.1, 1, 3, 1]
    #     pop_params = [0, 0.00001]

    #     n_ids = 8
    #     parameters = \
    #         [parameters[0]] * n_ids + \
    #         pop_params + \
    #         [parameters[1]] + \
    #         [parameters[2]] + \
    #         [parameters[3]] + \
    #         [parameters[4]] + \
    #         [parameters[5]]

    #     self.assertEqual(likelihood(parameters), -np.inf)

    def test_n_parameters(self):
        # 9 individual parameters, from which 1 is modelled heterogeneously,
        # 1 log-normally and the rest is pooled
        # And there are 2 individuals
        n_parameters = 2 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.assertEqual(
            self.hierarchical_model.n_parameters(), n_parameters)


class TestLogLikelihood(unittest.TestCase):
    """
    Test the erlotinib.LogLikelihood class.
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
        path = erlo.ModelLibrary().one_compartment_pk_model()
        cls.model = erlo.PharmacokineticModel(path)
        cls.model.set_administration('central', direct=False)
        cls.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        cls.error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2

        # Create log-likelihood
        cls.log_likelihood = erlo.LogLikelihood(
            cls.model, cls.error_models, cls.observations, cls.times)

    def test_bad_instantiation(self):
        # Mechantic model has wrong type
        mechanistic_model = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The mechanistic model'):
            erlo.LogLikelihood(
                mechanistic_model, self.error_models, self.observations,
                self.times)

        # Wrong number of error models
        error_models = ['There', 'are', 'only two outputs']
        with self.assertRaisesRegex(ValueError, 'One error model has to'):
            erlo.LogLikelihood(
                self.model, error_models, self.observations, self.times)

        # Wrong number of error models
        error_models = ['Wrong', 'type']
        with self.assertRaisesRegex(TypeError, 'The error models have to'):
            erlo.LogLikelihood(
                self.model, error_models, self.observations, self.times)

        # Wrong length of observations
        observations = [['There'], ['are'], ['only two outputs']]
        with self.assertRaisesRegex(ValueError, 'The observations have'):
            erlo.LogLikelihood(
                self.model, self.error_models, observations, self.times)

        # Wrong length of times
        times = [['There'], ['are'], ['only two outputs']]
        with self.assertRaisesRegex(ValueError, 'The times have the wrong'):
            erlo.LogLikelihood(
                self.model, self.error_models, self.observations, times)

        # Observations and times don't match
        observations = [[1, 2], [1, 2]]  # Times have 4 and 3
        with self.assertRaisesRegex(ValueError, 'The observations and times'):
            erlo.LogLikelihood(
                self.model, self.error_models, observations, self.times)

        # Observations or times have some weird higher dimensional structure
        observations = [[[1, 2], [1, 2]], [1, 2, 3, 4]]
        times = [[[1, 2], [1, 2]], [1, 2, 3, 4]]
        with self.assertRaisesRegex(ValueError, 'The observations for each'):
            erlo.LogLikelihood(
                self.model, self.error_models, observations, times)

    def test_call(self):
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

        self.assertAlmostEqual(score, ref_score)

        # Test case II: Compute reference score with two likelihoods
        parameters = [9, 8, 7, 6, 5, 4, 3, 2, 1]

        times = self.times[0]
        observations = self.observations[0]
        self.model.set_outputs(['central.drug_amount'])
        error_model = self.error_models[0]
        log_likelihood = erlo.LogLikelihood(
            self.model, error_model, observations, times)
        ref_score_1 = log_likelihood(parameters[:7])

        times = self.times[1]
        observations = self.observations[1]
        self.model.set_outputs(['dose.drug_amount'])
        error_model = self.error_models[1]
        log_likelihood = erlo.LogLikelihood(
            self.model, error_model, observations, times)
        ref_score_2 = log_likelihood(parameters[:5] + parameters[7:9])

        ref_score = ref_score_1 + ref_score_2
        score = self.log_likelihood(parameters)

        self.assertAlmostEqual(score, ref_score)

        # Reset number of outputs
        self.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])

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

        # Test case II: Set ID with space
        _id = 'some ID'
        with self.assertRaisesRegex(ValueError, 'The ID cannot contain'):
            self.log_likelihood.set_id(_id)

        # Test case II: Set ID with space
        _id = 'some-ID'
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
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 2)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)
        self.assertIsInstance(error_models[1], erlo.ErrorModel)

        # Test case II: some fixed parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'central.drug_amount': 1,
            'dose.absorption_rate': 1})
        submodels = self.log_likelihood.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 2)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)
        self.assertIsInstance(error_models[1], erlo.ErrorModel)

        # Unfix parameter
        self.log_likelihood.fix_parameters({
            'central.drug_amount': None,
            'dose.absorption_rate': None})

    def test_n_parameters(self):
        # Test case I:
        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 9)

        # Test case II:
        times = self.times[0]
        observations = self.observations[0]
        self.model.set_outputs(['central.drug_amount'])
        error_model = self.error_models[0]
        log_likelihood = erlo.LogLikelihood(
            self.model, error_model, observations, times)

        n_parameters = log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 7)

        # Reset number of outputs
        self.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])


class TestLogPosterior(unittest.TestCase):
    """
    Tests the erlotinib.LogPosterior class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test dataset
        times = [0, 1, 2, 3]
        values = [10, 11, 12, 13]

        # Create test model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem = erlo.InverseProblem(model, times, values)
        log_likelihood = pints.GaussianLogLikelihood(problem)
        log_prior = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1))
        cls.log_posterior = erlo.LogPosterior(log_likelihood, log_prior)

    def test_set_id_bad_input(self):
        # Provide the wrong number of labels
        labels = ['wrong', 'number']

        with self.assertRaisesRegex(ValueError, 'If a list of IDs is'):
            self.log_posterior.set_id(labels)

    def test_set_get_id(self):
        # Check default
        self.assertIsNone(self.log_posterior.get_id())

        # Set some id
        index = 'Test id'
        self.log_posterior.set_id(index)

        self.assertEqual(self.log_posterior.get_id(), index)

        # Set an ID for each parameter individually
        labels = [
            'ID 1', 'ID 2', 'ID 3', 'ID 4', 'ID 5', 'ID 6']
        self.log_posterior.set_id(labels)

        self.assertEqual(self.log_posterior.get_id(), labels)

    def test_set_get_parameter_names(self):
        # Check default
        default = [
            'Param 1', 'Param 2', 'Param 3', 'Param 4', 'Param 5', 'Param 6']
        self.assertEqual(self.log_posterior.get_parameter_names(), default)

        # Set some parameter names
        names = ['A', 'B', 'C', 'D', 'E', 'F']
        self.log_posterior.set_parameter_names(names)

        self.assertEqual(self.log_posterior.get_parameter_names(), names)

    def test_set_parameter_names_bad_input(self):
        # Number of names does not match the number of parameters
        names = ['too', 'few', 'params']

        with self.assertRaisesRegex(ValueError, 'The list of param'):
            self.log_posterior.set_parameter_names(names)


class TestReducedLogPDF(unittest.TestCase):
    """
    Tests the erlotinib.ReducedLogPDF class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        times = [1, 2, 3, 4, 5]
        values = [1, 2, 3, 4, 5]

        # Set up inverse problem
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem = erlo.InverseProblem(model, times, values)
        cls.log_likelihood = pints.GaussianLogLikelihood(problem)
        cls.mask = [True, False, False, True, False, True]
        cls.values = [11, 12, 13]
        cls.reduced_log_pdf = erlo.ReducedLogPDF(
            cls.log_likelihood, cls.mask, cls.values)

    def test_bad_input(self):
        # Wrong log-pdf
        log_pdf = 'Bad type'

        with self.assertRaisesRegex(ValueError, 'The log-pdf has to'):
            erlo.ReducedLogPDF(log_pdf, self.mask, self.values)

        # Mask is not as long as the number of parameyers
        mask = [True, True]

        with self.assertRaisesRegex(ValueError, 'Length of mask has to'):
            erlo.ReducedLogPDF(self.log_likelihood, mask, self.values)

        # Mask is not boolean
        mask = ['yes', 'no', 'yes', 'yes', 'yes', 'yes']

        with self.assertRaisesRegex(ValueError, 'Mask has to be a'):
            erlo.ReducedLogPDF(self.log_likelihood, mask, self.values)

        # There are not as many input values as fixed parameters
        values = [1]

        with self.assertRaisesRegex(ValueError, 'There have to be'):
            erlo.ReducedLogPDF(self.log_likelihood, self.mask, values)

    def test_call(self):
        parameters = np.array([11, 1, 1, 12, 1, 13])
        reduced_params = parameters[~np.array(self.mask)]

        self.assertEqual(
            self.reduced_log_pdf(reduced_params),
            self.log_likelihood(parameters))

    def test_n_parameters(self):
        before = self.log_likelihood.n_parameters()
        n_fixed = np.sum(self.mask)

        self.assertEqual(
            self.reduced_log_pdf.n_parameters(), before - n_fixed)


if __name__ == '__main__':
    unittest.main()
