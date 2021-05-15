#
# This script infers the posterior distribution according to
# `find_posterior.ipynb` and stores the samples
# to files.
#

import numpy as np
import pints

import erlotinib as erlo

print('Setting up model ...')

# Define mechanistic model
path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch_reparametrised()
mechanistic_model = erlo.PharmacodynamicModel(path)
mechanistic_model.set_parameter_names(names={
    'myokit.tumour_volume': 'Tumour volume in cm^3',
    'myokit.critical_volume': 'Critical volume in cm^3',
    'myokit.drug_concentration': 'Drug concentration in mg/L',
    'myokit.kappa': 'Potency in L/mg/day',
    'myokit.lambda': 'Exponential growth rate in 1/day'})
mechanistic_model.set_output_names({
    'myokit.tumour_volume': 'Tumour volume'})

# Define error model
error_model = erlo.ConstantAndMultiplicativeGaussianErrorModel()

# Define population model
population_models = [
    erlo.HeterogeneousModel(),  # Initial tumour volume
    erlo.HeterogeneousModel(),  # Critical volume
    erlo.HeterogeneousModel(),  # Growth rate
    erlo.PooledModel(),         # Sigma base
    erlo.PooledModel()]         # Sigma rel.

# Compose model and not identified parameters
problem = erlo.ProblemModellingController(
    mechanistic_model, error_model)
problem.fix_parameters({
    'Drug concentration in mg/L': 0,
    'Potency in L/mg/day': 0})
problem.set_population_model(population_models)

# Import data
data = erlo.DataLibrary().lung_cancer_control_group()

# Define prior distribution
log_priors = [
    pints.TruncatedGaussianLogPrior(
        mean=0.25, sd=0.1, a=0, b=np.inf),   # Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.25, sd=0.1, a=0, b=np.inf),   # Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.25, sd=0.1, a=0, b=np.inf),   # Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.25, sd=0.1, a=0, b=np.inf),   # Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.25, sd=0.1, a=0, b=np.inf),   # Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.25, sd=0.1, a=0, b=np.inf),   # Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.25, sd=0.1, a=0, b=np.inf),   # Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.25, sd=0.1, a=0, b=np.inf),   # Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Critical volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Critical volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Critical volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Critical volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Critical volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Critical volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Critical volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Critical volume
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0, sd=0.1, a=0, b=np.inf),      # Pooled Sigma base
    pints.TruncatedGaussianLogPrior(
        mean=0.5, sd=0.2, a=0, b=np.inf)]    # Pooled Sigma rel.
log_prior = pints.ComposedLogPrior(*log_priors)

# Define log-posterior
problem.set_data(data)
problem.set_log_prior(log_priors)
log_posterior = problem.get_log_posterior()

print('Model is set! \n')

print('Setting up inference ...')

# Setup sampling controller
n_chains = 5
n_iterations = 30000
logging_steps = 20
sampler = pints.HamiltonianMCMC
hyperparameters = [15, 0.1]
transform = pints.LogTransformation(
    n_parameters=problem.get_n_parameters())

# Sample initial parameters using erlotinib
np.random.seed(2)
initial_parameters = erlo.SamplingController(
    log_posterior)._initial_params[0, ...]

controller = pints.MCMCController(
    log_pdf=log_posterior,
    chains=n_chains,
    x0=initial_parameters,
    transform=transform,
    method=sampler)

# Set sampler hyperparameters
for s in controller.samplers():
    s.set_hyper_parameters(hyperparameters)

# Run in parallel and write to disk
controller.set_parallel(True)
controller.set_max_iterations(n_iterations)
controller.set_chain_filename('chain_parameters.csv')
controller.set_log_to_file(filename='chain_log.csv', csv=True)
controller.set_log_to_screen(True)
controller.set_chain_storage(store_in_memory=False)
controller.set_log_pdf_storage(store_in_memory=False)
controller.set_log_interval(iters=logging_steps)

print('Inference is set! \n')

print('Running inference ...')

controller.run()

print('Inference completed!')
print('Script terminated!')
