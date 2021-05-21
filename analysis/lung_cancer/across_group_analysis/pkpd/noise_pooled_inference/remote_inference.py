#
# This script infers the posterior distribution according to
# `find_posterior.ipynb` and stores the samples
# to files.
#

import numpy as np
import pandas as pd
import pints

import erlotinib as erlo

print('Setting up model ...')

# Define mechanistic model
path = erlo.ModelLibrary().erlotinib_tumour_growth_inhibition_model()
mechanistic_model = erlo.PharmacokineticModel(path)
mechanistic_model.set_administration(compartment='central', direct=False)
mechanistic_model.set_parameter_names(names={
    'central.drug_amount': 'Initial plasma drug amount in mg',
    'dose.drug_amount': 'Initial dose comp. drug amount in mg',
    'central.size': 'Volume of distribution in L',
    'dose.absorption_rate': 'Absorption rate in 1/d',
    'myokit.elimination_rate': 'Elimination rate in 1/d',
    'myokit.tumour_volume': 'Initial tumour volume in cm^3',
    'myokit.critical_volume': 'Critical volume in cm^3',
    'myokit.kappa': 'Potency in L/mg/day',
    'myokit.lambda': 'Exponential growth rate in 1/day'})
mechanistic_model.set_outputs([
    'central.drug_concentration',
    'myokit.tumour_volume'])
mechanistic_model.set_output_names({
    'central.drug_concentration': 'Plasma conc. in ng/mL',
    'myokit.tumour_volume': 'Tumour volume in cm^3'})

# Define error model
error_models = [
    erlo.ConstantAndMultiplicativeGaussianErrorModel(),  # Plasma conc.
    erlo.ConstantAndMultiplicativeGaussianErrorModel()]  # Tumour volume

# Define population model
population_models = [
    erlo.HeterogeneousModel(),  # Initial tumour volume
    erlo.PooledModel(),         # Volume of distribution
    erlo.HeterogeneousModel(),  # Critical volume
    erlo.PooledModel(),         # Elimination rate
    erlo.HeterogeneousModel(),  # Potency
    erlo.HeterogeneousModel(),  # Growth rate
    erlo.PooledModel(),         # Plasma conc. Sigma rel.
    erlo.PooledModel()]         # Tumour volume Sigma rel.

# Compose model and not identified parameters
problem = erlo.ProblemModellingController(mechanistic_model, error_models)
problem.fix_parameters({
    'Initial plasma drug amount in mg': 0,
    'Initial dose comp. drug amount in mg': 0,
    'Absorption rate in 1/d': 55,  # Data not rich enough (value from
                                   # Eigenmann et. al.)
    'Plasma conc. in ng/mL Sigma base': 0.001,   # Regularisation
    'Tumour volume in cm^3 Sigma base': 0.001})  # Regularisation
problem.set_population_model(population_models)

# Import data
lib = erlo.DataLibrary()
data = pd.concat((
    lib.lung_cancer_low_erlotinib_dose_group(),
    lib.lung_cancer_medium_erlotinib_dose_group(),
    lib.lung_cancer_high_erlotinib_dose_group()))

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
        mean=0.1, sd=0.05, a=0, b=np.inf),   # Volume of distribution
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
        mean=10, sd=5, a=0, b=np.inf),       # Pooled Elimination rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Potency
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
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Pooled Plasma conc. Sigma rel.
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf)]    # Pooled Tumour volume Sigma rel.

# Define log-posterior
problem.set_data(
    data,
    output_biomarker_dict={
        'Plasma conc. in ng/mL': 'Plasma concentration',
        'Tumour volume in cm^3': 'Tumour volume'})
problem.set_log_prior(log_priors)
log_posterior = problem.get_log_posterior()

print('Model is set! \n')

print('Setting up inference ...')

# Setup sampling controller
n_chains = 5
n_iterations = 30000
logging_steps = 100
sampler = pints.NoUTurnMCMC
max_tree_depth = 10
n_adaption_steps = 2000
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
    s.set_max_tree_depth(max_tree_depth)
    s.set_number_adaption_steps(n_adaption_steps)

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
