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
    erlo.LogNormalModel(),  # Initial tumour volume
    erlo.PooledModel(),     # Volume of distribution
    erlo.LogNormalModel(),  # Critical volume
    erlo.PooledModel(),     # Elimination rate
    erlo.LogNormalModel(),  # Potency
    erlo.LogNormalModel(),  # Growth rate
    erlo.PooledModel(),     # Plasma conc. Sigma rel.
    erlo.PooledModel()]     # Tumour volume Sigma rel.

# Compose model and not identified parameters
problem = erlo.ProblemModellingController(mechanistic_model, error_models)
problem.fix_parameters({
    'Initial plasma drug amount in mg': 0,
    'Initial dose comp. drug amount in mg': 0,
    'Absorption rate in 1/d': 55,  # Data not rich enough (value from
    # Eigenmann et. al.)
    'Plasma conc. in ng/mL Sigma base': 0.001,  # Regularises Mult. Error
    'Tumour volume in cm^3 Sigma base': 0.001})  # Regularises Mult. Error
problem.set_population_model(population_models)

# Import data
data = erlo.DataLibrary().lung_cancer_high_erlotinib_dose_group()

# Define prior distribution
log_priors = [
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Mean Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Std. Initial tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.05, a=0, b=np.inf),   # Volume of distribution
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Mean Critical tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=1, sd=1, a=0, b=np.inf),        # Std. Critical tumour volume
    pints.TruncatedGaussianLogPrior(
        mean=10, sd=5, a=0, b=np.inf),       # Pooled Elimination rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Mean Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=0.1, a=0, b=np.inf),    # Std. Potency
    pints.TruncatedGaussianLogPrior(
        mean=0.4, sd=1, a=0, b=np.inf),      # Mean Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.2, sd=1, a=0, b=np.inf),      # Std. Growth rate
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=1, a=0, b=np.inf),      # Pooled Plasma conc. Sigma rel.
    pints.TruncatedGaussianLogPrior(
        mean=0.1, sd=1, a=0, b=np.inf)]      # Pooled Tumour volume Sigma rel.

# Define log-posterior
problem.set_data(data, output_biomarker_dict={
    'Plasma conc. in ng/mL': 'Plasma concentration',
    'Tumour volume in cm^3': 'Tumour volume'})
problem.set_log_prior(log_priors)
log_posterior = problem.get_log_posterior()

print('Model is set! \n')

print('Setting up inference ...')

# Setup sampling controller
n_chains = 5
n_iterations = 100000
logging_steps = 20
sampler = pints.HamiltonianMCMC
hyperparameters = [15, 0.1]
transform = pints.LogTransformation(
    n_parameters=problem.get_n_parameters())

# Sample initial parameters using erlotinib
n_parameters = problem.get_n_parameters()
initial_parameters = np.empty(shape=(n_chains, n_parameters))
# temp = erlo.SamplingController(log_posterior)
# np.random.seed(2)
# temp.set_n_runs(1)
initial_parameters[0] = [
    1.87132098e-01, 2.82970456e-02, 6.02320141e-03, 2.41119987e-02,
    4.37185817e-02, 8.57529126e-02, 3.08907759e-02, 1.01759514e-01,
    1.26414735e-01, 1.20036324e+00, 9.17178607e-01, 8.68124415e-01,
    9.19457434e-01, 1.00714024e+00, 1.03497601e+00, 1.01785518e+00,
    1.60817076e-01, 1.05835469e+01, 1.14622759e-01, 1.17710257e-01,
    2.30280615e-02, 3.31698335e-01, 2.27073145e-01, 9.03274595e-02,
    1.57435261e-01, 1.23231805e-01, 1.55834092e+00, 1.58828487e+00,
    4.91547492e+00, 2.67814681e+00, 3.81373189e+00, 1.63308187e+00,
    2.19516382e+00, 9.07937591e-01, 1.72438967e+00, 1.86296991e-01]
initial_parameters[1] = [
    0.21512032,  0.14175314,  0.45775667,  0.14900228,  0.09238764,
    0.24805573,  0.19618934,  0.14261231,  0.13699292,  0.63584258,
    0.34395343,  0.03720201,  0.09120889,  0.08102501,  0.15522524,
    0.47856588,  0.7600611, 10.4378958,  0.09575491,  0.06933375,
    0.09265091,  0.08931292,  0.09188532,  0.07577038,  0.08962887,
    0.01479278,  3.51127659,  4.01797434,  2.53964693,  3.20030317,
    2.03121468,  2.20440555,  2.68893385,  0.65765381,  0.71961887,
    0.44467003]
initial_parameters[2] = [
    9.44283684e-02, 5.49192511e-02, 4.03431932e-02, 2.80414802e-01,
    1.60482892e-01, 9.88918248e-02, 9.32223450e-02, 1.58518091e-01,
    8.13893581e-02, 5.01305244e-01, 1.37942437e+00, 1.10219862e+00,
    4.58439587e-01, 5.89696007e-01, 1.47261657e+00, 1.35998912e+00,
    8.30677259e-01, 9.46970881e+00, 1.20161420e-02, 1.07701204e-03,
    1.50267701e-02, 8.19836784e-03, 2.19651470e-03, 1.76930009e-03,
    1.95043172e-02, 6.49092695e-02, 2.81416298e+00, 2.27657540e+00,
    1.93756417e+00, 2.76613625e+00, 2.67194372e+00, 2.66218836e+00,
    2.48329284e+00, 3.36554251e-01, 1.06789251e+00, 9.82762430e-01]
initial_parameters[3] = [
    0.01969529, 0.01204187, 0.11378868, 0.06713651, 0.0953615,
    0.03826852, 0.15049666, 0.20016558, 0.17318082, 2.0191619,
    1.74126935, 1.56990194, 3.85154595, 1.22365524, 5.2322022,
    2.45013088, 2.04767329, 3.99330179, 0.08192025, 0.08404696,
    0.09376439, 0.09414636, 0.08940114, 0.06438911, 0.09056648,
    0.01166759, 1.61986645, 2.04204267, 0.15171891, 0.33849811,
    0.1599799, 0.62809439, 0.9321051, 1.38460764, 0.06936539,
    0.2560933]
initial_parameters[4] = [
    0.03056112,  0.02670273,  0.0786333,  0.02837566,  0.16250397,
    0.07193757,  0.13591132,  0.19728649,  0.04456212,  0.5287136,
    0.12746558,  0.39775086,  2.1218405,  0.31951921,  0.50586701,
    0.7253859,  1.52017084, 10.05361278,  0.11278554,  0.05165671,
    0.10972977,  0.26674645,  0.1317265,  0.11343309,  0.12143003,
    0.11177966,  3.15109759,  2.38409123,  2.2058279,  1.0492956,
    2.29674011,  1.31438614,  1.88337293,  0.94426826,  0.89220805,
    0.6237867]

np.random.seed(1)
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
