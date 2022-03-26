import argparse
from ftplib import error_reply


# Set up argument parsing, so plotting and exports can be disabled for
# testing.
parser = argparse.ArgumentParser(
    description='Run example scripts for chi docs.',
)
parser.add_argument(
    '--test',
    action='store_true',
    help='Run testing version of script which ignores plotting.',)

# Parse!
args = parser.parse_args()

# Run testing verion of script
if args.test:
    ## 1. Start example one
    import chi.library

    # Define 1-compartment pharmacokinetic model
    mechanistic_model = chi.library.ModelLibrary().one_compartment_pk_model()

    # Define parameters
    parameters = [
        10,  # Initial drug amount
        2,   # Volume of the compartment
        1,   # Elimination rate
    ]

    # Define evaluation times
    times = [0, 0.2, 0.5, 0.6, 1]

    # Simulate the model
    result = mechanistic_model.simulate(parameters, times)
    ## end 1.

    ## Start 2.
    import numpy as np
    import plotly.graph_objects as go

    # Set administration and dosing regimen
    mechanistic_model.set_administration(
        compartment='central', amount_var='drug_amount')
    mechanistic_model.set_dosing_regimen(dose=1, period=0.5)

    # Simulate the model
    times = np.linspace(start=0, stop=10, num=1000)
    result = mechanistic_model.simulate(parameters, times)

    # Visualise results
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=result[0],
        mode='lines'
    ))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Drug concentration',
        template='plotly_white'
    )
    # fig.show()
    ## End 2.

    ## Start 3.
    import chi

    # Define error model and error model parameters
    error_model = chi.GaussianErrorModel()
    parameters = [0.2]  # Sigma

    # Down sample times and mechanistic model evaluations
    measurement_times = times[::25]
    corresponding_evaluations = result[0, ::25]

    # Simulate measurements
    measurements = error_model.sample(
        parameters, model_output=corresponding_evaluations, seed=1)[:, 0]

    # Visualise results
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=result[0],
        mode='lines',
        line_color='black',
        name='Mechanistic model'
    ))
    fig.add_trace(go.Scatter(
        x=measurement_times,
        y=measurements,
        mode='markers',
        name='Measurements'
    ))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Drug concentration',
        template='plotly_white'
    )
    # fig.show()
    # End 3.

    # Start 4.
    # Define log-likelihood
    log_likelihood = chi.LogLikelihood(
        mechanistic_model,
        error_model,
        observations=measurements,
        times=measurement_times
    )

    # Evaluate log-likelihood for made-up parameters
    madeup_parameters = [2, 7, 1.4, 3]
    score_1 = log_likelihood(madeup_parameters)

    # Evaluate log-likelihood for data-generating parameters
    true_parameters = [10, 2, 1, 0.2]
    score_2 = log_likelihood(true_parameters)
    # End 4.

    # Start 5.
    import pints

    # Run optimisation
    initial_parameters = [9, 3, 5, 1]
    parameters_mle, score = pints.optimise(
        log_likelihood, initial_parameters, method=pints.CMAES)
    # End 5.

    parameters_mle = [10.26564936, 2.01524534, 1.00148417, 0.18456719]
    # Start 6.
    mle_output = mechanistic_model.simulate(parameters_mle, times)
    fig.add_trace(go.Scatter(
        x=times,
        y=mle_output[0],
        mode='lines',
        line_dash='dash',
        name='Inferred model'
    ))
    # fig.show()
    # End 6.

    # Start 7.
    # Define log-posterior
    log_prior = pints.ComposedLogPrior(
        pints.UniformLogPrior(0, 20),  # Initial drug amount
        pints.UniformLogPrior(0, 20),  # Compartment volume
        pints.UniformLogPrior(0, 20),  # Elimination rate
        pints.UniformLogPrior(0, 20)   # Sigma
    )
    log_posterior = chi.LogPosterior(log_likelihood, log_prior)

    # Evaluate log-posterior
    score_1 = log_posterior(madeup_parameters)
    score_2 = log_posterior(true_parameters)
    # End 7.

    # Start 8.
    # Run inference
    controller = chi.SamplingController(log_posterior)
    n_iterations = 5000
    posterior_samples = controller.run(n_iterations)
    # End 8.

    # Start 9.
    from plotly.subplots import make_subplots

    # Discard warmup iterations
    warmup = 3000  # This number is not arbitrary but was carefully chosen
    posterior_samples = posterior_samples.sel(draw=slice(warmup, n_iterations))

    # Visualise posteriors
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True)
    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            x=posterior_samples['central.drug_amount'].values.flatten(),
            histnorm='probability',
            showlegend=False
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[true_parameters[0]]*2,
            y=[0, 0.04],
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            x=posterior_samples['central.size'].values.flatten(),
            histnorm='probability',
            showlegend=False
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[true_parameters[1]]*2,
            y=[0, 0.04],
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=1,
        col=2
    )

    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            x=posterior_samples['myokit.elimination_rate'].values.flatten(),
            histnorm='probability',
            showlegend=False
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[true_parameters[2]]*2,
            y=[0, 0.04],
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            x=posterior_samples['Sigma'].values.flatten(),
            histnorm='probability',
            showlegend=False
        ),
        row=2,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[true_parameters[3]]*2,
            y=[0, 0.04],
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=2,
        col=2
    )

    fig.update_layout(
        xaxis_title='Initial drug amount',
        xaxis2_title='Compartment volume',
        xaxis3_title='Elimination rate',
        xaxis4_title='Sigma',
        yaxis_title='Probability',
        yaxis3_title='Probability',
        template='plotly_white'
    )
    # fig.show()
    # End 9.

    # Exit script
    exit()

# Run non-testing version
import os

import chi
import chi.library
import numpy as np
import plotly.graph_objects as go

mechanistic_model = chi.library.ModelLibrary().one_compartment_pk_model()
mechanistic_model.set_administration(compartment='central')
mechanistic_model.set_dosing_regimen(dose=5, start=3, period=1, duration=0.3)
error_model = chi.GaussianErrorModel()

times = np.linspace(start=0, stop=10, num=1000)
measurement_times = times[::25]

directory = os.path.dirname(os.path.dirname(__file__))

# Start 1
import plotly.colors

# Set different dosing regimen (for illustration)
mechanistic_model.set_dosing_regimen(dose=5, start=3, period=1, duration=0.3)

# Define patient parameters
parameters_patient_1 = [
    10,  # Initial drug amount
    2,   # Volume of the compartment
    1,   # Elimination rate
]
parameters_patient_2 = [
    10,  # Initial drug amount
    2,   # Volume of the compartment
    0.5, # Elimination rate
]
parameters_patient_3 = [
    10,  # Initial drug amount
    2,   # Volume of the compartment
    0.8, # Elimination rate
]
error_model_params = [0.4]

# Measure drug concentrations of patients
n_times = 1000
times = np.linspace(start=0, stop=10, num=n_times)
measurement_times = times[::50]
result_1 = mechanistic_model.simulate(parameters_patient_1, times)[0]
measurements_1 = error_model.sample(
    error_model_params, result_1[::50], seed=41)[:, 0]
result_2 = mechanistic_model.simulate(parameters_patient_2, times)[0]
measurements_2 = error_model.sample(
    error_model_params, result_2[::50], seed=42)[:, 0]
result_3 = mechanistic_model.simulate(parameters_patient_3, times)[0]
measurements_3 = error_model.sample(
    error_model_params, result_3[::50], seed=43)[:, 0]

# Visualise results
fig = go.Figure()
colors = plotly.colors.qualitative.Plotly
fig.add_trace(go.Scatter(
    x=times,
    y=result_1,
    mode='lines',
    line_color=colors[0],
    name='Model patient 1'
))
fig.add_trace(go.Scatter(
    x=times,
    y=result_2,
    mode='lines',
    line_color=colors[1],
    name='Model patient 2'
))
fig.add_trace(go.Scatter(
    x=times,
    y=result_3,
    mode='lines',
    line_color=colors[2],
    name='Model patient 3'
))
fig.add_trace(go.Scatter(
    x=measurement_times,
    y=measurements_1,
    mode='markers',
    marker_color=colors[0],
    name='Meas. patient 1'
))
fig.add_trace(go.Scatter(
    x=measurement_times,
    y=measurements_2,
    mode='markers',
    marker_color=colors[1],
    name='Meas. patient 2'
))
fig.add_trace(go.Scatter(
    x=measurement_times,
    y=measurements_3,
    mode='markers',
    marker_color=colors[2],
    name='Meas. patient 3'
))
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Drug concentration',
    template='plotly_white'
)
fig.show()
# End 1.
fig.write_html(directory + '/images/1_simulation_6.html')


from plotly.subplots import make_subplots
# Start 2.
# Define population model
population_model = chi.ComposedPopulationModel([
    chi.PooledModel(dim_names=['Initial drug amount']),
    chi.PooledModel(dim_names=['Compartment volume']),
    chi.LogNormalModel(dim_names=['Elimination rate']),
    chi.PooledModel(dim_names=['Noise'])
])

# Define population parameters
population_parameters = [
    10,     # Pooled initial drug amount
    2,      # Pooled compartment volume
    -0.35,  # Log mean elimination rate (mu k_e)
    0.3,    # Log std. elimination rate (sigma k_e)
    0.2     # Pooled measurement noise
]

# Sample individuals from population model
n_ids = 1000
individual_parameters = population_model.sample(
    parameters=population_parameters,
    n_samples=n_ids,
    seed=1
)

# Visualise population model (parameter space)
fig = make_subplots(rows=2, cols=2)
fig.add_trace(
    go.Histogram(
        name='Pop. model samples',
        x=individual_parameters[:, 0],
        histnorm='probability',
        showlegend=False,
        xbins_size=0.01
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Histogram(
        name='Pop. model samples',
        x=individual_parameters[:, 1],
        histnorm='probability',
        showlegend=False,
        xbins_size=0.01
    ),
    row=1,
    col=2
)
fig.add_trace(
    go.Histogram(
        name='Pop. model samples',
        x=individual_parameters[:, 2],
        histnorm='probability',
        showlegend=False
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Histogram(
        name='Pop. model samples',
        x=individual_parameters[:, 3],
        histnorm='probability',
        showlegend=False,
        xbins_size=0.01
    ),
    row=2,
    col=2
)
fig.update_layout(
    xaxis_title='Initial drug amount',
    xaxis2_title='Compartment volume',
    xaxis3_title='Elimination rate',
    xaxis4_title='Sigma',
    yaxis_title='Probability',
    yaxis3_title='Probability',
    xaxis_range=[9.8, 10.2],
    xaxis2_range=[1.8, 2.2],
    xaxis4_range=[0, 0.4],
    template='plotly_white'
)
fig.show()
# End 2.
fig.write_html(directory + '/images/1_simulation_7.html')

# Start 3.
# Simulate population distribution of measurements
pop_measurements = np.empty(shape=(n_ids, n_times))
for idd, patient_parameters in enumerate(individual_parameters):
    result = mechanistic_model.simulate(patient_parameters[:-1], times)[0]
    pop_measurements[idd] = error_model.sample(
        patient_parameters[-1:], result)[:, 0]

# Visualise population model (measurement space)
fig = go.Figure()

# Plot 5th to 95th percentile of population distribution
fifth = np.percentile(pop_measurements, q=5, axis=0)
ninety_fifth = np.percentile(pop_measurements, q=95, axis=0)
fig.add_trace(go.Scatter(
    x=np.hstack([times, times[::-1]]),
    y=np.hstack([fifth, ninety_fifth[::-1]]),
    line=dict(width=1, color='grey'),
    fill='toself',
    name='Population model',
    text=r"90% bulk probability",
    hoverinfo='text',
    showlegend=True
))

# Plot patient measurements
fig.add_trace(go.Scatter(
    x=measurement_times,
    y=measurements_1,
    mode='markers',
    marker_color=colors[0],
    name='Meas. patient 1'
))
fig.add_trace(go.Scatter(
    x=measurement_times,
    y=measurements_2,
    mode='markers',
    marker_color=colors[1],
    name='Meas. patient 2'
))
fig.add_trace(go.Scatter(
    x=measurement_times,
    y=measurements_3,
    mode='markers',
    marker_color=colors[2],
    name='Meas. patient 3'
))
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Drug concentration',
    template='plotly_white'
)
fig.show()
# End 3.
fig.write_html(directory + '/images/1_simulation_8.html')