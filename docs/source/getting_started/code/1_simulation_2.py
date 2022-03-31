import argparse


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
    # End 1.

    from plotly.subplots import make_subplots
    # Start 2.
    # Define population model
    population_model = chi.ComposedPopulationModel([
        chi.PooledModel(dim_names=['Initial drug amount']),
        chi.PooledModel(dim_names=['Compartment volume']),
        chi.LogNormalModel(dim_names=['Elimination rate']),
        chi.PooledModel(dim_names=['Sigma'])
    ])

    # Define population parameters
    population_parameters = [
        10,     # Pooled initial drug amount
        2,      # Pooled compartment volume
        -0.55,  # Log mean elimination rate (mu k_e)
        0.3,    # Log std. elimination rate (sigma k_e)
        0.4     # Pooled measurement noise
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
            xbins_size=0.01,
            marker_color='lightgrey'
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
            xbins_size=0.01,
            marker_color='lightgrey'
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Histogram(
            name='Pop. model samples',
            x=individual_parameters[:, 2],
            histnorm='probability',
            showlegend=False,
            marker_color='lightgrey'
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
            xbins_size=0.01,
            marker_color='lightgrey'
        ),
        row=2,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Patient 1',
            x=[parameters_patient_1[2], parameters_patient_1[2]],
            y=[0, 0.12],
            mode='lines',
            line=dict(color=colors[0], dash='dash'),
            showlegend=False
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Patient 2',
            x=[parameters_patient_2[2], parameters_patient_2[2]],
            y=[0, 0.12],
            mode='lines',
            line=dict(color=colors[1], dash='dash'),
            showlegend=False
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Patient 3',
            x=[parameters_patient_3[2], parameters_patient_3[2]],
            y=[0, 0.12],
            mode='lines',
            line=dict(color=colors[2], dash='dash'),
            showlegend=False
        ),
        row=2,
        col=1
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
        xaxis4_range=[0.2, 0.6],
        template='plotly_white'
    )
    # End 2.

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
        line=dict(width=1, color='lightgrey'),
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
    # End 3.

    import pints
    from plotly.subplots import make_subplots
    # Start 4.
    # Define individual log-likelihoods for patients
    log_likelihood_1 = chi.LogLikelihood(
        mechanistic_model, error_model, measurements_1, measurement_times)
    log_likelihood_2 = chi.LogLikelihood(
        mechanistic_model, error_model, measurements_2, measurement_times)
    log_likelihood_3 = chi.LogLikelihood(
        mechanistic_model, error_model, measurements_3, measurement_times)

    # Define hierarchical log-likelihood
    log_likelihood = chi.HierarchicalLogLikelihood(
        log_likelihoods=[log_likelihood_1, log_likelihood_2, log_likelihood_3],
        population_model=population_model)
    # End 4.

    # Start 5.
    # Define hierarchical log-posterior
    log_prior = pints.ComposedLogPrior(
        pints.LogNormalLogPrior(1, 1),      # Initial drug amount
        pints.LogNormalLogPrior(1, 1),      # Compartment volume
        pints.GaussianLogPrior(-1, 1),      # Log mean elimination rate
        pints.LogNormalLogPrior(-1, 0.4),   # Log std. elimination rate
        pints.LogNormalLogPrior(1, 1)       # Sigma
    )
    log_posterior = chi.HierarchicalLogPosterior(
        log_likelihood=log_likelihood, log_prior=log_prior)

    # Infer posterior
    controller = chi.SamplingController(log_posterior, seed=42)
    controller.set_n_runs(1)
    controller.set_parallel_evaluation(False)
    controller.set_sampler(pints.NoUTurnMCMC)
    n_iterations = 600
    posterior_samples = controller.run(n_iterations, log_to_screen=True)

    # Discard warmup iterations
    warmup = 500  # This number is not arbitrary but was carefully chosen
    posterior_samples = posterior_samples.sel(draw=slice(warmup, n_iterations))

    # Visualise posteriors
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True)
    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            x=posterior_samples['Pooled Initial drug amount'].values.flatten(),
            histnorm='probability',
            marker_color='lightgrey',
            showlegend=False
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[population_parameters[0]]*2,
            y=[0, 0.08],
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
            x=posterior_samples['Pooled Compartment volume'].values.flatten(),
            histnorm='probability',
            marker_color='lightgrey',
            showlegend=False
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[population_parameters[1]]*2,
            y=[0, 0.08],
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=1,
        col=2
    )

    fig.add_trace(
        go.Histogram(
            name='Patient 1 Post. samples',
            x=posterior_samples['myokit.elimination_rate'].values[
                0, :, 0].flatten(),
            histnorm='probability',
            showlegend=False,
            marker=dict(color=colors[0], opacity=0.7)
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Histogram(
            name='Patient 1 Post. samples',
            x=posterior_samples['myokit.elimination_rate'].values[
                0, :, 1].flatten(),
            histnorm='probability',
            showlegend=False,
            marker=dict(color=colors[1], opacity=0.7)
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Histogram(
            name='Patient 1 post. samples',
            x=posterior_samples['myokit.elimination_rate'].values[
                0, :, 2].flatten(),
            histnorm='probability',
            showlegend=False,
            marker=dict(color=colors[2], opacity=0.7)
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Patent 1 data-gen. parameters',
            x=[parameters_patient_1[2]]*2,
            y=[0, 0.06],
            mode='lines',
            line_color='darkblue',
            showlegend=False
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Patent 2 data-gen. parameters',
            x=[parameters_patient_2[2]]*2,
            y=[0, 0.06],
            mode='lines',
            line_color='darkred',
            showlegend=False
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Patent 3 data-gen. parameters',
            x=[parameters_patient_3[2]]*2,
            y=[0, 0.06],
            mode='lines',
            line_color='darkgreen',
            showlegend=False
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            x=posterior_samples['Pooled Sigma'].values.flatten(),
            histnorm='probability',
            marker_color='lightgrey',
            showlegend=False
        ),
        row=2,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[population_parameters[4]]*2,
            y=[0, 0.06],
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
        template='plotly_white',
        bargap=0,
        bargroupgap=0,
        barmode='overlay'
    )
    # End 5.

    # Start 6.
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            x=posterior_samples['Log mean Elimination rate'].values.flatten(),
            histnorm='probability',
            marker_color='lightgrey',
            showlegend=False
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[population_parameters[2]]*2,
            y=[0, 0.08],
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
            x=posterior_samples['Log std. Elimination rate'].values.flatten(),
            histnorm='probability',
            marker_color='lightgrey',
            showlegend=False
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Data-generating parameters',
            x=[population_parameters[3]]*2,
            y=[0, 0.08],
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=1,
        col=2
    )

    fig.update_layout(
        xaxis_title='Log mean elimination rate',
        xaxis2_title='Log std. elimination rate',
        yaxis_title='Probability',
        template='plotly_white'
    )
    # End 6.

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
    chi.PooledModel(dim_names=['Sigma'])
])

# Define population parameters
population_parameters = [
    10,     # Pooled initial drug amount
    2,      # Pooled compartment volume
    -0.55,  # Log mean elimination rate (mu k_e)
    0.3,    # Log std. elimination rate (sigma k_e)
    0.4     # Pooled measurement noise
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
        xbins_size=0.01,
        marker_color='lightgrey'
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
        xbins_size=0.01,
        marker_color='lightgrey'
    ),
    row=1,
    col=2
)
fig.add_trace(
    go.Histogram(
        name='Pop. model samples',
        x=individual_parameters[:, 2],
        histnorm='probability',
        showlegend=False,
        marker_color='lightgrey'
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
        xbins_size=0.01,
        marker_color='lightgrey'
    ),
    row=2,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Patient 1',
        x=[parameters_patient_1[2], parameters_patient_1[2]],
        y=[0, 0.12],
        mode='lines',
        line=dict(color=colors[0], dash='dash'),
        showlegend=False
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Patient 2',
        x=[parameters_patient_2[2], parameters_patient_2[2]],
        y=[0, 0.12],
        mode='lines',
        line=dict(color=colors[1], dash='dash'),
        showlegend=False
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Patient 3',
        x=[parameters_patient_3[2], parameters_patient_3[2]],
        y=[0, 0.12],
        mode='lines',
        line=dict(color=colors[2], dash='dash'),
        showlegend=False
    ),
    row=2,
    col=1
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
    xaxis4_range=[0.2, 0.6],
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
    line=dict(width=1, color='lightgrey'),
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

import pints
from plotly.subplots import make_subplots
# Start 4.
# Define individual log-likelihoods for patients
log_likelihood_1 = chi.LogLikelihood(
    mechanistic_model, error_model, measurements_1, measurement_times)
log_likelihood_2 = chi.LogLikelihood(
    mechanistic_model, error_model, measurements_2, measurement_times)
log_likelihood_3 = chi.LogLikelihood(
    mechanistic_model, error_model, measurements_3, measurement_times)

# Define hierarchical log-likelihood
log_likelihood = chi.HierarchicalLogLikelihood(
    log_likelihoods=[log_likelihood_1, log_likelihood_2, log_likelihood_3],
    population_model=population_model)
# End 4.

# Start 5.
# Define hierarchical log-posterior
log_prior = pints.ComposedLogPrior(
    pints.LogNormalLogPrior(1, 1),      # Initial drug amount
    pints.LogNormalLogPrior(1, 1),      # Compartment volume
    pints.GaussianLogPrior(-1, 1),      # Log mean elimination rate
    pints.LogNormalLogPrior(-1, 0.4),   # Log std. elimination rate
    pints.LogNormalLogPrior(1, 1)       # Sigma
)
log_posterior = chi.HierarchicalLogPosterior(
    log_likelihood=log_likelihood, log_prior=log_prior)

# Infer posterior
controller = chi.SamplingController(log_posterior, seed=42)
controller.set_n_runs(1)
controller.set_parallel_evaluation(False)
controller.set_sampler(pints.NoUTurnMCMC)
n_iterations = 2000
posterior_samples = controller.run(n_iterations, log_to_screen=True)

# Discard warmup iterations
warmup = 500  # This number is not arbitrary but was carefully chosen
posterior_samples = posterior_samples.sel(draw=slice(warmup, n_iterations))

# Visualise posteriors
fig = make_subplots(rows=2, cols=2, shared_yaxes=True)
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        x=posterior_samples['Pooled Initial drug amount'].values.flatten(),
        histnorm='probability',
        marker_color='lightgrey',
        showlegend=False
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Data-generating parameters',
        x=[population_parameters[0]]*2,
        y=[0, 0.08],
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
        x=posterior_samples['Pooled Compartment volume'].values.flatten(),
        histnorm='probability',
        marker_color='lightgrey',
        showlegend=False
    ),
    row=1,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Data-generating parameters',
        x=[population_parameters[1]]*2,
        y=[0, 0.08],
        mode='lines',
        line_color='black',
        showlegend=False
    ),
    row=1,
    col=2
)

fig.add_trace(
    go.Histogram(
        name='Patient 1 Post. samples',
        x=posterior_samples['myokit.elimination_rate'].values[
            0, :, 0].flatten(),
        histnorm='probability',
        showlegend=False,
        marker=dict(color=colors[0], opacity=0.7)
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Histogram(
        name='Patient 1 Post. samples',
        x=posterior_samples['myokit.elimination_rate'].values[
            0, :, 1].flatten(),
        histnorm='probability',
        showlegend=False,
        marker=dict(color=colors[1], opacity=0.7)
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Histogram(
        name='Patient 1 post. samples',
        x=posterior_samples['myokit.elimination_rate'].values[
            0, :, 2].flatten(),
        histnorm='probability',
        showlegend=False,
        marker=dict(color=colors[2], opacity=0.7)
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Patent 1 data-gen. parameters',
        x=[parameters_patient_1[2]]*2,
        y=[0, 0.06],
        mode='lines',
        line_color='darkblue',
        showlegend=False
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Patent 2 data-gen. parameters',
        x=[parameters_patient_2[2]]*2,
        y=[0, 0.06],
        mode='lines',
        line_color='darkred',
        showlegend=False
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Patent 3 data-gen. parameters',
        x=[parameters_patient_3[2]]*2,
        y=[0, 0.06],
        mode='lines',
        line_color='darkgreen',
        showlegend=False
    ),
    row=2,
    col=1
)

fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        x=posterior_samples['Pooled Sigma'].values.flatten(),
        histnorm='probability',
        marker_color='lightgrey',
        showlegend=False
    ),
    row=2,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Data-generating parameters',
        x=[population_parameters[4]]*2,
        y=[0, 0.06],
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
    template='plotly_white',
    bargap=0,
    bargroupgap=0,
    barmode='overlay'
)
fig.show()
# End 5.
fig.write_html(directory + '/images/1_simulation_9.html')

# Start 6.
fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        x=posterior_samples['Log mean Elimination rate'].values.flatten(),
        histnorm='probability',
        marker_color='lightgrey',
        showlegend=False
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Data-generating parameters',
        x=[population_parameters[2]]*2,
        y=[0, 0.08],
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
        x=posterior_samples['Log std. Elimination rate'].values.flatten(),
        histnorm='probability',
        marker_color='lightgrey',
        showlegend=False
    ),
    row=1,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Data-generating parameters',
        x=[population_parameters[3]]*2,
        y=[0, 0.08],
        mode='lines',
        line_color='black',
        showlegend=False
    ),
    row=1,
    col=2
)

fig.update_layout(
    xaxis_title='Log mean elimination rate',
    xaxis2_title='Log std. elimination rate',
    yaxis_title='Probability',
    template='plotly_white'
)
fig.show()
# End 6.
fig.write_html(directory + '/images/1_simulation_10.html')
