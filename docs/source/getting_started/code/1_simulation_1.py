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
fig.show()
## End 2.

# Export for docs
import os

directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/1_simulation_1.html')

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
fig.show()
# End 3.

fig.write_html(directory + '/images/1_simulation_2.html')

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
fig.show()
# End 6.
fig.write_html(directory + '/images/1_simulation_3.html')

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
fig.show()
# End 9.
fig.write_html(directory + '/images/1_simulation_5.html')
