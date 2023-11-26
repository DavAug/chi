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
    import numpy as np
    import pandas as pd
    import pints
    import plotly.graph_objects as go

    # 3
    import pints


    # Define mechanistic model
    directory = os.path.dirname(__file__)
    filename = os.path.join(directory, 'one_compartment_pk_model.xml')
    model = chi.PKPDModel(sbml_file=filename)
    model.set_administration(compartment='global', direct=False)
    model.set_outputs(['global.drug_concentration'])

    # Define error model
    error_model = chi.LogNormalErrorModel()

    # Define data
    directory = os.path.dirname(__file__)
    data = pd.read_csv(directory + '/dataset_1.csv')

    # Define prior distribution
    log_prior = pints.ComposedLogPrior(
        pints.GaussianLogPrior(mean=10, sd=2),           # absorption rate
        pints.GaussianLogPrior(mean=6, sd=2),            # elimination rate
        pints.LogNormalLogPrior(log_mean=0, scale=1),    # volume of distribution
        pints.LogNormalLogPrior(log_mean=-2, scale=0.5)  # scale of meas. distrib.
    )

    # Define log-posterior using the ProblemModellingController
    problem = chi.ProblemModellingController(
        mechanistic_model=model, error_models=[error_model])
    problem.set_data(
        data=data,
        output_observable_dict={'global.drug_concentration': 'Drug concentration'}
    )
    problem.fix_parameters(name_value_dict={
        'dose.drug_amount': 0,
        'global.drug_amount': 0,
    })
    problem.set_log_prior(log_prior=log_prior)
    log_posterior = problem.get_log_posterior()


    # 4
    # Run MCMC algorithm
    n_iterations = 1000
    controller = chi.SamplingController(log_posterior=log_posterior, seed=1)
    controller.set_n_runs(n_runs=3)
    controller.set_parallel_evaluation(False)
    controller.set_sampler(pints.HaarioBardenetACMC)
    samples = controller.run(n_iterations=n_iterations, log_to_screen=False)

    # 5
    from plotly.colors import qualitative
    from plotly.subplots import make_subplots


    # Plot results
    fig = make_subplots(
        rows=4, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1)

    # Plot traces and histogram of parameter
    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            legendgroup='Group 1',
            x=samples['dose.absorption_rate'].values[:, n_iterations//2::(n_iterations//2)//1].flatten(),
            histnorm='probability density',
            showlegend=True,
            xbins=dict(size=0.5),
        marker_color=qualitative.Plotly[4],
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Run 1',
            legendgroup='Group 2',
            x=np.arange(1, n_iterations+1),
            y=samples['dose.absorption_rate'].values[0],
            mode='lines',
            line_color=qualitative.Plotly[2],
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Run 2',
            legendgroup='Group 3',
            x=np.arange(1, n_iterations+1),
            y=samples['dose.absorption_rate'].values[1],
            mode='lines',
            line_color=qualitative.Plotly[1],
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Run 3',
            legendgroup='Group 4',
            x=np.arange(1, n_iterations+1),
            y=samples['dose.absorption_rate'].values[2],
            mode='lines',
            line_color=qualitative.Plotly[0],
        ),
        row=1,
        col=2
    )

    # Plot traces and histogram of parameter
    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            legendgroup='Group 1',
            x=samples['global.elimination_rate'].values[:, n_iterations//2::(n_iterations//2)//1].flatten(),
            histnorm='probability density',
            showlegend=False,
            xbins=dict(size=0.2),
        marker_color=qualitative.Plotly[4],
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Run 1',
            legendgroup='Group 2',
            x=np.arange(1, n_iterations+1),
            y=samples['global.elimination_rate'].values[0],
            mode='lines',
            line_color=qualitative.Plotly[2],
            showlegend=False
        ),
        row=2,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Run 2',
            legendgroup='Group 3',
            x=np.arange(1, n_iterations+1),
            y=samples['global.elimination_rate'].values[1],
            mode='lines',
            line_color=qualitative.Plotly[1],
            showlegend=False
        ),
        row=2,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Run 3',
            legendgroup='Group 4',
            x=np.arange(1, n_iterations+1),
            y=samples['global.elimination_rate'].values[2],
            mode='lines',
            line_color=qualitative.Plotly[0],
            showlegend=False
        ),
        row=2,
        col=2
    )

    # Plot traces and histogram of parameter
    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            legendgroup='Group 1',
            x=samples['global.volume'].values[:, n_iterations//2::(n_iterations//2)//1].flatten(),
            histnorm='probability density',
            showlegend=False,
            xbins=dict(size=0.5),
        marker_color=qualitative.Plotly[4],
        ),
        row=3,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Run 1',
            legendgroup='Group 2',
            x=np.arange(1, n_iterations+1),
            y=samples['global.volume'].values[0],
            mode='lines',
            line_color=qualitative.Plotly[2],
            showlegend=False
        ),
        row=3,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Run 2',
            legendgroup='Group 3',
            x=np.arange(1, n_iterations+1),
            y=samples['global.volume'].values[1],
            mode='lines',
            line_color=qualitative.Plotly[1],
            showlegend=False
        ),
        row=3,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Run 3',
            legendgroup='Group 4',
            x=np.arange(1, n_iterations+1),
            y=samples['global.volume'].values[2],
            mode='lines',
            line_color=qualitative.Plotly[0],
            showlegend=False
        ),
        row=3,
        col=2
    )

    # Plot traces and histogram of parameter
    fig.add_trace(
        go.Histogram(
            name='Posterior samples',
            legendgroup='Group 1',
            x=samples['Sigma log'].values[:, n_iterations//2::(n_iterations//2)//1].flatten(),
            histnorm='probability density',
            showlegend=False,
            xbins=dict(size=0.02),
        marker_color=qualitative.Plotly[4],
        ),
        row=4,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            name='Run 1',
            legendgroup='Group 2',
            x=np.arange(1, n_iterations+1),
            y=samples['Sigma log'].values[0],
            mode='lines',
            line_color=qualitative.Plotly[2],
            showlegend=False
        ),
        row=4,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Run 2',
            legendgroup='Group 3',
            x=np.arange(1, n_iterations+1),
            y=samples['Sigma log'].values[1],
            mode='lines',
            line_color=qualitative.Plotly[1],
            showlegend=False
        ),
        row=4,
        col=2
    )
    fig.add_trace(
        go.Scatter(
            name='Run 3',
            legendgroup='Group 4',
            x=np.arange(1, n_iterations+1),
            y=samples['Sigma log'].values[2],
            mode='lines',
            line_color=qualitative.Plotly[0],
            showlegend=False
        ),
        row=4,
        col=2
    )

    # Visualise prior distribution
    parameter_values = np.linspace(4, 16, num=200)
    pdf_values = np.exp([
        log_prior._priors[0]([parameter_value])
        for parameter_value in parameter_values])
    fig.add_trace(
        go.Scatter(
            name='Prior distribution',
            legendgroup='Group 5',
            x=parameter_values,
            y=pdf_values,
            mode='lines',
            line_color='black',
        ),
        row=1,
        col=1
    )

    parameter_values = np.linspace(0, 12, num=200)
    pdf_values = np.exp([
        log_prior._priors[1]([parameter_value])
        for parameter_value in parameter_values])
    fig.add_trace(
        go.Scatter(
            name='Prior distribution',
            legendgroup='Group 5',
            x=parameter_values,
            y=pdf_values,
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=2,
        col=1
    )

    parameter_values = np.linspace(0, 12, num=200)
    pdf_values = np.exp([
        log_prior._priors[2]([parameter_value])
        for parameter_value in parameter_values])
    fig.add_trace(
        go.Scatter(
            name='Prior distribution',
            legendgroup='Group 5',
            x=parameter_values,
            y=pdf_values,
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=3,
        col=1
    )

    parameter_values = np.linspace(0, 0.6, num=200)
    pdf_values = np.exp([
        log_prior._priors[3]([parameter_value])
        for parameter_value in parameter_values])
    fig.add_trace(
        go.Scatter(
            name='Prior distribution',
            legendgroup='Group 5',
            x=parameter_values,
            y=pdf_values,
            mode='lines',
            line_color='black',
            showlegend=False
        ),
        row=4,
        col=1
    )

    fig.update_layout(
        xaxis_title='k_a',
        yaxis_title='p',
        yaxis2_title='k_a',
        xaxis3_title='k_e',
        yaxis3_title='p',
        yaxis4_title='k_e',
        xaxis5_title='v',
        yaxis5_title='p',
        yaxis6_title='v',
        xaxis7_title='sigma',
        yaxis7_title='p',
        xaxis8_title='Number of iterations',
        yaxis8_title='sigma',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=0, r=0, l=0)
    )
    # fig.show()

    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/3_fitting_models_3.html')


    # 6
    fig = go.Figure()

    # Plot histograms
    fig.add_trace(
        go.Histogram(
            name='Run 1',
            x=samples['dose.absorption_rate'].values[0, ::n_iterations//1],
            histnorm='probability density',
            showlegend=True,
            xbins=dict(size=0.5),
        marker_color=qualitative.Plotly[2],
        )
    )
    fig.add_trace(
        go.Histogram(
            name='Run 2',
            x=samples['dose.absorption_rate'].values[1, ::n_iterations//1],
            histnorm='probability density',
            showlegend=True,
            xbins=dict(size=0.5),
        marker_color=qualitative.Plotly[1],
        )
    )
    fig.add_trace(
        go.Histogram(
            name='Run 3',
            x=samples['dose.absorption_rate'].values[2, ::n_iterations//1],
            histnorm='probability density',
            showlegend=True,
            xbins=dict(size=0.5),
        marker_color=qualitative.Plotly[0],
        )
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Absorption rate in 1/day',
        yaxis_title='Probability density',
        barmode='overlay'
    )
    fig.update_traces(opacity=0.75)
    # fig.show()

    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/3_fitting_models_4.html')


    # 7
    fig = go.Figure()

    # Plot histograms
    fig.add_trace(
        go.Histogram(
            name='Run 1',
            x=samples['dose.absorption_rate'].values[0, n_iterations//2::(n_iterations//2)//1],
            histnorm='probability density',
            showlegend=True,
            xbins=dict(size=0.5),
        marker_color=qualitative.Plotly[2],
        )
    )
    fig.add_trace(
        go.Histogram(
            name='Run 2',
            x=samples['dose.absorption_rate'].values[1, n_iterations//2::n_iterations//1],
            histnorm='probability density',
            showlegend=True,
            xbins=dict(size=0.5),
        marker_color=qualitative.Plotly[1],
        )
    )
    fig.add_trace(
        go.Histogram(
            name='Run 3',
            x=samples['dose.absorption_rate'].values[2, n_iterations//2::n_iterations//1],
            histnorm='probability density',
            showlegend=True,
            xbins=dict(size=0.5),
        marker_color=qualitative.Plotly[0],
        )
    )

    fig.update_layout(
        template='plotly_white',
        xaxis_title='Absorption rate in 1/day',
        yaxis_title='Probability density',
        barmode='overlay'
    )
    fig.update_traces(opacity=0.75)
    # fig.show()

    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/3_fitting_models_5.html')


    # 8
    import arviz as az


    # Summary for all samples
    summary1 = az.summary(samples)
    # print(summary1)

    # Summary for the samples post warmup
    summary2 = az.summary(samples.sel(draw=slice(n_iterations//2, n_iterations)))
    # print(summary2)

    # directory = os.path.dirname(__file__)
    # summary1.to_csv(directory + '/3_fitting_models_summary_1.csv')
    # summary2.to_csv(directory + '/3_fitting_models_summary_2.csv')


    # 9
    # Fix model parameters
    model = chi.ReducedMechanisticModel(mechanistic_model=model)
    model.fix_parameters(name_value_dict={
        'dose.drug_amount': 0,
        'global.drug_amount': 0,
    })

    # Set dosing regimen
    dosing_regimen = problem.get_dosing_regimens()['1']
    model.set_dosing_regimen(dosing_regimen)

    # Extract model parameters from summary
    parameters = [
        summary2['mean'].loc[parameter] for parameter in model.parameters()
    ]

    # Simulate result
    times = np.linspace(0.2, 3, 200)
    simulation = model.simulate(parameters=parameters, times=times)[0]

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=simulation,
        mode='lines',
        name='Fit'
    ))
    fig.add_trace(go.Scatter(
        x=data.Time,
        y=data.Value,
        mode='markers',
        name='Measurements'
    ))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Drug concentration',
        template='plotly_white'
    )
    # fig.show()

    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/3_fitting_models_6.html')


    # 10
    # Define posterior predictive distribution
    predictive_model = problem.get_predictive_model()
    samples = samples.sel(draw=slice(n_iterations//2, n_iterations))
    posterior_model = chi.PosteriorPredictiveModel(
        predictive_model=predictive_model, posterior_samples=samples)

    # Approximate distribution using sampling
    n_samples = 10
    conc_samples = posterior_model.sample(
        times=times, n_samples=n_samples, seed=1)

    # Reshape samples, so we can calculate mean and percentiles at the different
    # time points
    reshaped_samples = np.empty(shape=(n_samples, len(times)))
    for sample_idx, sample_id in enumerate(conc_samples.ID.unique()):
        reshaped_samples[
            sample_idx] = conc_samples[conc_samples.ID == sample_id].Value.values

    # Calculate mean, 5th and 95th percentile of the distribution at each time
    # point
    means = np.mean(reshaped_samples, axis=0)
    lower = np.percentile(reshaped_samples, q=5, axis=0)
    upper = np.percentile(reshaped_samples, q=95, axis=0)

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=simulation,
        mode='lines',
        name='Fit: mean parameters'
    ))
    fig.add_trace(go.Scatter(
        x=data.Time,
        y=data.Value,
        mode='markers',
        name='Measurements'
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=means,
        mode='lines',
        name='Fit: mean posterior pred. distr.',
        line=dict(color='black')
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=lower,
        mode='lines',
        name='Fit: 5th-95th perc. posterior pred. distr.',
        line=dict(color='black', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=upper,
        mode='lines',
        name='Fit: 5th-95th perc. posterior pred. distr.',
        line=dict(color='black', dash='dash'),
        showlegend=False
    ))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Drug concentration',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    # fig.show()

    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/3_fitting_models_7.html')

    # Exit script
    exit()


import os

import chi
import numpy as np
import pandas as pd
import pints
import plotly.graph_objects as go

# 3
import pints


# Define mechanistic model
directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'one_compartment_pk_model.xml')
model = chi.PKPDModel(sbml_file=filename)
model.set_administration(compartment='global', direct=False)
model.set_outputs(['global.drug_concentration'])

# Define error model
error_model = chi.LogNormalErrorModel()

# Define data
directory = os.path.dirname(__file__)
data = pd.read_csv(directory + '/dataset_1.csv')

# Define prior distribution
log_prior = pints.ComposedLogPrior(
    pints.GaussianLogPrior(mean=10, sd=2),           # absorption rate
    pints.GaussianLogPrior(mean=6, sd=2),            # elimination rate
    pints.LogNormalLogPrior(log_mean=0, scale=1),    # volume of distribution
    pints.LogNormalLogPrior(log_mean=-2, scale=0.5)  # scale of meas. distrib.
)

# Define log-posterior using the ProblemModellingController
problem = chi.ProblemModellingController(
    mechanistic_model=model, error_models=[error_model])
problem.set_data(
    data=data,
    output_observable_dict={'global.drug_concentration': 'Drug concentration'}
)
problem.fix_parameters(name_value_dict={
    'dose.drug_amount': 0,
    'global.drug_amount': 0,
})
problem.set_log_prior(log_prior=log_prior)
log_posterior = problem.get_log_posterior()


# 4
# Run MCMC algorithm
n_iterations = 20000
controller = chi.SamplingController(log_posterior=log_posterior, seed=1)
controller.set_n_runs(n_runs=3)
controller.set_parallel_evaluation(False)
controller.set_sampler(pints.HaarioBardenetACMC)
samples = controller.run(n_iterations=n_iterations, log_to_screen=True)

# 5
from plotly.colors import qualitative
from plotly.subplots import make_subplots


# Plot results
fig = make_subplots(
    rows=4, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1)

# Plot traces and histogram of parameter
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        legendgroup='Group 1',
        x=samples['dose.absorption_rate'].values[:, n_iterations//2::(n_iterations//2)//1000].flatten(),
        histnorm='probability density',
        showlegend=True,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[4],
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 1',
        legendgroup='Group 2',
        x=np.arange(1, n_iterations+1),
        y=samples['dose.absorption_rate'].values[0],
        mode='lines',
        line_color=qualitative.Plotly[2],
    ),
    row=1,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Run 2',
        legendgroup='Group 3',
        x=np.arange(1, n_iterations+1),
        y=samples['dose.absorption_rate'].values[1],
        mode='lines',
        line_color=qualitative.Plotly[1],
    ),
    row=1,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Run 3',
        legendgroup='Group 4',
        x=np.arange(1, n_iterations+1),
        y=samples['dose.absorption_rate'].values[2],
        mode='lines',
        line_color=qualitative.Plotly[0],
    ),
    row=1,
    col=2
)

# Plot traces and histogram of parameter
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        legendgroup='Group 1',
        x=samples['global.elimination_rate'].values[:, n_iterations//2::(n_iterations//2)//1000].flatten(),
        histnorm='probability density',
        showlegend=False,
        xbins=dict(size=0.2),
    marker_color=qualitative.Plotly[4],
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 1',
        legendgroup='Group 2',
        x=np.arange(1, n_iterations+1),
        y=samples['global.elimination_rate'].values[0],
        mode='lines',
        line_color=qualitative.Plotly[2],
        showlegend=False
    ),
    row=2,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Run 2',
        legendgroup='Group 3',
        x=np.arange(1, n_iterations+1),
        y=samples['global.elimination_rate'].values[1],
        mode='lines',
        line_color=qualitative.Plotly[1],
        showlegend=False
    ),
    row=2,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Run 3',
        legendgroup='Group 4',
        x=np.arange(1, n_iterations+1),
        y=samples['global.elimination_rate'].values[2],
        mode='lines',
        line_color=qualitative.Plotly[0],
        showlegend=False
    ),
    row=2,
    col=2
)

# Plot traces and histogram of parameter
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        legendgroup='Group 1',
        x=samples['global.volume'].values[:, n_iterations//2::(n_iterations//2)//1000].flatten(),
        histnorm='probability density',
        showlegend=False,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[4],
    ),
    row=3,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 1',
        legendgroup='Group 2',
        x=np.arange(1, n_iterations+1),
        y=samples['global.volume'].values[0],
        mode='lines',
        line_color=qualitative.Plotly[2],
        showlegend=False
    ),
    row=3,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Run 2',
        legendgroup='Group 3',
        x=np.arange(1, n_iterations+1),
        y=samples['global.volume'].values[1],
        mode='lines',
        line_color=qualitative.Plotly[1],
        showlegend=False
    ),
    row=3,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Run 3',
        legendgroup='Group 4',
        x=np.arange(1, n_iterations+1),
        y=samples['global.volume'].values[2],
        mode='lines',
        line_color=qualitative.Plotly[0],
        showlegend=False
    ),
    row=3,
    col=2
)

# Plot traces and histogram of parameter
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        legendgroup='Group 1',
        x=samples['Sigma log'].values[:, n_iterations//2::(n_iterations//2)//1000].flatten(),
        histnorm='probability density',
        showlegend=False,
        xbins=dict(size=0.02),
    marker_color=qualitative.Plotly[4],
    ),
    row=4,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 1',
        legendgroup='Group 2',
        x=np.arange(1, n_iterations+1),
        y=samples['Sigma log'].values[0],
        mode='lines',
        line_color=qualitative.Plotly[2],
        showlegend=False
    ),
    row=4,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Run 2',
        legendgroup='Group 3',
        x=np.arange(1, n_iterations+1),
        y=samples['Sigma log'].values[1],
        mode='lines',
        line_color=qualitative.Plotly[1],
        showlegend=False
    ),
    row=4,
    col=2
)
fig.add_trace(
    go.Scatter(
        name='Run 3',
        legendgroup='Group 4',
        x=np.arange(1, n_iterations+1),
        y=samples['Sigma log'].values[2],
        mode='lines',
        line_color=qualitative.Plotly[0],
        showlegend=False
    ),
    row=4,
    col=2
)

# Visualise prior distribution
parameter_values = np.linspace(4, 16, num=200)
pdf_values = np.exp([
    log_prior._priors[0]([parameter_value])
    for parameter_value in parameter_values])
fig.add_trace(
    go.Scatter(
        name='Prior distribution',
        legendgroup='Group 5',
        x=parameter_values,
        y=pdf_values,
        mode='lines',
        line_color='black',
    ),
    row=1,
    col=1
)

parameter_values = np.linspace(0, 12, num=200)
pdf_values = np.exp([
    log_prior._priors[1]([parameter_value])
    for parameter_value in parameter_values])
fig.add_trace(
    go.Scatter(
        name='Prior distribution',
        legendgroup='Group 5',
        x=parameter_values,
        y=pdf_values,
        mode='lines',
        line_color='black',
        showlegend=False
    ),
    row=2,
    col=1
)

parameter_values = np.linspace(0, 12, num=200)
pdf_values = np.exp([
    log_prior._priors[2]([parameter_value])
    for parameter_value in parameter_values])
fig.add_trace(
    go.Scatter(
        name='Prior distribution',
        legendgroup='Group 5',
        x=parameter_values,
        y=pdf_values,
        mode='lines',
        line_color='black',
        showlegend=False
    ),
    row=3,
    col=1
)

parameter_values = np.linspace(0, 0.6, num=200)
pdf_values = np.exp([
    log_prior._priors[3]([parameter_value])
    for parameter_value in parameter_values])
fig.add_trace(
    go.Scatter(
        name='Prior distribution',
        legendgroup='Group 5',
        x=parameter_values,
        y=pdf_values,
        mode='lines',
        line_color='black',
        showlegend=False
    ),
    row=4,
    col=1
)

fig.update_layout(
    xaxis_title='k_a',
    yaxis_title='p',
    yaxis2_title='k_a',
    xaxis3_title='k_e',
    yaxis3_title='p',
    yaxis4_title='k_e',
    xaxis5_title='v',
    yaxis5_title='p',
    yaxis6_title='v',
    xaxis7_title='sigma',
    yaxis7_title='p',
    xaxis8_title='Number of iterations',
    yaxis8_title='sigma',
    template='plotly_white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(t=0, r=0, l=0)
)
fig.show()

directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/3_fitting_models_3.html')


# 6
fig = go.Figure()

# Plot histograms
fig.add_trace(
    go.Histogram(
        name='Run 1',
        x=samples['dose.absorption_rate'].values[0, ::n_iterations//1000],
        histnorm='probability density',
        showlegend=True,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[2],
    )
)
fig.add_trace(
    go.Histogram(
        name='Run 2',
        x=samples['dose.absorption_rate'].values[1, ::n_iterations//1000],
        histnorm='probability density',
        showlegend=True,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[1],
    )
)
fig.add_trace(
    go.Histogram(
        name='Run 3',
        x=samples['dose.absorption_rate'].values[2, ::n_iterations//1000],
        histnorm='probability density',
        showlegend=True,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[0],
    )
)

fig.update_layout(
    template='plotly_white',
    xaxis_title='Absorption rate in 1/day',
    yaxis_title='Probability density',
    barmode='overlay'
)
fig.update_traces(opacity=0.75)
fig.show()

directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/3_fitting_models_4.html')


# 7
fig = go.Figure()

# Plot histograms
fig.add_trace(
    go.Histogram(
        name='Run 1',
        x=samples['dose.absorption_rate'].values[0, n_iterations//2::(n_iterations//2)//1000],
        histnorm='probability density',
        showlegend=True,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[2],
    )
)
fig.add_trace(
    go.Histogram(
        name='Run 2',
        x=samples['dose.absorption_rate'].values[1, n_iterations//2::n_iterations//1000],
        histnorm='probability density',
        showlegend=True,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[1],
    )
)
fig.add_trace(
    go.Histogram(
        name='Run 3',
        x=samples['dose.absorption_rate'].values[2, n_iterations//2::n_iterations//1000],
        histnorm='probability density',
        showlegend=True,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[0],
    )
)

fig.update_layout(
    template='plotly_white',
    xaxis_title='Absorption rate in 1/day',
    yaxis_title='Probability density',
    barmode='overlay'
)
fig.update_traces(opacity=0.75)
fig.show()

directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/3_fitting_models_5.html')


# 8
import arviz as az


# Summary for all samples
summary1 = az.summary(samples)
print(summary1)

# Summary for the samples post warmup
summary2 = az.summary(samples.sel(draw=slice(n_iterations//2, n_iterations)))
print(summary2)

directory = os.path.dirname(__file__)
summary1.to_csv(directory + '/3_fitting_models_summary_1.csv')
summary2.to_csv(directory + '/3_fitting_models_summary_2.csv')


# 9
# Fix model parameters
model = chi.ReducedMechanisticModel(mechanistic_model=model)
model.fix_parameters(name_value_dict={
    'dose.drug_amount': 0,
    'global.drug_amount': 0,
})

# Set dosing regimen
dosing_regimen = problem.get_dosing_regimens()['1']
model.set_dosing_regimen(dosing_regimen)

# Extract model parameters from summary
parameters = [
    summary2['mean'].loc[parameter] for parameter in model.parameters()
]

# Simulate result
times = np.linspace(0.2, 3, 200)
simulation = model.simulate(parameters=parameters, times=times)[0]

# Plot results
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=times,
    y=simulation,
    mode='lines',
    name='Fit'
))
fig.add_trace(go.Scatter(
    x=data.Time,
    y=data.Value,
    mode='markers',
    name='Measurements'
))
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Drug concentration',
    template='plotly_white'
)
fig.show()

directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/3_fitting_models_6.html')


# 10
# Define posterior predictive distribution
predictive_model = problem.get_predictive_model()
samples = samples.sel(draw=slice(n_iterations//2, n_iterations))
posterior_model = chi.PosteriorPredictiveModel(
    predictive_model=predictive_model, posterior_samples=samples)

# Approximate distribution using sampling
n_samples = 1000
conc_samples = posterior_model.sample(
    times=times, n_samples=n_samples, seed=1)

# Reshape samples, so we can calculate mean and percentiles at the different
# time points
reshaped_samples = np.empty(shape=(n_samples, len(times)))
for sample_idx, sample_id in enumerate(conc_samples.ID.unique()):
    reshaped_samples[
        sample_idx] = conc_samples[conc_samples.ID == sample_id].Value.values

# Calculate mean, 5th and 95th percentile of the distribution at each time
# point
means = np.mean(reshaped_samples, axis=0)
lower = np.percentile(reshaped_samples, q=5, axis=0)
upper = np.percentile(reshaped_samples, q=95, axis=0)

# Plot results
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=times,
    y=simulation,
    mode='lines',
    name='Fit: mean parameters'
))
fig.add_trace(go.Scatter(
    x=data.Time,
    y=data.Value,
    mode='markers',
    name='Measurements'
))
fig.add_trace(go.Scatter(
    x=times,
    y=means,
    mode='lines',
    name='Fit: mean posterior pred. distr.',
    line=dict(color='black')
))
fig.add_trace(go.Scatter(
    x=times,
    y=lower,
    mode='lines',
    name='Fit: 5th-95th perc. posterior pred. distr.',
    line=dict(color='black', dash='dash')
))
fig.add_trace(go.Scatter(
    x=times,
    y=upper,
    mode='lines',
    name='Fit: 5th-95th perc. posterior pred. distr.',
    line=dict(color='black', dash='dash'),
    showlegend=False
))
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Drug concentration',
    template='plotly_white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
)
fig.show()

directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/3_fitting_models_7.html')