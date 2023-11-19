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

# 1
import os

import chi
import numpy as np
import plotly.graph_objects as go

# Implement the model
directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'one_compartment_pk_model.xml')
model = chi.PKPDModel(sbml_file=filename)
model.set_outputs(['global.drug_concentration'])

# Set administration and dosing regimen
model.set_administration(compartment='global', direct=False)
model.set_dosing_regimen(dose=2, period=1)

# Simulate treatment response
parameters = [0, 0, 10, 1, 2]
times = np.linspace(start=0, stop=3, num=200)
conc = model.simulate(parameters=parameters, times=times)[0]

# Plot results
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=times,
    y=conc,
    mode='lines',
))
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Drug concentration',
    template='plotly_white'
)
fig.show()

directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/3_fitting_models_1.html')


### Generate synthetic data
directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'two_compartment_pk_model.xml')
model = chi.PKPDModel(sbml_file=filename)
model.set_administration(compartment='central', direct=False)
model.set_outputs(['central.drug_concentration'])

# Synthethise data
error_model = chi.LogNormalErrorModel()
model = chi.PredictiveModel(
    mechanistic_model=model, error_models=[error_model])
model.set_dosing_regimen(dose=2, period=1)

parameters = [0, 0, 0, 2, 10, 1, 5, 1, 10, 0.1]
times = [0.5, 1, 1.5, 2, 2.5, 3]
df = model.sample(
    parameters=parameters, times=times, seed=1, include_regimen=True)

# Save data
directory = os.path.dirname(os.path.dirname(__file__))
df.to_csv(directory + '/data/3_fitting_models_1.csv', index=False)
