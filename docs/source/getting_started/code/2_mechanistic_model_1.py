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


directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'template.xml')
model = chi.PKPDModel(sbml_file=filename)

print(model._model.name())


# 2
import os

import chi


directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'one_compartment_pk_model.xml')
model = chi.PKPDModel(sbml_file=filename)

print(model._model.code())
print(model.parameters())

# 3
import os

import chi
import numpy as np
import plotly.graph_objects as go


# Define model
directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'one_compartment_pk_model.xml')
model = chi.PKPDModel(sbml_file=filename)
model.set_outputs(['global.drug_concentration'])

# Run simulation
times = np.linspace(start=0, stop=10, num=200)
parameters = [
    10,        # Initial drug amount
    1,         # Elimination rate
    2,         # Volume of distribution
]
simulation = model.simulate(parameters=parameters, times=times)[0]

# Plot drug concentation
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=times,
    y=simulation,
    mode='lines',
))
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Drug concentration',
    template='plotly_white'
)
fig.show()


directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/2_mechanistic_model_1.html')
