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
    #1
    import chi
    import numpy as np


    class OneCompPKModel(chi.MechanisticModel):
        def __init__(self):
            super().__init__()

        def simulate(self, parameters, times):
            a0, ke, v = parameters
            times = np.array(times)

            c = a0 / v * np.exp(-ke * times)

            output = np.empty(shape=(self.n_outputs(), len(times)))
            output[0] = c

            return output

        def has_sensitivities(self):
            # We will not implement sensitivities in this part of the tutorial, so
            # the output of this method is always False
            return False

        def n_outputs(self):
            return 1

        def outputs(self):
            return ['Drug concentration']

        def n_parameters(self):
            return 3

        def parameters(self):
            return ['Initial amount', 'Elimination rate', 'Volume of distribution']


    import os
    # 2
    import plotly.graph_objects as go


    # Define model
    model = OneCompPKModel()

    # Run simulation
    times = np.linspace(start=0, stop=10, num=200)
    parameters = [
        10,        # Initial drug amount
        1,         # Elimination rate
        2,         # Volume of distribution
    ]
    simulation = model.simulate(parameters=parameters, times=times)[0]

    # Plot drug concentration
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
    # fig.show()


    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/2_mechanistic_model_1.html')

    # Exit script
    exit()

#1
import chi
import numpy as np


class OneCompPKModel(chi.MechanisticModel):
    def __init__(self):
        super().__init__()

    def simulate(self, parameters, times):
        a0, ke, v = parameters
        times = np.array(times)

        c = a0 / v * np.exp(-ke * times)

        output = np.empty(shape=(self.n_outputs(), len(times)))
        output[0] = c

        return output

    def has_sensitivities(self):
        # We will not implement sensitivities in this part of the tutorial, so
        # the output of this method is always False
        return False

    def n_outputs(self):
        return 1

    def outputs(self):
        return ['Drug concentration']

    def n_parameters(self):
        return 3

    def parameters(self):
        return ['Initial amount', 'Elimination rate', 'Volume of distribution']


import os
# 2
import plotly.graph_objects as go


# Define model
model = OneCompPKModel()

# Run simulation
times = np.linspace(start=0, stop=10, num=200)
parameters = [
    10,        # Initial drug amount
    1,         # Elimination rate
    2,         # Volume of distribution
]
simulation = model.simulate(parameters=parameters, times=times)[0]

# Plot drug concentration
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