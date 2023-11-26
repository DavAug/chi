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
    # 1
    import os

    import chi


    directory = os.path.dirname(__file__)
    filename = os.path.join(directory, 'template.xml')
    model = chi.PKPDModel(sbml_file=filename)

    # print(model._model.name())


    # 2
    import os

    import chi


    directory = os.path.dirname(__file__)
    filename = os.path.join(directory, 'one_compartment_pk_model.xml')
    model = chi.PKPDModel(sbml_file=filename)

    # print(model._model.code())
    # print(model.parameters())

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
    # fig.show()


    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/2_mechanistic_model_2.html')


    # 4
    model.set_administration(compartment='global')

    # print(model._model.code())

    # 5
    times = np.linspace(start=0, stop=10, num=200)
    parameters = [0, 1, 2]

    # 2 mg every day
    model.set_dosing_regimen(dose=2, period=1)
    sim1 = model.simulate(parameters=parameters, times=times)[0]

    # 4 mg every 2 days
    model.set_dosing_regimen(dose=4, period=2)
    sim2 = model.simulate(parameters=parameters, times=times)[0]

    # 2 mg every day with an infusion of 12 hours
    model.set_dosing_regimen(dose=2, period=1, duration=0.5)
    sim3 = model.simulate(parameters=parameters, times=times)[0]

    # Plot drug concentations
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=sim1,
        mode='lines',
        name='2mg every day'
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=sim2,
        mode='lines',
        name='4mg every 2 days'
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=sim3,
        mode='lines',
        name='2mg every day infused for 12h'
    ))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Drug concentration',
        template='plotly_white'
    )
    # fig.show()


    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/2_mechanistic_model_3.html')

    # 6
    # Define model
    directory = os.path.dirname(__file__)
    filename = os.path.join(directory, 'one_compartment_pk_model.xml')
    model = chi.PKPDModel(sbml_file=filename)
    model.set_outputs(['global.drug_concentration'])

    # Set indirect administration
    model.set_administration(compartment='global', direct=False)

    times = np.linspace(start=0, stop=10, num=200)
    parameters = [0, 0, 10, 1, 2]

    # 2 mg every day
    model.set_dosing_regimen(dose=2, period=1)
    sim1 = model.simulate(parameters=parameters, times=times)[0]

    # 4 mg every 2 days
    model.set_dosing_regimen(dose=4, period=2)
    sim2 = model.simulate(parameters=parameters, times=times)[0]

    # 2 mg every day with an infusion of 12 hours
    model.set_dosing_regimen(dose=2, period=1, duration=0.5)
    sim3 = model.simulate(parameters=parameters, times=times)[0]

    # Plot drug concentations
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=sim1,
        mode='lines',
        name='2mg every day'
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=sim2,
        mode='lines',
        name='4mg every 2 days'
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=sim3,
        mode='lines',
        name='2mg every day infused for 12h'
    ))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Drug concentration',
        template='plotly_white'
    )
    # fig.show()


    # directory = os.path.dirname(os.path.dirname(__file__))
    # fig.write_html(directory + '/images/2_mechanistic_model_4.html')

    # Print model
    model = chi.PKPDModel(sbml_file=filename)
    model.set_outputs(['global.drug_concentration'])

    # Set indirect administration
    model.set_administration(compartment='global', direct=False)

    # print(model._model.code())

    # 7
    from plotly.subplots import make_subplots


    # Simulate concentration and sensitivities
    model.set_dosing_regimen(dose=2, period=1)
    model.enable_sensitivities(True)
    simulation, sensitivities = model.simulate(parameters=parameters, times=times)

    # Plot results
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=simulation[0],
            mode='lines',
            name='Drug concentration'
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=sensitivities[:, 0, 1],
            mode='lines',
            name='Sensitivity w.r.t. initial amount'
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=sensitivities[:, 0, 3],
            mode='lines',
            name='Sensitivity w.r.t. elim. rate'
        ),
        row=3,
        col=1
    )

    fig.update_layout(
        xaxis3_title='Time',
        yaxis_title='c',
        yaxis2_title='dc / da_0',
        yaxis3_title='dc / dk_e',
        template='plotly_white'
    )
    # fig.show()

    # fig.write_html(directory + '/images/2_mechanistic_model_5.html')

    # Exit script
    exit()

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
fig.write_html(directory + '/images/2_mechanistic_model_2.html')


# 4
model.set_administration(compartment='global')

print(model._model.code())

# 5
times = np.linspace(start=0, stop=10, num=200)
parameters = [0, 1, 2]

# 2 mg every day
model.set_dosing_regimen(dose=2, period=1)
sim1 = model.simulate(parameters=parameters, times=times)[0]

# 4 mg every 2 days
model.set_dosing_regimen(dose=4, period=2)
sim2 = model.simulate(parameters=parameters, times=times)[0]

# 2 mg every day with an infusion of 12 hours
model.set_dosing_regimen(dose=2, period=1, duration=0.5)
sim3 = model.simulate(parameters=parameters, times=times)[0]

# Plot drug concentations
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=times,
    y=sim1,
    mode='lines',
    name='2mg every day'
))
fig.add_trace(go.Scatter(
    x=times,
    y=sim2,
    mode='lines',
    name='4mg every 2 days'
))
fig.add_trace(go.Scatter(
    x=times,
    y=sim3,
    mode='lines',
    name='2mg every day infused for 12h'
))
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Drug concentration',
    template='plotly_white'
)
fig.show()


directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/2_mechanistic_model_3.html')

# 6
# Define model
directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'one_compartment_pk_model.xml')
model = chi.PKPDModel(sbml_file=filename)
model.set_outputs(['global.drug_concentration'])

# Set indirect administration
model.set_administration(compartment='global', direct=False)

times = np.linspace(start=0, stop=10, num=200)
parameters = [0, 0, 10, 1, 2]

# 2 mg every day
model.set_dosing_regimen(dose=2, period=1)
sim1 = model.simulate(parameters=parameters, times=times)[0]

# 4 mg every 2 days
model.set_dosing_regimen(dose=4, period=2)
sim2 = model.simulate(parameters=parameters, times=times)[0]

# 2 mg every day with an infusion of 12 hours
model.set_dosing_regimen(dose=2, period=1, duration=0.5)
sim3 = model.simulate(parameters=parameters, times=times)[0]

# Plot drug concentations
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=times,
    y=sim1,
    mode='lines',
    name='2mg every day'
))
fig.add_trace(go.Scatter(
    x=times,
    y=sim2,
    mode='lines',
    name='4mg every 2 days'
))
fig.add_trace(go.Scatter(
    x=times,
    y=sim3,
    mode='lines',
    name='2mg every day infused for 12h'
))
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Drug concentration',
    template='plotly_white'
)
fig.show()


directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/2_mechanistic_model_4.html')

# Print model
model = chi.PKPDModel(sbml_file=filename)
model.set_outputs(['global.drug_concentration'])

# Set indirect administration
model.set_administration(compartment='global', direct=False)

print(model._model.code())

# 7
from plotly.subplots import make_subplots


# Simulate concentration and sensitivities
model.set_dosing_regimen(dose=2, period=1)
model.enable_sensitivities(True)
simulation, sensitivities = model.simulate(parameters=parameters, times=times)

# Plot results
fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
fig.add_trace(
    go.Scatter(
        x=times,
        y=simulation[0],
        mode='lines',
        name='Drug concentration'
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=times,
        y=sensitivities[:, 0, 1],
        mode='lines',
        name='Sensitivity w.r.t. initial amount'
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=times,
        y=sensitivities[:, 0, 3],
        mode='lines',
        name='Sensitivity w.r.t. elim. rate'
    ),
    row=3,
    col=1
)

fig.update_layout(
    xaxis3_title='Time',
    yaxis_title='c',
    yaxis2_title='dc / da_0',
    yaxis3_title='dc / dk_e',
    template='plotly_white'
)
fig.show()

fig.write_html(directory + '/images/2_mechanistic_model_5.html')