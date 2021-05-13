#
# This script visualises the MCMC traces inferred in
# `hmc_three_hierarchical_parameters.py`
#
import os

import matplotlib.pyplot as plt
import numpy as np
import pints.plot

print('Importing samples...')

# Import parameters from disk
path = os.path.dirname(os.path.abspath(__file__))
filenames = [
    'chain_parameters_0.csv',
    'chain_parameters_1.csv',
    'chain_parameters_2.csv',
    'chain_parameters_3.csv',
    'chain_parameters_4.csv']
temp = []
for filename in filenames:
    temp.append(np.genfromtxt(
        os.path.join(path, filename),
        delimiter=',',
        skip_header=1))

print('Import completed!')

print('Visualising traces...')

# Reshape samples to standard pints format
n_chains = len(temp)
n_iterations, n_parameters = temp[0].shape
samples = np.empty(shape=(n_chains, n_iterations, n_parameters))
for index, s in enumerate(temp):
    samples[index, ...] = s

# Plot traces
pints.plot.trace(samples)
plt.show()

print('Visualisation completed!')
print('Script terminated!')
