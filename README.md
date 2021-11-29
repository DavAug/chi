# Chi 𝞆

[![Unit tests on multiple python versions](https://github.com/DavAug/chi/workflows/Unit%20tests%20(python%20versions)/badge.svg)](https://github.com/DavAug/chi/actions)
[![Unit tests on multiple operating systems](https://github.com/DavAug/chi/workflows/Unit%20tests%20(OS%20versions)/badge.svg)](https://github.com/DavAug/chi/actions)
[![codecov](https://codecov.io/gh/DavAug/chi/branch/main/graph/badge.svg)](https://codecov.io/gh/DavAug/chi)
[![Documentation Status](https://readthedocs.org/projects/chi/badge/?version=latest)](https://chi.readthedocs.io/en/latest/?badge=latest)

## About

**Chi** is an open source Python package hosted on GitHub,
which is designed for dose response modelling.

The main features of chi are

- Simulation of mechanistic dose response models (differential equations) for arbitrary dosing regimens.
- Inference of mechanistic model parameters from data (classical or Bayesian).
- Simulation of the dose response variability in a population
    (hierarchical models/non-linear mixed effects models).
- Inference of population parameters from data (classical or Bayesian).
- Simulation of structured populations, where inter-individual variability can
    be partly explained by covariates.
- Inference of model parameters in a structured population from data
    (classical or Bayesian).
    
Internally, Chi uses [Myokit](https://github.com/MichaelClerx/myokit) as its simulation engine and [Pints](https://github.com/pints-team/pints) as its inference engine.

All features of our software are described in detail in our
[full API documentation](https://chi.readthedocs.io/en/latest/).

## Getting started
### Installation
At the moment chi is not yet distributed with PyPI. One way to use chi
nevertheless is to clone the repository and install it from there, i.e. 
1. Clone the repository with
```bash
git clone https://github.com/DavAug/chi.git
```
2. Install chi by moving into the repository and executing a pip install
```bash
cd chi && pip install .
```

### Modelling and inference with chi
Polished tutorials do currently not exist, but are on the way. In the mean time, please find examples of how chi is 
used to study the tumour growth inhibiting effect of the drug erlotinib in mice [here](https://github.com/DavAug/chi/tree/main/analysis).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
