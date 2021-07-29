# Erlotinib

[![Unit tests on multiple python versions](https://github.com/DavAug/erlotinib/workflows/Unit%20tests%20(python%20versions)/badge.svg)](https://github.com/DavAug/erlotinib/actions)
[![Unit tests on multiple operating systems](https://github.com/DavAug/erlotinib/workflows/Unit%20tests%20(OS%20versions)/badge.svg)](https://github.com/DavAug/erlotinib/actions)
[![codecov](https://codecov.io/gh/DavAug/erlotinib/branch/main/graph/badge.svg)](https://codecov.io/gh/DavAug/erlotinib)
[![Documentation Status](https://readthedocs.org/projects/erlotinib/badge/?version=latest)](https://erlotinib.readthedocs.io/en/latest/?badge=latest)

## About

**Erlotinib** is an open source Python package hosted on GitHub_,
which is designed for pharmacokinetic and pharmacodynamic (PKPD) modelling.

The main features of erlotinib are

- Simulation of mechanistic dose response models (differential equations)
    for arbitrary dosing regimens.
- Inference of mechanistic model parameters from data (classical or Bayesian).
- Simulation of the dose response variability in a population
    (hierarchical models/non-linear mixed effects models).
- Inference of population parameters from data (classical or Bayesian).
- Simulation of structured populations, where inter-individual variability can
    be partly explained by covariates.
- Inference of model parameters in a strcutured population from data
    (classical or Bayesian).

All features of our software are described in detail in our
[full API documentation](https://erlotinib.readthedocs.io/en/latest/).

## Getting started
### Installation
At the moment erlotinib is not yet distributed with PyPI. One way to use erlotinib
nevertheless is to clone the repository and install it from there, i.e. 
1. Clone the repository with
```bash
git clone https://github.com/DavAug/erlotinib.git
```
2. Install erlotinib by moving into the repository and executing a pip install
```bash
cd erlotinib && pip install .
```

### Modelling and inference with erlotinib
Polished tutorials do currently not exist, but are on the way. In the mean time, please find examples of how erlotinib is 
used to study the tumour growth inhibiting effect of the drug *erlotinib* in mice [here](https://github.com/DavAug/erlotinib/tree/main/analysis).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
