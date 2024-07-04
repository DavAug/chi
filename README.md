# Chi

[![Unit tests on multiple python versions](https://github.com/DavAug/chi/workflows/Unit%20tests%20(python%20versions)/badge.svg)](https://github.com/DavAug/chi/actions)
[![Unit tests on multiple operating systems](https://github.com/DavAug/chi/workflows/Unit%20tests%20(OS%20versions)/badge.svg)](https://github.com/DavAug/chi/actions)
[![codecov](https://codecov.io/gh/DavAug/chi/branch/main/graph/badge.svg)](https://codecov.io/gh/DavAug/chi)
[![Documentation Status](https://readthedocs.org/projects/chi/badge/?version=latest)](https://chi.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05925/status.svg)](https://doi.org/10.21105/joss.05925)

## About

**Chi** is an open source Python package for pharmacokinetic and pharmacodynamic (PKPD) modelling.

All features of the software are described in detail in the
[full API documentation](https://chi.readthedocs.io/en/latest/).

## Getting started
### Installation

1. Install sundials

Chi uses the open source package Myokit to solve ordinary differential equations
and compute their sensitivities efficiently. Myokit does this using sundials' CVODESS,
which needs to be installed with:

- On Ubuntu:
```bash
apt-get install libsundials-dev
```

- On MacOS:
```bash
brew install sundials
```

- On Windows: No action required. Myokit will install sundial automatically.

2. Install chi
```bash
pip install chi-drm
```

 You can now use chi's functionality by importing it
 ```python
import chi
 ```

 ### Tutorials

 Tutorials and more detailed explanations on how to use chi can be found in the [documentation's getting started](https://chi.readthedocs.io/en/latest/getting_started/index.html) section.

## Citation

If you use this software in your work, please cite it using the following metadata:

#### Citation string
```
Augustin, D., (2024). Chi: A Python package for treatment response modelling. Journal of Open Source Software, 9(94), 5925, https://doi.org/10.21105/joss.05925
```

#### BibTeX
```
@article{
    Augustin2024, 
    doi = {10.21105/joss.05925}, 
    url = {https://doi.org/10.21105/joss.05925}, 
    year = {2024}, 
    publisher = {The Open Journal}, 
    volume = {9}, 
    number = {94}, 
    pages = {5925}, 
    author = {David Augustin}, 
    title = {Chi: A Python package for treatment response modelling}, 
    journal = {Journal of Open Source Software} 
}
```

## Contributing
There are lots of ways how you can contribute to Chi's development, and you are welcome to join in!
For example, you can report problems or make feature requests on the [issues](https://github.com/DavAug/chi/issues) pages.

Similarly, if you would like to contribute documentation or code you can create and issue, and then provide a pull request for review.
To facilitate contributions, we have written extensive [contribution guidelines](https://github.com/DavAug/chi/blob/main/CONTRIBUTING.md) to help you navigate the code.

## License
[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
