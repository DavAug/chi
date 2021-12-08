# Chi

[![Unit tests on multiple python versions](https://github.com/DavAug/chi/workflows/Unit%20tests%20(python%20versions)/badge.svg)](https://github.com/DavAug/chi/actions)
[![Unit tests on multiple operating systems](https://github.com/DavAug/chi/workflows/Unit%20tests%20(OS%20versions)/badge.svg)](https://github.com/DavAug/chi/actions)
[![codecov](https://codecov.io/gh/DavAug/chi/branch/main/graph/badge.svg)](https://codecov.io/gh/DavAug/chi)
[![Documentation Status](https://readthedocs.org/projects/chi/badge/?version=latest)](https://chi.readthedocs.io/en/latest/?badge=latest)

## About

**Chi** is an open source Python package hosted on GitHub,
which can be used to model dose response dynamics.

All features of our software are described in detail in our
[full API documentation](https://chi.readthedocs.io/en/latest/).

## Getting started
### Installation

1. Install CVODES

Chi uses the open-source package Myokit to solve mechanistic models
and compute their sensitivities efficiently. Myokit does this using CVODESS,
which need to be installed with:

- On Ubuntu:
```bash
apt-get install libsundials-dev
```

- On MacOS:
```bash
brew install sundials
```

- On Windows: No action required. Myokit will install CVODESS automatically.

2. Install chi
```bash
pip install chi-drm
```

### Usage
 You can now use chi's functionality by importing it
 ```python
import chi
 ```

 Tutorials and more detailed explanations on how to use chi are currently
 under development.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)
