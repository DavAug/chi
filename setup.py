from setuptools import setup, find_packages

# Go!
setup(
    # Module name
    name='chi',
    version='0.1.0',

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    maintainer='David Augustin',
    maintainer_email='david.augustin@cs.ox.ac.uk',

    # Packages and data to include
    packages=find_packages(include=('chi', 'chi.*')),
    include_package_data=True,

    # List of dependencies
    install_requires=[
        'arviz>=0.11',
        'myokit>=1.33',
        'numpy>=1.17',
        'pandas>=0.24',
        'pints>=0.4',
        'plotly>=4.8.1',
        'tqdm>=4.46.1',
        'xarray>=0.19'
    ],
    extras_require={
        'docs': [
            'furo',
            'sphinx>=1.5, !=1.7.3',     # For doc generation
        ],
        'notebooks': [
            'jupyter==1.0.0',
        ]
    },
)
