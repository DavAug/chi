from setuptools import setup, find_packages

# Load text for description
with open('README.md') as f:
    readme = f.read()


# Go!
setup(
    # Module name
    name='chi-drm',
    version='0.1.0',
    description='Package to model dose response dynamics',
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://chi.readthedocs.io",

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    maintainer='David Augustin',
    maintainer_email='david.augustin@cs.ox.ac.uk',

    # Packages and data to include
    packages=find_packages(
        include=('chi', 'chi.*'),
        exclude=('chi/tests',)),
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
