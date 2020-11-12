from setuptools import setup, find_packages

# Go!
setup(
    # Module name
    name='erlotinib',
    version='0.0.1dev0',

    # License name
    license='BSD 3-clause license',

    # Maintainer information
    maintainer='David Augustin',
    maintainer_email='david.augustin@cs.ox.ac.uk',

    # Packages and data to include
    packages=find_packages(include=('erlotinib', 'erlotinib.*')),
    include_package_data=True,

    # List of dependencies
    install_requires=[
        'dash>=1.17.0',
        'jupyter==1.0.0',
        'myokit>=1.31',
        'numpy>=1.8',
        'pandas>=0.24',
        'pints @ git+git://github.com/pints-team/pints.git#egg=pints',
        'plotly==4.8.1',
        'tqdm==4.46.1'
    ],
    dependency_links=[
     "git+git://github.com/pints-team/pints.git#egg=pints-latest",
    ],
    extras_require={
        'docs': [
            'sphinx>=1.5, !=1.7.3',     # For doc generation
        ],
    },
)
