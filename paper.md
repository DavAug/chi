---
title: 'Chi: A Python package for treatment response modelling'
tags:
  - Python
  - pkpd
  - treatment planning
  - inference
  - Bayesian inference
authors:
  - name: David Augustin
    orcid: 0000-0002-4885-1088
    corresponding: true
    affiliation: 1
affiliations:
 - name: Department of Computer Science, University of Oxford, Oxford, United Kingdom
   index: 1
date: 05 August 2023
bibliography: paper.bib
---

# Summary

[Chi](https://chi.readthedocs.io) is an open source Python package for treatment response modelling with support for implementation, simulation and parameter estimation. Supported treatment response models include pharmacokinetic & pharmacodynamic (PKPD) models, physiology-based pharmacokinetic (PBPK) models, quantitative systems pharmacology (QSP) models, and nonlinear mixed effects (NLME) models. The package provides two interfaces to implement single-individual treatment response models: 1. an SBML interface, which implements models based on SBML file specifications [@hucka:2003]; and 2. a general purpose interface that allows users to implement their own, custom models using Python code. Models implemented using SBML files automatically provide routines to administer dosing regimens and to evaluate parameter sensitivities using the simulation engine [Myokit](http://myokit.org/) [@clerx2016myokit]. These single-individual treatment response models can be extended to NLME models, making the simulation of inter-individual variability of treatment responses possible.

In [Chi](https://chi.readthedocs.io), model parameters can be estimated from data using Bayesian inference. We provide a simple interface to infer posterior distributions of model parameters from single-patient data or from population data. For the extreme case where the available population data only contains a single measurement for each individual, the package also implements filter inference, enabling the inference of NLME model parameters from such snapshot time series data [@Augustin:2023]. For the purpose of model-informed precision dosing (MIPD), [Chi](https://chi.readthedocs.io) can be used to find individual-specific dosing regimens that optimise treatment responses with respect to a target treatment outcome.

To sample from posterior distributions, [Chi](https://chi.readthedocs.io) uses Markov chain Monte Carlo (MCMC) algorithms implemented in the Python package [PINTS](https://pints.readthedocs.io/en/stable/) [@Clerx:2019]. To optimise dosing regimens, different optimisation algorithms can be used, including optimisers implemented in [SciPy](https://scipy.org/) [@2020SciPy-NMeth] or in [PINTS](https://pints.readthedocs.io/en/stable/) [@Clerx:2019].

Documentation, tutorials and install instructions are available at https://chi.readthedocs.io.

# Statement of need

Treatment response modelling has become an integral part of pharmaceutical research [@SCHUCK:2015; @MORGAN:2018]. In the early phase of drug development, treatment response models help with target and lead identification, and contribute to a mechanistic understanding of the relevant pharmacological processes. In the transition to the clinical development phase, these models provide guidance and help to identify safe and efficacious dosing regimens [@LAVE:2016]. During clinical trials, treatment response models further facilitate the assessment of safety, efficacy and treatment response variability. More recently, treatment response models are also being used in the context of MIPD, where models help to predict individualised dosing regimens for otherwise difficult-to-administer drugs [@Augustin:20232].

The most widely used software packages and computer programs for treatment response modelling include NONMEM [@keizer2013modeling], [Monolix](https://lixoft.com/products/monolix/), and Matlab Simbiology [@hosseini2018gpkpdsim]. Other software packages include Scipion PKPD [@sorzano2021scipion], [PoPy](https://product.popypkpd.com/), Pumas [@rackauckas2020accelerated], and a number of [R libraries](https://cran.r-project.org/web/views/Pharmacokinetics.html). These packages provide an extensive toolkit for PKPD modelling. However, most of these solutions are difficult to use for research as their source code is neither publicly distributed nor subject to open-source licenses, which conceals the algorithmic details, limits the transparency of the modelling results, and hinders the methological development. Notable exceptions are Scipion PKPD and the R libraries, which make their source code publicly available on GitHub.

[Chi](https://chi.readthedocs.io/en/latest/index.html) is an easy-to-use, open-source Python package for treatment response modelling. It is targeted at modellers on all levels of programming expertise. Modellers with a primary focus on the pharmacology can use [Chi](https://chi.readthedocs.io/en/latest/index.html) to quickly implement models and estimate their model parameters from data. Modellers with an interest in methodological research can use [Chi](https://chi.readthedocs.io/en/latest/index.html)'s modular, open source framework to study the advantages and limitations of different modelling choices, as well as research new approaches for treatment response modelling. We hope that the open-source nature of this package will increase the transparency of treatment response models and facilitate a community effort to further develop their methodology.

# References