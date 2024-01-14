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
 - name: Department of Computer Science, University of Oxford, Oxford, UK
   index: 1
date: 05 August 2023
bibliography: paper.bib
---

# Summary

[Chi](https://chi.readthedocs.io) is an open source Python package designed for the modelling of treatment responses with support for implementation, simulation and inference. Supported treatment response models include pharmacokinetic & pharmacodynamic (PKPD) models, physiology-based pharmacokinetic (PBPK) models, quantitative systems pharmacology (QSP) models, and nonlinear mixed effects (NLME) models. The package provides two interfaces to implement single-individual treatment response models: 1. an SBML interface, which implements models based on SBML file specifications [@hucka:2003]; and 2. a general purpose interface that allows users to implement their own, custom models using Python code. Models implemented using SBML files automatically provide routines to administer dosing regimens and to evaluate parameter sensitivities using the simulation engine [Myokit](http://myokit.org/) [@clerx2016myokit]. These single-individual treatment response models can be extended to NLME models, making the simulation of inter-individual variability of treatment responses possible.

In [Chi](https://chi.readthedocs.io), model parameters can be estimated from data using Bayesian inference. We provide a simple interface to estimate posterior distributions of model parameters from single-patient data or from population data. For the extreme case where the available population data only contains a single measurement for each individual, the package also implements a novel inference approach for NLME models, called filter inference, which enables the inference of NLME model parameters from snapshot time series data [@Augustin:2023]. For the purpose of model-informed precision dosing, [Chi](https://chi.readthedocs.io) can be used together with optimisation algorithms to find individual-specific dosing regimens that target desired treatment responses.

For the sampling from posterior distributions, [Chi](https://chi.readthedocs.io) uses Markov chain Monte Carlo (MCMC) algorithms implemented in the Python package [PINTS](https://pints.readthedocs.io/en/stable/) [@Clerx:2019]. For the optimisation of dosing regimens, different optimisers can be used, including optimisers implemented in [SciPy](https://scipy.org/) [@2020SciPy-NMeth] or in [PINTS](https://pints.readthedocs.io/en/stable/) [@Clerx:2019].

Documentation, tutorials and install instructions are available at https://chi.readthedocs.io.

# Statement of need

Treatment response modelling has become an integral part of pharmaceutical research [@SCHUCK:2015; @MORGAN:2018]. In the early phase of drug development, treatment response models help with target and lead identification, and contribute to a (semi-)mechanistic understanding of the relevant pharmacological processes. The modelling results thereafter provide guidance in the transition to the clinical development phase [@LAVE:2016]. During clinical trials, treatment response models help to predict the safety, efficacy and treatment response variability of different dosing strategies. More recently, treatment response models are also being used in the context of model-informed precision dosing, where models help to identify individualised dosing regimens for otherwise difficult-to-administer drugs [@Augustin:20232].

The most widely used software packages and computer programs for treatment response modelling include NONMEM [@keizer2013modeling], [Monolix](https://lixoft.com/products/monolix/), and Matlab Simbiology [@hosseini2018gpkpdsim]. Other software packages include Scipion PKPD [@sorzano2021scipion], [PoPy](https://product.popypkpd.com/), Pumas [@rackauckas2020accelerated], and a number of [R libraries](https://cran.r-project.org/web/views/Pharmacokinetics.html). These packages provide an extensive toolkit for PKPD modelling. However, most of these solutions are difficult to use for research as their source code is not publicly distributed and/or subject to a open-source licenses, concealing the algorithmic details, limiting the scrutiny of the modelling results, and hindering the methdological development. Notable exceptions are Scipion PKPD and the R libraries, which make their source code publicly available on GitHub.

[Chi](https://chi.readthedocs.io/en/latest/index.html) is an easy-to-use, open-source Python package for the modelling of treatment responses. It is targeted at PKPD modellers on all levels of programming expertise. For modellers with an interest in methodological research, [Chi](https://chi.readthedocs.io/en/latest/index.html)'s modular, open source framework enables users to extend or replace individual components of treatment response models and to study their advantages and limitations. We hope that this modularity and transparency of the treatment response modelling framework will encourage the community to contribute to improved methodology and modelling results.

# References