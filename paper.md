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

[Chi](https://chi.readthedocs.io/en/latest/index.html) is an easy-to-use, open source Python package for the modelling of pharmacokinetics & pharmacodynamics (PKPD). We provide two flexible interfaces to implement PKPD models: 1. an SBML interface, which implements PKPD models based on SBML file specifications [@hucka:2003]; and 2. a general purpose interface that allows users to implement their own, custom PKPD models using Python code. PKPD models instantiated from SBML files automatically implement the administration of custom dosing regimens and the evaluation of parameter sensitivities [@clerx2016myokit]. We also provide a simple framework to extend PKPD models to nonlinear mixed effects (NLME) models, making the simulation of inter-individual variability of treatment responses possible.

In [Chi](https://chi.readthedocs.io/en/latest/index.html), model parameters can be estimated from data using Bayesian inference. We provide a simple interface to estimate posterior distributions of PKPD model parameters and NLME model parameters. [Chi](https://chi.readthedocs.io/en/latest/index.html) also implements filter inference, a novel inference approach which makes the estimation of NLME model parameters from snapshot time series data possible [@Augustin:2023].

For the sampling from posterior distributions, [Chi](https://chi.readthedocs.io/en/latest/index.html) uses Markov chain Monte Carlo (MCMC) algorithms implemented in the Python package [PINTS](https://pints.readthedocs.io/en/stable/) [@Clerx:2019].

# Statement of need

PKPD modelling has become an integral part of pharmaceutical research [@SCHUCK:2015; @MORGAN:2018]. In the early phase of drug development, PKPD models help with target and lead identification, and contribute to a (semi-)mechanistic understanding of the relevant pharmacological processes. The modelling results thereafter provide guidance in the transition to the clinical development phase [@LAVE:2016]. During clinical trials, PKPD models help predict the safety, efficacy and treatment response variability of different dosing strategies. More recently, PKPD models are also considered to help with the individualisation of dosing regimens of otherwise difficult-to-administer drugs [@Augustin:20232].

The most widely used programmes for PKPD modelling include NONMEM [@keizer2013modeling], [Monolix](https://lixoft.com/products/monolix/), and Matlab Simbiology [@hosseini2018gpkpdsim]. Other software packages include Scipion PKPD [@sorzano2021scipion], [PoPy](https://product.popypkpd.com/), Pumas [@rackauckas2020accelerated], and a number of [R libraries](https://cran.r-project.org/web/views/Pharmacokinetics.html). These packages provide an extensive toolkit for PKPD modelling, but are challenging to use for research into novel methodologies for PKPD modelling and inference, as most of them are closed source software packages. The notable exceptions are Scipion PKPD and the R libraries, which share their source code on GitHub.

[Chi](https://chi.readthedocs.io/en/latest/index.html) is an easy-to-use, open-source Python package for the modelling of PKPD processes. It is targeted at PKPD modellers on all levels of programming expertise. For modellers with an interest in methodological research, [Chi](https://chi.readthedocs.io/en/latest/index.html)'s modular, open source framework makes it easy to extend or replace individual components of PKPD models, as well as investigate their advantages and limitations. The straightforward evaluation of log-likelihoods and log-posteriors, and their sensitivities also facilitates the estimation of parametric and structural uncertainty [@cole2014maximum; @gelman2014understanding; @augustin2022treatment].

# References