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

[Chi](https://chi.readthedocs.io/en/latest/index.html) is an easy-to-use Python package for pharmacokinetic & pharmacodynamic (PKPD) modelling. We provide two interfaces to implement PKPD models: 1. a general purpose interface; and 2. an SBML interface [@hucka:2003]. PKPD models instantiated from SBML files automatically implement the administration of custom dosing regimens and the simulation of model sensitivities. We also provide a simple framework to extend PKPD models to nonlinear mixed effects (NLME) models, making the simulation of inter-individual variability of treatment responses possible.

In [Chi](https://chi.readthedocs.io/en/latest/index.html), model parameters can be estimated from data using Bayesian inference. We provide a simple interface to estimate posterior distributions of PKPD model parameters and NLME model parameters. [Chi](https://chi.readthedocs.io/en/latest/index.html) also implements filter inference, a novel inference approach which makes the estimation of NLME model parameters from snapshot time series data possible [@Augustin:2023].

For the sampling from posterior distributions, [Chi](https://chi.readthedocs.io/en/latest/index.html) uses Markov chain Monte Carlo (MCMC) algorithms implemented in the Python package [PINTS](https://pints.readthedocs.io/en/stable/) [@Clerx:2019].

# Statement of need

PKPD modelling has grown to be an integral part of pharmaceutical research [@SCHUCK:2015; @MORGAN:2018]. In the early phase of the drug development, PKPD models help to establish a (semi-)mechanistic understanding of pharmacological processes in preclinical models. These modelling results provide guidance in the transition to the clinical development phase [@LAVE:2016]. Between clinical trials, PKPD models help to predict the safety, the efficacy and the treatment response variability of different dosing strategies for future clinical trials. More recently, PKPD models are also considered to help with the individualisation of dosing regimens of otherwise difficult-to-administer drugs [@Augustin:20232].

[Chi](https://chi.readthedocs.io/en/latest/index.html) provides an easy-to-use Python framework for PKPD modelling, making it easier for researchers with a non-technical background to model pharmacological processes. In addition, [Chi](https://chi.readthedocs.io/en/latest/index.html)'s integrated Bayesian inference framework simplifies the estimation of parametric uncertainty and its effects on treatment response predictions.

# References