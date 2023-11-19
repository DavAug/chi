.. _Pints: https://github.com/pints-team/pints

.. currentmodule:: chi

**************
Quick overview
**************

The quick overview displays some of Chi's main features to help you decide
whether Chi is the right software for you. The running example for this
overview will be a 1-compartment pharmacokinetic model (any other deterministic
time series model would have been an equally good choice, but the 1-compartment
pharmacokinetic model happens to be predefined in Chi's model library;
:class:`chi.library.ModelLibrary`). The model is defined by an ordinary
differential equation for the drug amount and an algebraic equation for the
drug concentration

.. math::
    \frac{\mathrm{d}a}{\mathrm{d}t} = -k_e a,
    \quad c = \frac{a}{v},
    \quad a(t=0) = a_0,

where :math:`a` is the drug amount in the compartment, :math:`t` is the time,
:math:`k_e` is the elimination rate of the drug, :math:`c` is the drug
concentration, :math:`v` is the volume of the compartment and :math:`a_0` is
the initial drug amount (note that during this quick overview we will omit
units for simplicity).

Simulating a mechanistic model
******************************

Using the 1-compartment pharmacokinetic model from Chi's model library,
simulation of the model simply involves specifying the set of model parameters
:math:`\psi = (a_0, v, k_e)` and the time points of interest.

.. literalinclude:: code/1_simulation_1.py
    :lines: 285-301

The simulation returns a :class:`numpy.ndarray` with the simulated drug
concentrations at the specified times
:math:`[c(t=0), c(t=0.2), c(t=0.5), c(t=0.6), c(t=1)]`.

.. code-block:: bash

    >>> result
    array([[5.        , 4.09359579, 3.03157658, 2.74324885, 1.83903834]])

The simulation results are of shape ``(n_outputs, n_times)`` which in this case
is ``(1, 5)`` because we simuated the drug concentration for 5 different time
points.
For details on how to implement your own :class:`chi.MechanisticModel` and
many other details concerning mechanistic models in Chi, we refer to
section :doc:`mechanistic_model`.

Visualisation of the simulation
*******************************

The results of the simulation can be visualised with any of the standard Python
visualisation libraries (e.g. matplotlib, seaborn, etc.).
We will use plotly in this overview. Below is an example of how to visualise
the simulation results, where we also made the time course of the drug
concentration slightly more interesting by administering bolus doses to the
compartment in intervals of 0.5

.. literalinclude:: code/1_simulation_1.py
    :lines: 305-329

.. raw:: html
   :file: images/1_simulation_1.html


Simulation of measurements
**************************

Measurements are in Chi modelled using a :class:`chi.ErrorModel` on top of the
:class:`chi.MechanisticModel` . Formally this takes the form of

.. math::
    y(\psi, t) = \bar{y}(\psi, t) + \epsilon (\bar{y}, \sigma, t),

where :math:`y` models the measurements and :math:`\epsilon` models the
residual error of the mechanistic model output with respect to the
measurements. :math:`\sigma` are the error model parameters.

The simplest error model is a Gaussian error model
:math:`\epsilon \sim \mathcal{N}(\cdot | 0, \sigma ^2)`. So let us use a
Gaussian error model to simulate measurements of the drug concentration using
the simulation results from the previous section (we will down sample the
measurement times relative to the evaluation times of the mechanistic model for
aestetic reasons)

.. literalinclude:: code/1_simulation_1.py
    :lines: 339-373

.. raw:: html
   :file: images/1_simulation_2.html

For details on how to implement a :class:`chi.ErrorModel` and
many other details concerning error models in Chi, we refer to the API
reference :doc:`../api/error_models`.


Inference of model parameters
*****************************

While the simulation of mechanistic model outputs and measurements is an
interesting feature of Chi in its own right, the inference of model parameters
from real-world measurements is arguably even more interesting. In this
overview, we will use the simulated measurements from above to infer model
parameters, but in practice the simulated measurements may be straightforwardly
replaced by real-world measurements.

Inference in Chi leverages the fact that the mechanistic model and the
error model define a distribution for the modelled measurements :math:`y`.
In the case of a Gaussian error model, as in the previous example,
the distribution of the measurements is (yet again) a Gaussian distribution
whose mean is equal to the mechanistic model output

.. math::
    p(y | \psi , \sigma , t) = \mathcal{N}\left(
        y \, |\, \bar{y}(\psi, t), \sigma ^2
        \right) .

This distribution can be used to define the log-likelihood of different
parameter values for the measurements
:math:`\mathcal{D}=\{ (y_1, t_1), (y_2, t_2), \ldots (y_n, t_n)\}`

.. math::
    \log p(\mathcal{D} | \psi , \sigma) =
        \sum _{j=1}^n \log p(y_j | \psi , \sigma , t_j).

In Chi the log-likelihood of model parameters can be defined using
:class:`chi.LogLikelihood`. A :class:`chi.LogLikelihood` defines the
distribution of the measurements using a :class:`chi.MechanisticModel` and a
:class:`chi.ErrorModel` , and couples the
distribution to the measurements as defined above. The log-likelihood of
different parameter values can now be evaluated using the
:meth:`chi.LogLikelihood.__call__` method

.. literalinclude:: code/1_simulation_1.py
    :lines: 379-393

.. code-block:: bash

    >>> score_1
    -86.14214936785024
    >>> score_2
    10.324673731287309


We can see that the data-generating parameter values have a larger
log-likelihood than the made-up parameter values (which should intuitively make
sense). For details on how to define a :class:`chi.LogLikelihood` and
many other details concerning log-likelihoods in Chi, we refer to the API
reference :doc:`../api/log_pdfs`.

Maximum likelihood estimation
-----------------------------

A popular approach to infer model parameters that best describe the
measurements is to find the parameter values that maximise the log-likelihood,
a.k.a. maximum likelihood estimation

.. math::
    \hat{\psi}, \hat{\sigma} =
        \mathop{\text{arg}\,\text{max}}_{\psi, \sigma}
        \log p(\mathcal{D} | \psi , \sigma).

In Chi the maximum likelihood estimates of the model parameters can be found
using any of the standard Python optimisation libraries such as scipy.optimize.
We will use Pints_ and its implementation of the Covariance Matrix
Adaption-Evolution Strategy (CMA-ES) optimisation algorithm,
:class:`pints.CMAES`, to optimise the log-likelihood

.. literalinclude:: code/1_simulation_1.py
    :lines: 397-402

.. code-block:: bash

    >>> parameters_mle
    array([10.26564936, 2.01524534, 1.00148417, 0.18456719])
    >>> score
    10.832126448890321

The optimisation algorithm finds a set of parameter values that is different,
but close to the data-generating parameters. Note, however, that the inferred
parameters have a larger log-likelihood than the data-generating parameters,
which may seem surprising at first. How can
a different set of parameters be more likely to describe the measurements than
the set of parameters that generated the measurements?
A thorough discussion of the shortcomings of maximum likelihood estimation
is beyond the scope of this overview, but let us plot the
mechanistic model output for the inferred parameters to confirm that
the inferred model is indeed marginally closer to the measurements

.. literalinclude:: code/1_simulation_1.py
    :lines: 407-415

.. raw:: html
   :file: images/1_simulation_3.html


Bayesian inference
------------------

The reason why maximum likelihood estimation can differ from the
data-generating parameters is that a finite number of
measurements do not uniquely define the data-generating parameters of a
probabilistic model. The
maximum likelihood estimates *can* therefore be far away from the
data-generating parameters just because the realisation of the measurement
noise happens to make other parameter values seem more likely than the
data-generating values (in this example the maximum likelihood estimates are
clearly quite good).
In other words, a finite number of measurements
leaves uncertainty about the model parameters, a.k.a. *parametric uncertainty*.
While for real-world measurements the notion of
data-generating parameters may seem alien since models only
approximate the real data-generating processes, we can generalise the notion of
data-generating parameters to being the effective set of parameter values that
capture the most about the data-generating process within the limitations of
the model approximation. Here, the maximum likelihood estimates can analogously
differ substantially from the sought after data-generating parameters.

In Chi the uncertainty of parameter estimates can be estimated using Bayesian
inference. In Bayesian inference Bayes' rule is used to define a distribution
of likely parameter values conditional on the measurements, a.k.a. the
posterior parameter distribution

.. math::
    \log p(\psi, \sigma | \mathcal{D}) =
        \log p(\mathcal{D} | \psi , \sigma) + \log p(\psi , \sigma )
        + \text{constant}.

Here, :math:`\log p(\psi, \sigma | \mathcal{D})` is the log-posterior,
:math:`\log p(\mathcal{D} | \psi , \sigma)` is the log-likelihood as defined
above and :math:`\log p(\psi , \sigma )` is the log-prior distribution of the
model parameters. The prior distribution of the model parameters is a
modelling choice and captures prior knowlegde about the model parameters.

In Chi the log-posterior can be defined using :class:`chi.LogPosterior` which
is instantiated with a :class:`chi.LogLikelihood` and a :class:`pints.LogPrior`.
For the sake of this overview, we will use uniform priors that constrain the
parameters to values between 0 and 20. In practice, informative priors are
likely better choices. The resulting log-posterior can now be evaluated similar
to the :class:`chi.LogLikelihood` using :meth:`chi.LogPosterior.__call__`

.. literalinclude:: code/1_simulation_1.py
    :lines: 420-431

.. code-block:: bash

    >>> score_1
    -104.56283011180261
    >>> score_2
    -8.096007012665059

For details on how to define a :class:`chi.LogPosterior` and
many other details concerning log-posteriors in Chi, the API
reference :doc:`../api/log_pdfs`.

While the :class:`chi.LogPosterior` allows us to evaluate the log-posterior
up to the constant term for different parameter values it does not yet tell us
how the posterior distribution of likely parameter values looks like.
This distribution can be inferred from a :class:`chi.LogPosterior` using e.g.
Markov Chain Monte Carlo (MCMC) sampling algorithms. Below we will use Pints_'
implementation of Haario and Bardenet's Adaptive Covariance Matrix MCMC
algorithm, :class:`pints.HaarioBardenetACMC`, to infer the posterior
distribution.

.. literalinclude:: code/1_simulation_1.py
    :lines: 435-439

The inferred posterior distributions can now be compared to the data-generating
parameters.

.. literalinclude:: code/1_simulation_1.py
    :lines: 443-552

.. raw:: html
   :file: images/1_simulation_5.html

For details on how to infer posterior distributions in Chi and
many other details on MCMC sampling, we refer to section
:doc:`fitting_models_to_data`.

Simulating a population model
*****************************

Above we have seen how Chi may be used to infer model parameters from
measurements. To this end, we assumed that measurements originate
from a data-generating process that is approximated by a mechanistic model,
an associated error model and a set of data-generating parameters.
For measurements of many biological systems this
is, however, a too naïve assumption, as measurements are often taken from
different, non-identical entities, ranging from measurements across
cells to measurements across patients. In these cases, it is well established
that differences between individual entities may lead to additional variability
in the measurements that cannot be attributed to measurement noise.
To separate this inter-individual variability from measurement noise, we need
to extend our modelling framework.
A natural extension is to assume that differences
across individuals can be modelled by different data-generating parameter
values for each individual. Below we illustrate how inter-individual
variability may mainifest in measurements from 3 patients with different
data-generating parameters.

.. literalinclude:: code/1_simulation_2.py
    :lines: 586-673

.. raw:: html
   :file: images/1_simulation_6.html

In this case, we assume for simplicity that the
inter-individual variability in drug concentration measurements originates from
different drug elimination rates, :math:`k_e` (a more realistic model would
also include inter-individual variations in the compartment volume :math:`v`).

Defining a population model
---------------------------

When sufficiently informative measurements are available for each indiviudal,
we can use the above introduced inference approaches to learn the
data-generating parameters for each individual separately. However, when
measurements are of limited availability it is beneficial to infer the model
parameters from all available data simultaneously. This requires us to
explicitly model the inter-individual variability.
A popular approach to model inter-individual variability is **non-linear
mixed effects** modelling, where the parameters of individuals are modelled as
a composite of fixed effects :math:`\mu` and random effects :math:`\eta`

.. math::
    \psi = \mu _{\psi} + \eta _{\psi} \quad \text{and}\quad
    \sigma = \mu _{\sigma} + \eta _{\sigma},

Here, :math:`\mu _{\psi}` and :math:`\mu _{\sigma}` are constants that model
the typical parameter values across individuals, and
:math:`\eta _{\psi}` and :math:`\eta _{\sigma}` are random variables that
model the inter-individual differences.
Note that this assumes that the data-generating parameters of individuals
follow a probabilistic distribution, and that any given individual is just a
random sample from this distribution.
While the mixed-effects picture can
be useful to develop some intuition for population models, an alternative
picture is to focus on the distribution of parameters that is
defined by the mixed-effects model

.. math::
    p(\psi, \sigma | \theta),

where :math:`\theta` are the population model parameters.
The population distribution picture has the advantage that we can define
the population distribution, :math:`p(\psi, \sigma | \theta)`, directly,
while it is not always trivial to workout the distribution of :math:`\psi`
from the distribution of :math:`\eta`. In chi we will therefore adopt the
population distribution picture (note however, that this is just a different
representation of non-linear mixed effects modelling and the approaches are
equivalent).
In the remainder, we will for notational ease include :math:`\sigma` in the
definition of
:math:`\psi`, i.e. :math:`(\psi , \sigma) \rightarrow \psi`. We will also make
the hierarchy between the model parameters explicit by refering to
:math:`\psi` as bottom-level or individual parameters and
to :math:`\theta` as top-level or population parameters.

With the notion of a population model, we can now construct a data-generating
model that explicitly models inter-individual variability in measurements

.. math::
    p(y, \psi | \theta , t) = p(y | \psi , t)\, p(\psi | \theta),

where :math:`p(y | \psi , t)` models the measurements of an individual with
data-generating parameters :math:`\psi`, and :math:`p(\psi | \theta)` models
the distribution over data-generating parameters in the population.

To make the concept of a population model less abstract let us revisit the
1 compartment pharmacokinetic model from above where the elimination rate
varies across patients, and the initial drug amount :math:`a_0`, the
compartment volume :math:`v` and the measurement noise :math:`\sigma` are the
same for all patients. Since all parameters but the elimination rate are the
same across patients, the population model takes the form

.. math::
    p(\psi | \theta) =
        \delta (a_0 - \theta _{a_0})\, \delta (v - \theta _{v})\,
        p(k_e | \theta _{k_e})\, \delta (\sigma - \theta _{\sigma}),

where :math:`\delta (x)` is Dirac's delta distribution which has non-zero
probability only for :math:`x=0`. In this example, this ensures that all
patients have the same initial drug amount :math:`a_0 = \theta _{a_0}`, the
same compartment volume :math:`v = \theta _{v}` and the same measurement noise
:math:`\sigma = \theta _{\sigma}`. In Chi delta distributions
can be implemented with a :class:`chi.PooledModel`. The distribution of the
elimination rate in the population, :math:`p(k_e | \theta _{k_e})`, is a
modelling choice. The simplest, sensible choice for
:math:`p(k_e | \theta _{k_e})` is a lognormal distribution,
:class:`chi.LogNormalModel`, which assumes that
the logarithm of :math:`k_e` is normally distributed (in the mixed-effects
picture: :math:`\log k_e = \mu _{k_e} + \eta _{k_e}` with
:math:`\mathcal{N}(\eta _{k_e} | 0, \sigma _{k_e}^2)`).
This ensures that :math:`k_e` can only assume positive values (choosing a
Gaussian distribution for :math:`k_e` makes in this case no sense, because
negative elimination rates would increase the drug amount in the compartment
over time)

.. math::
    p(\psi | \theta) =
        \delta (a_0 - \theta _{a_0})\, \delta (v - \theta _{v})\,
        \mathrm{LN}(k_e | \mu _{k_e}, \sigma _{k_e}) \,
        \delta (\sigma - \theta _{\sigma}).

The resulting model has 5 population parameters
:math:`\theta = (\theta _{a_0}, \theta _{v}, \mu _{k_e}, \sigma _{k_e}, \theta _{\sigma})`.

In Chi we can define this population model from instances of
:class:`chi.PooledModel` and :class:`chi.LogNormalModel` using
:class:`chi.ComposedPopulationModel`. From the population model we can then
simulate the distribution of the individual parameters in the population via
sampling. Below we have chosen the values of the population parameters
:math:`\theta` such that the 3 patients from above could have feasibly been
sampled from the population distribution.

.. literalinclude:: code/1_simulation_2.py
    :lines: 680-802

.. raw:: html
   :file: images/1_simulation_7.html

The dashed lines illustrate the elimination rates of the 3 simulated patients
from above and the grey distributions illustrate the distirbution of
patient parameters in the population. We omit plotting the amount, volume and
noise parameters of the 3 patients as they are all the same
and uniquely defined by the population distribution.

The inter-individual variability of drug concentration measurements in the
population can be simulated by measuring the drug concentration for each
sampled set of
individual parameters, i.e. by simulating measurements for the simulated
patients. Below we will illustrate the 5th to 95th percentile
range of the population distribution together with the measurements of the 3
patients from above.

.. literalinclude:: code/1_simulation_2.py
    :lines: 807-858

.. raw:: html
   :file: images/1_simulation_8.html

For details on how to implement a :class:`chi.PopulationModel` and
many other details concerning population models in Chi, the API
reference :doc:`../api/population_models`.

Hierarchical inference
**********************

The simulation of population models is interesting in its own right
and may, for example, be used to understand the inter-individual variation of
the modelled dynamics once estimates for the population
parameters exist.
However, as claimed earlier, the population model also allows us to infer
parameters from measurements that originate from different, non-identical
entities.
In particular, the population model allows us to define a hierarchical
log-posterior from
which we can estimate the individual parameters and the population parameters,
simultaneously. To this end, let us first define the hierarchical
log-likelihood which is a straightforward extension of the above introduced
log-likelihood

.. math::
    \log p(\mathcal{D}, \Psi | \theta) =
        \sum _i \log p(\mathcal{D} _i | \psi _i)
        + \sum _i \log p(\psi _i | \theta).

Here, :math:`\mathcal{D}=\{\mathcal{D}_i\}` are the data and
:math:`\mathcal{D}_i = \{(y_{ij}, t_{ij})\}` are the measurements of
individual :math:`i`. Just like in section 1.4, we use :math:`j` to index
measurement events. :math:`\Psi = \{ \psi _i \}` denotes the
bottom-level parameters across all individuals with :math:`\psi _i` being the
parameters of individual :math:`i`. Thus, the hierarchical
log-likelihood has two contributions: 1. a contribution from the cumulative
log-likelihood of the bottom-level parameters to describe the
observations; and 2. a
contribution from the log-likelihood of the population parameters to describe
the distribution of the bottom-level parameters.
In Chi a hierarchical log-likelihood may be
defined using :class:`chi.HierarchicalLogLikelihood` with a list of
bottom-level :class:`chi.LogLikelihood` instances and a
:class:`chi.PopulationModel`.

.. literalinclude:: code/1_simulation_2.py
    :lines: 865-876

The hierarchical log-likelihood is a function of both :math:`\Psi`
and :math:`\theta`, so in principle we could find maximum likelihood estimates
by maximising :math:`\log p(\mathcal{D}, \Psi | \theta)`, just like in section
1.4.
However, maximising
:math:`\log p(\mathcal{D}, \Psi | \theta)` will in general not find estimates
of the data-generating :math:`\Psi` and :math:`\theta` for
reasons that deserve a more thorough discussion than possible in this overview.
Observe, however,
that the 2. term of the hierarchical log-likelihood receives larger
contributions for narrow population distributions.
This introduces a bias towards understimating the population variance
which affects the estimates of the population parameters.
In addition the individual parameters will be biased away from the
data-generating parameters towards the modes of the population distribution
(a.k.a. shrinkage).
In some cases these biases are not a concern, especially when the
measurements leave very little uncertainty about the
bottom-level parameters.
In general, they will, however, influence the inference results.

A partial remedy for these shortcomings is Bayesian inference, where
analogously to above we can use Bayes' rule to define a hierarchical
log-posterior

.. math::
    \log p(\Psi , \theta | \mathcal{D}) =
        \log p(\mathcal{D}, \Psi | \theta) + \log p(\theta )
        + \text{constant},

where :math:`\log p(\theta )` is the log-prior of the population parameters.
The posterior distribution :math:`p(\Psi , \theta | \mathcal{D})` is able
to mitigate the biases of the hierarchical log-likelihood by inferring a
distribution of likely parameters. This will not remove the biases entirely, but it will
make sure that the data-generating parameters are captured by the posterior
distribution (as long as the prior choice is appropriate).

In Chi we can define a hierarchical log-posterior
using :class:`chi.HierarchicalLogPosterior` which takes a
:class:`chi.HierarchicalLogLikelihood` and a :class:`chi.LogPrior` as inputs.
The posterior distribution can then be inferred using e.g. MCMC sampling, just
like in section 1.4.2.
Below we infer the parameters of the 3 patients from above using hierarchical
inference and compare the posteriors to the data-generating parameters.

.. literalinclude:: code/1_simulation_2.py
    :lines: 880-1062

.. raw:: html
   :file: images/1_simulation_9.html

In addition to the data-generating parameters of the patients, the hierarchical
inference also provides estimates for the population-level parameters.
Below we only illustrate the log mean and the log standard deviation of the
elimination rate in the population, because the other parameters are pooled and
therefore the population parameters are equivalent to the bottom-level
parameters.

.. literalinclude:: code/1_simulation_2.py
    :lines: 1067-1122

.. raw:: html
   :file: images/1_simulation_10.html

Note that in this case the posteriors of :math:`\mu _{k_e}` and
:math:`\sigma _{k_e}` are largely dominated by the prior distribution as 3
patients are not informative enough to substantially influence the posterior
distribution.

This concludes the quick overview of chi. You now know how to use Chi to
simulate mechanistic model outputs, measurements and population models. You
also know how to use chi to infer model parameters using maximum likelihood
estimation or Bayesian inference, and you know how to use Chi to do
hierarchical inference. There are a few things that the quick overview has not
touched on. The two most interesting things
being: 1. the :class:`chi.ProblemModellingController` which is a convenience
class that helps you to build your models more easily, especially when
measurement times and dosing regimens vary across individuals; 2. the
:class:`CovariatePopulationModel` which allows you to define population models
where some of the inter-individual is explained by covariates.

We hope you enjoyed this overview and have fun working with Chi!