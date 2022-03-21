.. _Pints: https://github.com/pints-team/pints

.. currentmodule:: chi

**************
Quick overview
**************

The quick overview displays some of chi's main features to help you decide
whether chi is the right software for you. The running example for this
overview will be a 1-compartment pharmacokinetic model (any other deterministic
time series model would have been an equally good choice, but the 1-compartment
pharmacokinetic model happens to be predefined in chi's model library;
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

Using the 1-compartment pharmacokinetic model from chi's model library,
simulation of the model simply involves specifying the set of model parameters
:math:`\psi = (a_0, v, k_e)` and the time points of interest

.. literalinclude:: code/1_simulation_1.py
    :lines: 285-301

The simulation returns a :class:`numpy.ndarray` with the simulated drug
concentrations at the specified times
:math:`[c(t=0), c(t=0.2), c(t=0.5), c(t=0.6), c(t=1)]`

.. code-block:: console

    >>> result
    array([[5.        , 4.09359579, 3.03157658, 2.74324885, 1.83903834]])

The simulation results are of shape ``(n_outputs, n_times)`` which in this case
is ``(1, 5)`` because we simuated the drug concentration for 5 different times.
For details on how to implement your own :class:`chi.MechanisticModel` and
many other details concerning the mechanistic models in chi, we refer to
section :doc:`mechanistic_model`.

Visualisation of the simulation
*******************************

The results of the simulation can be visualised with any of the standard Python
visualisation libraries (e.g. matplotlib, seaborn, etc.).
We will use plotly in this overview. Below is an example of how to visualise
the simulation results, where we also made the results slightly more
interesting by administering bolus doses to the compartment in intervals of 0.5

.. literalinclude:: code/1_simulation_1.py
    :lines: 305-329

.. raw:: html
   :file: images/1_simulation_1.html


Simulation of measurements
**************************

Measurements are in chi modelled using a :class:`chi.ErrorModel` on top of the
:class:`chi.MechanisticModel` of the form

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
many other details concerning error models in chi, we refer to section
:doc:`error_model`.


Inference of model parameters
*****************************

While the simulation of mechanistic model outputs and measurements is an
interesting feature of chi in its own right, the inference of model parameters
from real-world measurements is arguably even more interesting. For simplicity,
we will use the simulated measurements to infer the model
parameters, but in practice the simulated measurements can be straightforwardly
replaced by real-world measurements.

Inference in chi leverages the fact that the modelled measurements :math:`y`
follow a distribution that is defined by the mechanistic model and error model.
In the case of a Gaussian error model, as in the previous example, the
distribution of the measurements is also a Gaussian distribution whose mean is
equal to the mechanistic model output

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

In chi the log-likelihood of model parameters can be defined using
:class:`chi.LogLikelihood`. A :class:`chi.LogLikelihood` defines the
distribution of the measurements using a :class:`chi.MechanisticModel` and a
:class:`chi.ErrorModel` for each mechanistic model output, and couples the
distribution to the measurements as defined above. The log-likelihood of
different parameter values can now be evaluated using the
:meth:`chi.LogLikelihood.__call__` method

.. literalinclude:: code/1_simulation_1.py
    :lines: 379-393

.. code-block:: console

    >>> score_1
    -86.14214936785024
    >>> score_2
    10.324673731287309


We can see that the data-generating parameter values have a larger
log-likelihood than the made-up parameter values (which should intuitively make
sense). For details on how to define a :class:`chi.LogLikelihood` and
many other details concerning log-likelihoods in chi, we refer to section
:doc:`log_likelihood`.

Maximum likelihood estimation
-----------------------------

A popular approach to infer the model parameters that best describe the
measurements is to find the parameter values that maximise the log-likelihood,
a.k.a. maximum likelihood estimation

.. math::
    \hat{\psi}, \hat{\sigma} =
        \mathop{\text{arg}\,\text{max}}_{\psi, \sigma}
        \log p(\mathcal{D} | \psi , \sigma).

In chi the maximum likelihood estimates of the model parameters can be found
using any of the standard Python optimisation libraries such as scipy.optimize.
We will use Pints_ and its implementation of the Covariance Matrix
Adaption-Evolution Strategy (CMA-ES) optimisation algorithm to optimise the
log-likelihood

.. literalinclude:: code/1_simulation_1.py
    :lines: 397-402

.. code-block:: console

    >>> parameters_mle
    array([10.26564936, 2.01524534, 1.00148417, 0.18456719])
    >>> score
    10.832126448890321

The optimisation algorithm finds a set of parameter values that is different,
but close to the data-generating parameters. Note, however, that the inferred
parameters have a larger log-likelihood than the data-generating parameters,
which may seem surprising at first. How can
a different set of parameters be more likely to describe the measurements than
the set of parameters that generated the measurments?
A thorough discussion of the shortcomings of maximum likelihood estimation
is beyond the scope of this documentation, but let us plot the
mechanstic model output for the inferred parameters to confirm that
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
data-generating parameters (in this example
the maximum likelihood estimates are clearly quite good).
In other words, a finite number of measurements
leaves uncertainty about the model parameters, a.k.a. *parametric uncertainty*.
While for real-world measurements the notion of
data-generating parameters may seem alien since models only
approximate the real data-generating processes, we can generalise the notion of
data-generating parameters to being the set of parameters that capture the most
about the data-generating process within the limitations of the model
approximation. Here, the maximum likelihood estimates can analogously differ
significantly from the sought after data-generating parameters.

In chi the uncertainty of parameter estimates can be estimated using Bayesian
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

In chi the log-posterior can be defined using :class:`chi.LogPosterior` which
is instantiated with a :class:`chi.LogLikelihood` and a :class:`pints.LogPrior`.
For simplicity, we will use uniform priors that constrain the parameters to
values between 0 and 20. The log-posterior can be evaluated similar to the
:class:`chi.LogLikelihood` using :meth:`chi.LogPosterior.__call__`

.. literalinclude:: code/1_simulation_1.py
    :lines: 420-431

.. code-block:: console

    >>> score_1
    -104.56283011180261
    >>> score_2
    -8.096007012665059

For details on how to define a :class:`chi.LogPosterior` and
many other details concerning log-posetriors in chi, we refer to section
:doc:`log_posterior`.

While the :class:`chi.LogPosterior` allows us to evaluate the log-posterior
up to constant term for different parameter values it does not yet tell us
how the posterior distribution of likely parameter values looks like.
This distribution can be inferred from a :class:`chi.LogPosterior` using MCMC
sampling algorithms. Below we will use the implementation of Haario
and Bardenet's Adaptive Covariance Matrix Marcov Chain Monte Carlo,
:class:`pints.HaarioBardenetACMC` to infer the posterior distribution

.. literalinclude:: code/1_simulation_1.py
    :lines: 435-438

The inferred posterior distributions can now be compared to the data-generating
parameters

.. literalinclude:: code/1_simulation_1.py
    :lines: 442-551

.. raw:: html
   :file: images/1_simulation_5.html

Note that a discussion of how to analyse the convergence of MCMC chains
is beyond the scope of this overview, and we refer to section
:doc:`mcmc_sampling`.