.. currentmodule:: chi

.. _Dataset_1 : https://github.com/DavAug/chi/blob/main/docs/source/getting_started/data/dataset_1.csv
.. _Pints: https://github.com/pints-team/pints

**********************************
Fitting mechanistic models to data
**********************************

In the previous tutorial, :doc:`mechanistic_model`, we have seen how we can
implement and simulate treatment response models in Chi. For example, using the same
1-compartment PK model from before, ``one_compartment_pk_model.xml``, we can simulate
the time course of drug concentration levels following repeated dose
adminstrations

.. literalinclude:: code/3_fitting_models_1.py
    :lines: 18-52

.. raw:: html
   :file: images/3_fitting_models_1.html

This ability to simulate treatment response is pretty cool in its own right,
but, at this point, our model has little to do with
real treatment responses. If our goal is to describe *real* treatment
responses, we need to somehow connect our model to reality.

The most common approach to relate models to real treatment responses is to
compare the model predictions to measurements. Below, we have prepared an example
dataset with drug concentration measurements. These drug concentrations were
measured after repeatedly adminstering 2 mg of a drug every 24 hours.
You can download the dataset from the following link, if you would like to
follow this tutorial: Dataset_1_.

.. csv-table:: Drug concentration measurements
   :file: code/dataset_1.csv
   :widths: 4, 12, 12, 12, 12, 12, 12, 12, 12
   :header-rows: 1

The dataset contains one column that identifies the measured individual (``ID``),
two columns that specify the time of a measurement or a dose administration
(``Time`` and ``Time unit``), three columns that specify the measured values
(``Observable``, ``Value``, ``Observable unit``), and three columns that specify
the dose administrations (``Dose``, ``Duration``, ``Dose unit``).

If we download the file and save it in the same directory as the Python script,
we can visualise the measurements by executing the below script

.. literalinclude:: code/3_fitting_models_1.py
    :lines: 82-101

.. raw:: html
   :file: images/3_fitting_models_2.html

The figure shows that the treatment response dynamics indicated by the measurements
is similar to treatment response simulated by the one-compartment PK model above.
But looking more closely at the magnitude of the values,
it appears that the measured values are much smaller
than the simulated ones. We can therefore conclude that, at this point, our
model does not provide an accurate description of the measured treatment
response.

To find a better description of the treatment response, we have two options:
1. we can try to find parameters values that make the model output describe the
measurements more closely; or 2. we can define a new mechanistic model and see
whether this new model is able to describe the measurements better. This tutorial
will be about how we can find better model parameters.

Estimating model parameters from data: Background
*************************************************

Before we can try to find parameter values that most closely describe the treatment response
measurements, we first need to agree on what we mean by
"*most closely*" for the relationship between the mechanistic model output and the measurements.
The most naive way to define this notion of closeness is to use the distance
between measurements and model outputs,
i.e. the difference between the measured value and the
simulated value. If we used
the distance to define closeness, the model parameters that most closely
describe the measurements would be those parameter values that make the mechanistic
model output perfectly match the measurements, leading to a distance of 0 ng/mL
for all measurements. However, as outlined in Sections 1.3 and 1.4 of the
:doc:`quick_overview`, measurements of treatment responses are imprecise and
noisy, and will therefore not perfectly represent the treatment response dynamics.
Consequently, if we were to perfectly match the model outputs to measurements,
we would actually end up with an inaccurate description of the treatment response
because we would be paying too much attention to the measurement noise.

One way to improve our notion of closeness is to incorporate the measurement
process into our computational model of the treatment response, thereby
explicitly stating that we do not expect the mechanistic model output to match
the measurements perfectly. In Chi, this can be done
using :class:`chi.ErrorModel` s. Error models promote the single value output
of mechanistic model simulations to a distribution of
values. This distribution characterises a
range of values around the mechanistic model output where measurements may be
expected.
For simulation, this distribution can be used to sample measurement values and
imitate the measurement process of real treatment responses, see
Section 1.3 in :doc:`quick_overview`. For parameter estimation, the
distribution can be used to quantify the likelihood with which the observed
measurements would have been produced by our model of the measurement process,
see Section 1.4 in :doc:`quick_overview`.

To account for measurement noise during the parameter estimation, we therefore
choose to quantify the closeness between the model output an the measurements
using likelihoods. Formally, the measurement process can be denoted by
:math:`p(y | \psi, t, r)`, where :math:`p` denotes the probability density of the
measurement distribution, :math:`y`
denotes the measurement value, :math:`\psi` denotes the model parameters,
:math:`t` denotes the time, and :math:`r` denotes the dosing
regimen. The likelihood of a single measurement :math:`((t_1, y_1), r)` is then
simply given by the value of the probability density evaluated at the measurement,
:math:`p(y_1 | \psi, t_1, r)`. This value depends on the chosen set of model
parameters, :math:`\psi`. The model parameters with the maximum likelihood are
the parameter values that most closely describe the measurements.

.. note::
    The measurement distribution, :math:`p(y | \psi, t, r)`, is uniquely defined
    by the mechanistic model output and the error model. For example for
    the 1-compartment model, we may denote the simulated drug concentration
    values by :math:`c(\psi, t, r)`, where the drug concentration values, :math:`c`, are
    a function of the model parameters, :math:`\psi = (a_0, k_a, k_e, v)`, the time,
    :math:`t`, and the dosing regimen, :math:`r`.

    1. If we choose a :class:`chi.GaussianErrorModel` to describe the difference
    between the model output and the measurements, we assume that measurements
    are distributed according to a Normal distribution around the model output

    .. math::
        p(y | \psi, t, r) = \frac{1}{\sqrt{2\pi \sigma ^2}}\mathrm{e}^{-\big(y - c(\psi, t, r)\big)^2 / 2\sigma ^2},

    where :math:`\sigma` defines the width of the distribution. For ease of notation,
    we extend the definition of the model parameters to include :math:`\sigma`,
    :math:`\psi = (a_0, k_a, k_e, v, \sigma)`.

    We can see that the model output
    defines the mean or Expectation Value of the measurement distribution.

    2. If we choose a :class:`chi.LogNormalErrorModel` to describe the difference
    between the model output and the measurements, we assume that measurements
    are distributed according to a lognormal distribution around the model output

    .. math::
        p(y | \psi, t, r) = \frac{1}{\sqrt{2\pi \sigma ^2}}\frac{1}{y}\mathrm{e}^{-\big(\log y - \log c(\psi, t, r) + \sigma / 2\big)^2 / 2\sigma ^2}.

    One can show that also for this distribution the model output defines the mean
    or Expectation Value of the measurement distribution.

    The main difference between the two distributions is the shape. The
    :class:`chi.GaussianErrorModel` is symmetrically distributed around the
    model output, while :class:`chi.LogNormalErrorModel` is scewed in such a
    way that measurements can never assume negative values. To visualise these
    differences, we recommend simulating many measurements with different
    error models, similar to Section 1.3 in :doc:`quick_overview`. But instead
    of choosing different times, sample all measurements at the same time. You
    can then histogram the samples, using for example ``go.Histogram``, as used
    in Section 1.4.2 in :doc:`quick_overview`, to visualise the shape of
    the probability density.


Assuming the independence of measurements, the likelihood for a dataset with :math:`n` measurements,
:math:`\mathcal{D}=((t_1, y_1), (t_2, y_2), \ldots, (t_n, y_n), r)`, is given
by the product of the individual likelihoods

.. math::
    p(\mathcal{D}| \psi ) = \prod _{j=1}^n p(y_n | \psi, t_n, r),

where *independence* refers to the assumption that measurements are
independently and identically distributed according to our model of the
measurement process (this does not have to be the case, and is especially unlikely
to be the case when our model does not describe the measurement process).

While it is easy to numerically maximise the likelihood and
to derive maximum likelihood estimates in Chi, see Section 1.4.1 in :doc:`quick_overview`,
the main objective of Chi is to support the Bayesian inference of model parameters.
Bayesian inference is conceptually different from maximum likelihood estimation
becuase it does not seek to find a single set of model parameters that "best"
describe the observed measurements. Instead, Bayesian inference acknowledges
the fact that noisy measurements cannot uniquely define
the best model parameters, and focuses on deriving a distribution of parameter values
consistent with the observed measurements, see Section 1.4.2 in :doc:`quick_overview`
for a more detailed discussion. This
distribution is derived from the likelihood using Bayes' rule

.. math::
    p(\psi| \mathcal{D} ) = \frac{p(\mathcal{D}| \psi )\, p(\psi)}{p(\mathcal{D} )},

where :math:`p(\psi)` denotes the prior distribution of the model parameters
which quantifies our belief of likely parameter values for the model prior
to the inference from data. Note that Bayes' rule is just based on the fact
that we can condition a joint probability distribution
of two random variables on either random variable:
:math:`p(A, B) = p(A | B) p(B) = p(B | A) p(A)`.

Estimating model parameters from data: The ProblemModellingController
***************************************************************************

In Chi, you can estimate model parameters from measurements by inferring posterior
distributions of parameter values using Markov chain
Monte Carlo (MCMC) algorithms. In this tutorial, we will use MCMC algorithms
implemented in the open source Python package
Pints_, but in principle you can also use implementations from other libraries.
In Sections 1.4.1 and 1.4.2 in the :doc:`quick_overview`,
we showed in some detail how we can define (log-)posterior distributions,
:class:`chi.LogPosterior`, in Chi for this purpose. Here, we want to show
how you can use the :class:`chi.ProblemModellingController` to more easily
construct the :class:`chi.LogPosterior`, provided the measurements are available
in a CSV format of the form of Dataset_1_.

Defining the log-posterior
^^^^^^^^^^^^^^^^^^^^^^^^^^

The tricky bit when implementing log-posteriors for treatment response models
is often that those log-posteriors do not only depend on the treatment
response measurements, :math:`((t_1, y_1), (t_2, y_2), \ldots, (t_n, y_n))`,
but that they also depend on the administered dosing regimen, :math:`r`.
This can make it tedious to define log-posteriors,
especially when you are inferring parameters across measurements of multiple
individuals with difference dosing regimens.

To simplify the process of constructing a :class:`LogPosterior`, we have
implemented the :class:`chi.ProblemModellingController`.
The :class:`chi.ProblemModellingController` facilitates the construction of
log-posteriors and reduces the workflow to a simple 4 steps approach:

- 1. Definition of the mechanistic model
- 2. Definition of the error model
- 3. Definition of the measurements
- 4. Definition of the prior distribution

In the below code block, we illustrate this workflow for the above drug
concentration dataset, Dataset_1_.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 27-64

The first four blocks in the code define the individual components of the
log-posterior: the mechanistic model, the error model, the data, and the prior
distribution. Note that the administration of the dosing regimen is set
before passing the mechanistic model to the :class:`ProblemModellingController`.

The prior distribution defines marginal distributions for the parameters, and
is implemented using Pints_. In Bayesian inference, we can use the prior
distribution to bias the inference
results towards feasible areas of the parameter space. In this
case, we have little prior knowledge about feasible parameter ranges, so we
choose relatively uninformative prior
distributions (below will be an illustration of the prior distribution).
Note that we seem to be specifying marginal prior distributions only for
4 of the 6 model parameters. This is because we fix the values of the initial drug
amounts to ``0`` prior to the inference
in the lines below, reflecting our knowledge that the subject had no prior
exposure to the drug before starting the trial. This reduces the number of model
parameters from 6 to 4. The fixing of model parmaters
is optional, but can sometimes improve the inference results when some model
parameters are already well understood.

For the remaining 4 parameters, only positive values make biological sense, so
we choose prior distributions that focus on positive values. For
two model parameters, the volume of distribution and the scale parameter,
negative or zero values are particularly bad as they will break the simulation
(a volume of zero causes a division by zero error, and the lognormal distribution
is only defined for positive sigma). We therefore use ``pints.LogNormalLogPrior``
to constrain those parameters to strictly positive values.

In the final block of the code, we define the log-posterior. In the first line,
we specify the mechanistic model and the error model. In the next line, we set
the dataset. Note that we need to use the ``output_observable_dict`` to map the
output variable of the model, ``global.drug_concentration``, to the Observable name
in the dataset, ``Drug concentration``. Other specifications are not required, and
dosing regimens are automatically set, when the dosing regimen related columns,
``Dose`` and ``Duration``, are present in the dataset. In the following line, we
fix the initial drug amounts to ``0`` using :meth:`ProblemModellingController.fix_parameters`.
You can use this method to fix any parameters of the model.
In the last two lines, we set the log-prior and implement the log-posterior using
the :class:`ProblemModellingController.get_log_posterior` method.

Inferring the posterior distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With this :class:`chi.LogPosterior` in place, we can infer the posterior
distribution using any MCMC algorithm of our choice. Recall that MCMC
algorithms infer distributions by sampling from them.
If we sample sufficiently many samples,
the histogram over the samples should converge to the posterior
distribution.
To illustrate this, we run an MCMC algorithm to infer the above
defined posterior distribution of the 1-compartment PK
model.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 68-74

In the code block, we use an MCMC algorithm implemented in Pints_, called
``pints.HaarioBardenetACMC``. For technical reasons that we will discuss below,
we run the algorithm three times for 20,000 iterations.
Note that we use the :class:`chi.SamplingController` to set the number of
runs, the number of iterations, and to run the sampling algorithm. The
:class:`chi.SamplingController`  can also do other things, such as running the
chains in parallel, but we will not go into this additional functionality in this
tutorial, and refer instead to the API reference.

Executing the above code block will spawns a response of the below form.
The left most column indicates the current
iteration of the MCMC algorithm. The other columns are details specific to the
MCMC algorithm that we will not go into now.

.. code-block:: bash

    Using Haario-Bardenet adaptive covariance MCMC
    Generating 3 chains.
    Running in sequential mode.
    Iter. Eval. Accept.   Accept.   Accept.   Time m:s
    0     3      0         0         0          0:00.0
    1     6      0         0.5       0.5        0:00.0
    2     9      0.333     0.667     0.333      0:00.0
    3     12     0.25      0.75      0.5        0:00.0
    20    63     0.714     0.571     0.476      0:00.0
    40    123    0.756     0.61      0.561      0:00.0
    60    183    0.738     0.475     0.59       0:00.0
    80    243    0.716     0.407     0.642      0:00.1
    100   303    0.693     0.406     0.653      0:00.1
    120   363    0.736     0.421     0.686      0:00.1
    .
    .
    .

When the three runs of the algorithm terminate, the inference is completed.
We can visualise the samples from the posterior distribution using the code
block documented at the end of this section (we move the code block to the end,
to avoid disrupting the flow of the tutorial with less relevant code snippets).

.. raw:: html
   :file: images/3_fitting_models_3.html

The left column of the figure shows the
samples drawn from the :class:`chi.LogPosterior` at each iteration of the
MCMC algorithm. The samples from the different runs are illustrated in different
colours: run 1 (green); run 2 (red); run 3 (blue). The first row shows the
samples of the absorption rate, the second row shows the samples of the elimination
rate, the third row shows the samples of the volume of distribution, and the
fourth row shows the samples of the scale parameter of the error model, sigma.
The right column of the
figure shows the histogram over the samples across runs
in orange, as well as the probability density of the prior distribution in black.

The orange distribution is the result of the inference -- the posterior distribution.
It contains all parameter values that are consistent with the drug concentration
measurements and our prior knowledge, assigning each set of parameter values
with a probability of being the data-generating parameter values. Noticably, the
figure shows that our prior knowledge and Dataset_1_ are insufficient to
conclude on a single set of parameter values (see posterior distribution).
Instead, the measurements only
allow us to refine our understanding of feasible parameter values. For example,
we can see in the second row of the figure that the marginal posterior distribution
substantially differs from the marginal prior distribution. This is because the
drug concentration measurements contain important information about the elimination rate, rendering
rates above 1.5 1/day or below 0.25 1/day as extremely unlikely for the
measured treatment response. However, the measurements are not conclusive enough
to reduce the distribution of feasible parameters to a single value. Still,
the inference was able to subtantially update our understanding of feasible
elimination rates.
In comparison, the measurements appear less informative about the absorption rate
(see row 1), given that the marginal posterior distribution of
the absorption rate is almost identical to its prior distribution.
We will have a closer look at an intuitive understanding of why the measurements
contain little information about
the absorption rate below. The take-away from this discussion is that inferring
distributions of parameter values consistent with the measurements is in many
ways more informative than estimating a single set of model parameters from
measurements. In a Bayesian setting, inferring posterior distributions also
allow us to supplement the information contained in the measurements by prior
knowledge of feasible model paramters.

The second take-away relates to the left column of the figure and concerns the
straightforward ability to assess the convergence of MCMC algorithms...

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 77-374

.. autosummary::

    chi.ErrorModel
    chi.GaussianErrorModel
    chi.LogNormalErrorModel
    chi.MultiplicativeGaussianErrorModel
    chi.ConstantAndMultiplicativeGaussianErrorModel
    chi.ReducedErrorModel
    chi.LogLikelihood
    chi.LogPosterior
    chi.SamplingController