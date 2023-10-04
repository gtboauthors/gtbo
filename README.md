# Welcome to GTBO.

[![pytest](https://github.com/gtboauthors/gtbo/actions/workflows/pytest.yml/badge.svg)](https://github.com/gtboauthors/gtbo/actions/workflows/pytest.yml)

You find the documentation for the GTBO package [here](https://gtboauthors.github.io/gtbo/gtbo.html).
You find the repository for the paper [here](https://github.com/gtboauthors/gtbo).

This package contains the code for the paper "High-dimensional Bayesian Optimization with Group Testing".

> We would like to give credit to the authors of [Noisy Adaptive Group Testing using Bayesian Sequential Experimental Design](https://arxiv.org/abs/2004.12508) as large parts of the group testing code is based on [their implementation](https://github.com/google-research/group_testing/).

## Installation

First, clone the project and go into the project root directory

```
git clone git@github.com:gtboauthors/gtbo.git
```

GTBO uses [poetry](https://python-poetry.org/) for dependency management and requires at least Python 3.9.
We refer to [pyenv](https://github.com/pyenv/pyenv) for managing several different Python versions.
Assuming you have both installed, you can install GTBO as follows

```
pyenv shell 3.11.3                  # switch to Python 3.11
poetry env use -- $(which python)   # create a new Poetry environment with Python 3.11
poetry install                      # install the project
```

If you have at least Python 3.9 and don't want to use pyenv,

```
poetry install
```

should also work.

### Singularity container

We provide a [Singularity](https://docs.sylabs.io/guides/latest/user-guide/) container to install the project
conveniently.
After installing Singularity, you can build the container with

```
sudo singularity build GTBO.sif singularity_container
```

After building the container, you can run it e.g. as follows:

```bash
singularity run --nv \
        --bind configs:/bs/gtbo/exp_configs \ # bind a directory with gin config files
        --bind results:/bs/gtbo/results \     # bind a results directory
         GTBO.sif \ 
        --gin-files exp_configs/gtbo.gin \    # path to gin config file
        --gin-bindings \                      # you can overwrite the settings in above file explicitly
        \"GTBO.benchmark = '@HartmannEffectiveDim()'\" \ # benchmark to run, options are @HartmannEffectiveDim, @BraninEffectiveDim, @LevyEffectiveDim, @GriewankEffectiveDim, @Mopta08, @LassoDNA 
        \'GTBO.maximum_number_evaluations = 1000\'
```

### Additional benchmarks

The real-world benchmarks require cloning an external repository parallel to the GTBO repository:.
This is done automatically when using the Singularity container.

```
git clone https://github.com/LeoIV/BenchSuite.git
```

## `gin` configuration

This project uses [gin-config](https://github.com/google/gin-config) for configuration management.
An example configuration with extensive comments is provided in `gtbo/configs/default.gin`:

```
import gtbo.gtbo
import gtbo.benchmarks
import gtbo.group_testing.tester
import gtbo.group_testing.ground_truth_evaluator
import gtbo.gaussian_process


EffectiveDimBoTorchBenchmark.noise_std = 0.01   # the standard deviation of the Gaussian white noise of the benchmark
EffectiveDimBoTorchBenchmark.drift = 0.0        # the synthetic 'drift', i.e., how much each inactive dimension affects the function value
EffectiveDimBoTorchBenchmark.effective_dim = 6  # the effective dimensionality of synthetic benchmarks supporting this attribute

#
# Macros
#
FUNCTION_DIMENSIONALITY = 30 # the (ambient) dimensionality of synthetic benchmarks
DTYPE = "float64"            # the dtype to use for reals
DEVICE = "cpu"               # the device to use ('cpu' or 'cuda')

#
# log level
#
GTBO.logging_level = "INFO" # possible values are "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

#
# Group tester configuration
#
GroupTester.gt_iterations = 2                       # max iters of the group testing (GT) phase
GroupTester.maximum_mutual_information_loss = 0.01  # when batching, continue until we either have max_groups_per_iteration groups or we find a group whose mutual information is maximum_mutual_information_loss*100% worse than the best found
GroupTester.test_only_non_converged_lb = 0.1e-4     # we only check individuals with a marginal > this in the GT phase
GroupTester.test_only_non_converged_ub = 0.95       # we only check individuals with a marginal < this in the GT phase
GroupTester.upper_convergence_threshold = 0.9       # marginals larger this are considered converged
GroupTester.prior_activeness_rate = 0.05            # prior probability of an element of being active
GroupTester.n_particles = 10000                     # number of particles of the SMC samples
GroupTester.n_initial_groups = 3                    # initial random groups in the GT phase
GroupTester.max_groups_per_iteration = 5            # when using batching in the GT phase, max batch size
GroupTester.lower_convergence_threshold = 1e-3      # marginals smaller this are considered converged
GroupTester.activeness_threshold = 0.05             # individuals with probability larger this are considered 'active variables'

#
# GroundTruthEvaluator configuration
#
GroundTruthEvaluator.n_default_samples = 10 # number of times we evaluate the default before starting GT

#
# fit_mll configuration
#
fit_mll.max_cholesky_size = 1000 # maximum number of points for the Cholesky decomposition

#
# robust_optimize_acqf configuration
#
robust_optimize_acqf.raw_samples = 4096 # number of raw_samples for optimize_acq (check BoTorch config)
robust_optimize_acqf.num_restarts = 16  # number of restarts for optimize_acq (check BoTorch config)

# get_gp configuration
get_gp.active_prior_parameters = (0, 1)     # mean and std for the LogNormalPrior for active vars
get_gp.inactive_prior_parameters = (3, 1)   # mean and std for the LogNormalPrior for inactive vars

#
# GTBO configuration
#
GTBO.benchmark = @HartmannEffectiveDim()    # the benchmark to run
GTBO.number_initial_points = 5              # the number of initial points in the DOE phase
GTBO.results_dir = "results"                # directory to write the results to
GTBO.maximum_number_evaluations = 200       # maximum overall number of function evaluations
GTBO.device = %DEVICE                       # the device to use
GTBO.dtype = %DTYPE                         # the dtype to use
GTBO.retrain_gp_from_scratch_every = 10     # we retrain the gp from scratch every couple iterations instead of initializing it with the previous solution

#
# Benchmark configuration
#
HartmannEffectiveDim.dim = %FUNCTION_DIMENSIONALITY
BraninEffectiveDim.dim = %FUNCTION_DIMENSIONALITY
LevyEffectiveDim.dim = %FUNCTION_DIMENSIONALITY
GriewankEffectiveDim.dim = %FUNCTION_DIMENSIONALITY
```

The maybe most relevant parameter to change is `GTBO.benchmark`, changing the problem tackled by GTBO.