"""
==================================================
Using the black-box optimization interface of SMAC
==================================================
"""

import logging

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.epm.partial_sparse_gaussian_process import PartialSparseGaussianProcess

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.optimizer.local_bo.rh2epm_boing import RunHistory2EPM4LogCostWithRaw
from smac.optimizer.local_bo.epm_chooser_boing import EPMChooserBOinG
from smac.epm.util_funcs import get_types, get_rng

# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from gpytorch.kernels.kernel import ProductKernel
from botorch.models.kernels.categorical import CategoricalKernel

from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors import LogNormalPrior, UniformPrior, HorseshoePrior

import torch

def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """
    x1 = x["x0"]
    x2 = x["x1"]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
cs.add_hyperparameters([x0, x1])

# Scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                     "runcount-limit": 100,  # max. number of function evaluations; for this example set to a low number
                     "cs": cs,  # configuration space
                     "deterministic": "true"
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = rosenbrock_2d(cs.get_default_configuration())
print("Default Value: %.2f" % def_value)

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")



types, bounds = get_types(scenario.cs, instance_features=None)


cont_dims = np.where(np.array(types) == 0)[0]
cat_dims = np.where(np.array(types) != 0)[0]

if len(cont_dims) > 0:
    exp_kernel = MaternKernel(2.5,
                              lengthscale_constraint=Interval(
                                  torch.tensor(np.exp(-6.754111155189306).repeat(cont_dims.shape[-1])),
                                  torch.tensor(np.exp(0.0858637988771976).repeat(cont_dims.shape[-1])),
                                  transform=None,
                                  initial_value=1.0
                              ),
                              ard_num_dims=cont_dims.shape[-1],
                              active_dims=tuple(cont_dims)).double()

if len(cat_dims) > 0:
    ham_kernel = CategoricalKernel(
        lengthscale_constraint=Interval(
            torch.tensor(np.exp(-6.754111155189306).repeat(cat_dims.shape[-1])),
            torch.tensor(np.exp(0.0858637988771976).repeat(cat_dims.shape[-1])),
            transform=None,
            initial_value=1.0
        ),
        ard_num_dims=cat_dims.shape[-1],
        active_dims=tuple(cat_dims)
    ).double()

assert (len(cont_dims) + len(cat_dims)) == len(scenario.cs.get_hyperparameters())
noise_prior = HorseshoePrior(0.1)
likelihood = GaussianLikelihood(
    noise_prior=noise_prior,
    noise_constraint=Interval(np.exp(-25), np.exp(2), transform=None)
).double()

if len(cont_dims) > 0 and len(cat_dims) > 0:
    # both
    kernel = ProductKernel(exp_kernel, ham_kernel)
elif len(cont_dims) > 0 and len(cat_dims) == 0:
    # only cont
    kernel = exp_kernel
elif len(cont_dims) == 0 and len(cat_dims) > 0:
    # only cont
    kernel = ham_kernel
else:
    raise ValueError()

kernel = ScaleKernel(kernel,
                     outputscale_constraint=Interval(
                         np.exp(-10.),
                         np.exp(2.),
                         transform=None,
                         initial_value=2.0
                     ),
                     outputscale_prior=LogNormalPrior(0.0, 1.0))

epm_chooser_kwargs = {"model_local": PartialSparseGaussianProcess,
                      "model_local_kwargs": dict(kernel=kernel,
                                                 likelihood=likelihood),
                      "do_switching": False}
smac = SMAC4HPO(scenario=scenario,
               rng=np.random.RandomState(42),
               smbo_kwargs={"epm_chooser":EPMChooserBOinG,
                            "epm_chooser_kwargs":epm_chooser_kwargs},
               tae_runner=rosenbrock_2d,
               runhistory2epm=RunHistory2EPM4LogCostWithRaw
               )

smac.optimize()
