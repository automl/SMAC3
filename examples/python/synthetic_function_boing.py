"""
Synthetic Function with BOinG as optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An example of applying SMAC with BO inside Grove(BOinG) to optimize a
synthetic function (2d rosenbrock function).

BOinG optimizer requires EPMChooserBOinG to suggest next configuration to be evaluated
"""

import logging

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from smac.epm.globally_augmented_local_gp import GloballyAugmentedLocalGP

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.optimizer.local_bo.rh2epm_boing import RunHistory2EPM4ScaledLogCostWithRaw
from smac.optimizer.local_bo.epm_chooser_boing import EPMChooserBOinG

# Import SMAC-utilities
from smac.scenario.scenario import Scenario


from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors import LogNormalPrior, HorseshoePrior

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])

    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 100,
                         # max. number of function evaluations; for this example set to a low number
                         "cs": cs,  # configuration space
                         "deterministic": "true"
                         })

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = rosenbrock_2d(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    cont_dims = np.arange(len(cs.get_hyperparameters()))
    exp_kernel = MaternKernel(2.5,
                              lengthscale_constraint=Interval(
                                  torch.tensor(np.exp(-6.754111155189306).repeat(cont_dims.shape[-1])),
                                  torch.tensor(np.exp(0.0858637988771976).repeat(cont_dims.shape[-1])),
                                  transform=None,
                                  initial_value=1.0
                              ),
                              ard_num_dims=cont_dims.shape[-1],
                              active_dims=tuple(cont_dims)).double()

    # by setting lower bound of noise_constraint we could make it more stable
    noise_prior = HorseshoePrior(0.1)
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=Interval(1e-5, np.exp(2), transform=None)
    ).double()

    kernel = ScaleKernel(exp_kernel,
                         outputscale_constraint=Interval(
                             np.exp(-10.),
                             np.exp(2.),
                             transform=None,
                             initial_value=2.0
                         ),
                         outputscale_prior=LogNormalPrior(0.0, 1.0))

    epm_chooser_kwargs = {"model_local": GloballyAugmentedLocalGP,
                          "model_local_kwargs": dict(kernel=kernel,
                                                     likelihood=likelihood),
                          'min_configs_local': 10,  # comment this out for BOinG with default setting
                          "do_switching": True}
    # same as SMAC4BB
    random_configuration_chooser_kwargs = {'prob': 0.08447232371720552}

    smac = SMAC4HPO(scenario=scenario,
                    rng=np.random.RandomState(42),
                    smbo_kwargs={"epm_chooser": EPMChooserBOinG,
                                 "epm_chooser_kwargs": epm_chooser_kwargs},
                    tae_runner=rosenbrock_2d,
                    random_configuration_chooser_kwargs=random_configuration_chooser_kwargs,
                    runhistory2epm=RunHistory2EPM4ScaledLogCostWithRaw
                    )

    smac.optimize()
