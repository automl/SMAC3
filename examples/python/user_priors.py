"""
Synthetic Function with few Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize a synthetic function (rosenbrock function)
using a prior over the optimum and the PiBO framework, implemented in 
PriorAcquisitionFunction.

We use the SMAC4BB facade because it is designed for black-box function optimization.
SMAC4BB uses a :term:`Gaussian Process<GP>` or a set of Gaussian Processes whose
hyperparameters are integrated by Markov-Chain Monte-Carlo as its surrogate model.
SMAC4BB works best on numerical hyperparameter configuration space and should not
be applied to the problems with large evaluation budgets (up to 1000 evaluations).
"""

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.hyperparameters import NormalFloatHyperparameter, \
    BetaFloatHyperparameter
from smac.initial_design.random_configuration_design import RandomConfigurations
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB

from smac.optimizer.acquisition import EI

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
import matplotlib.pyplot as plt

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


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

    #print('X1:   ', x1, 'X2:   ', x2, 'Value:   ', val)
    return val


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges
    # Uniform and Categorical Hyperparameters (with weights) are also accepted when doing
    # optimization with priors
    cs = ConfigurationSpace()
    # A log-normal prior on the first hyperparameter
    x0 = NormalFloatHyperparameter("lognormal_param", lower=0.1, upper=10, default_value=2, mu=1, sigma=0.5, log=True)
    # And a beta prior on the other hyperparameter
    x1 = BetaFloatHyperparameter("beta_param", lower=-5, upper=10, default_value=2, alpha=3, beta=8, log=False)

    cs.add_hyperparameters([x0, x1])
    
    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 100,  # max. number of function evaluations
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         })
    
    # can be any surrogate for user priors
    model_type = 'gp'
    
    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = rosenbrock_2d(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC4BB(scenario=scenario,
                   rng=np.random.RandomState(42),
                   acquisition_function=EI,  # or others like PI, LCB as acquisition functions can also be used with user priors
                   user_priors=True, # this is the flag to optimize with user priors
                   user_prior_kwargs={"decay_beta": 3}, # the confidence in the prior used. Defaults to #iterations / 10. Lower beta makes prior decay quicker.
                   tae_runner=rosenbrock_2d,
                   initial_design=RandomConfigurations, # will initiate with configurations sampled from the prior 
                   )
    
    smac.optimize()
