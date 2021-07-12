"""
==================================================
Using the black-box optimization interface of SMAC
==================================================
"""

import logging

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bo_facade import SMAC4BO
from smac.optimizer.epm_chooser_turbo import EPMChooserTurBO

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.optimizer.ei_optimization import SobolSearch


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
smac = SMAC4BO(scenario=scenario,
               rng=np.random.RandomState(42),
               model_type="gp",
               smbo_kwargs={"epm_chooser":EPMChooserTurBO},
               initial_design_kwargs={"init_budget": 1},
               tae_runner=rosenbrock_2d,
               acquisition_function_optimizer=SobolSearch,
               )

smac.optimize()
