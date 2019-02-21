import logging

import numpy as np

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.optimizer.acquisition import LCB
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.runhistory.runhistory2epm import RunHistory2EPM4InvScaledCost

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
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 200,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = rosenbrock_2d(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=rosenbrock_2d,
        initial_design=LHDesign,
        initial_design_kwargs={'n_configs_x_params':4,
                               'max_config_fracs':1.0},
        runhistory2epm=RunHistory2EPM4InvScaledCost,
        model_kwargs={'num_trees': 42},
        acquisition_function_optimizer_kwargs={'max_steps':100},
        acquisition_function=LCB,
        acquisition_function_kwargs={'par':0.01}
        )

smac.optimize()
