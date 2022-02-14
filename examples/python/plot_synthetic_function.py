"""
Synthetic Function with few Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize a synthetic function (2d rosenbrock function).

We use the SMAC4BB facade because it is designed for black-box function optimization.
SMAC4BB uses a :term:`Gaussian Process<GP>` or a set of Gaussian Processes whose
hyperparameters are integrated by Markov-Chain Monte-Carlo as its surrogate model.
SMAC4BB works best on numerical hyperparameter configuration space and should not
be applied to the problems with large evaluation budgets (up to 1000 evaluations).
"""

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI

# Import SMAC-utilities
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def rosenbrock_2d(x):
    """The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """

    x1 = x["x0"]
    x2 = x["x1"]

    val = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
    return val


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])

    # Scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 10,  # max. number of function evaluations
            "cs": cs,  # configuration space
            "deterministic": True,
        }
    )

    # Use 'gp' or 'gp_mcmc' here
    model_type = "gp"

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = rosenbrock_2d(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC4BB(
        scenario=scenario,
        model_type=model_type,
        rng=np.random.RandomState(42),
        acquisition_function=EI,  # or others like PI, LCB as acquisition functions
        tae_runner=rosenbrock_2d,
    )

    smac.optimize()
