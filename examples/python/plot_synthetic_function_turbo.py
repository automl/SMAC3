"""
Synthetic Function with TuRBO as optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of applying SMAC with trust region BO (TuRBO) to optimize a
synthetic function (2d rosenbrock function).

Eriksson et al. Scalable Global Optimization via Local {Bayesian} Optimization,
http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf

TurBO gradually shrinks its search space to the vicinity of the optimum configuration that is ever optimized.
TuRBO optimizer requires EPMChooserTurBO to suggest the next configuration. Currently, it only supports pure numerical
hyperparameters.
"""

import logging

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.configuration_chooser.turbo_chooser import TurBOChooser

# Import SMAC-utilities
from smac.scenario.scenario import Scenario


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
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])

    # Scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternatively runtime)
            "runcount-limit": 100,
            "cs": cs,  # configuration space
            "deterministic": "true",
        }
    )

    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = rosenbrock_2d(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC4BB(
        scenario=scenario,
        rng=np.random.RandomState(42),
        model_type="gp",
        smbo_kwargs={"epm_chooser": TurBOChooser},
        initial_design_kwargs={"init_budget": 0},
        tae_runner=rosenbrock_2d,
    )

    smac.optimize()
