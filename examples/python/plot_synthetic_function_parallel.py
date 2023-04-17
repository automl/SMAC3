"""
Synthetic Function with few Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize a synthetic function (2d rosenbrock function).

We use the pSMAC [1]_ facade to demonstrate the parallelization of SMAC.
Other than that, we use a :term:`Gaussian Process<GP>` to optimize our black-box
function.


.. [1] Ramage, S. E. A. (2015). Advances in meta-algorithmic software libraries for
    distributed automated algorithm configuration (T). University of British
    Columbia. Retrieved from
    https://open.library.ubc.ca/collections/ubctheses/24/items/1.0167184.
"""
import importlib
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

import smac

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.psmac_facade import PSMAC
from smac.facade.smac_bb_facade import SMAC4BB

importlib.reload(smac.facade.psmac_facade)
from smac.facade.psmac_facade import PSMAC
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
            "runcount-limit": 20,  # max. number of function evaluations PER WORKER
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
    smac = PSMAC(
        scenario=scenario,
        facade_class=SMAC4BB,
        model_type=model_type,
        rng=np.random.RandomState(42),
        acquisition_function=EI,  # or others like PI, LCB as acquisition functions
        tae_runner=rosenbrock_2d,
        n_workers=2,  # 2 parallel workers
    )

    incumbent = smac.optimize()
    # Get trajectory of optimization (incumbent over time)
    trajectory_json = smac.get_trajectory()  # trajectory in json format

    # Plot trajectory: cost of incumbent against number of evaluations
    # import matplotlib.pyplot as plt
    # X = [t["evaluations"] for t in trajectory_json]
    # Y = [t["cost"] for t in trajectory_json]
    # plt.plot(X, Y)
    # plt.yscale("log")
    # plt.xlabel("Number of Evaluations")
    # plt.ylabel("Cost of Incumbent")
    # plt.show()
