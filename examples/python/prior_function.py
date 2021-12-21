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
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, NormalFloatHyperparameter, \
    BetaFloatHyperparameter, NormalIntegerHyperparameter, BetaIntegerHyperparameter, CategoricalHyperparameter
from smac.initial_design.random_configuration_design import RandomConfigurations
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.facade.smac_hpo_facade import SMAC4HPO

from smac.optimizer.acquisition import EI

# Import SMAC-utilities
from smac.scenario.scenario import Scenario

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
    x2 = 1 #x["x1"]
    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.

    #print('X1:   ', x1, 'X2:   ', x2, 'Value:   ', val)
    return val


if __name__ == "__main__":
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    x0 = NormalFloatHyperparameter("x0", lower=1, upper=100, default_value=10, mu=3, sigma=1, log=True)
    #x0 = BetaFloatHyperparameter("x0", lower=1, upper=100, default_value=5, log=True, alpha=5, beta=1.1)
    #x1 = BetaFloatHyperparameter("x1", lower=1, upper=2, default_value=2, alpha=2, beta=1, log=False)#, mu=2, sigma=1, log=True)
    #x2 = CategoricalHyperparameter("cat", choices=['a', 'b', 'c', 'd', 'e'], default_value='c', weights=[1, 2, 3, 4, 5])
    #x2._sample(np.random.RandomState(1), size=10)
    #x1 = BetaFloatHyperparameter("x1", lower=-5, upper=10, default_value=0, alpha=15, beta=10)
    import matplotlib.pyplot as plt
    #x, y = x1.get_probs()
        #x1 = BetaFloatHyperparameter("x1", lower=-5, upper=10, default_value=0, alpha=3, beta=1)
    #x0 = UniformFloatHyperparameter("x0", lower=0, upper=5, default_value=5)
    #x1 = UniformFloatHyperparameter("x1", lower=1, upper=np.exp(5), default_value=5, log=True)
    cs.add_hyperparameters([x0])
    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 100,  # max. number of function evaluations
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         })
    
        # Use 'gp' or 'gp_mcmc' here
    model_type = 'gp'
    
    # Example call of the function
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = rosenbrock_2d(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)

    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = SMAC4HPO(scenario=scenario,
                   rng=np.random.RandomState(42),
                   acquisition_function=EI,  # or others like PI, LCB as acquisition functions
                   tae_runner=rosenbrock_2d,
                   optimize_with_priors=True,
                   initial_design=RandomConfigurations,
                   )
    acq = smac.solver.epm_chooser.acquisition_func
    smac.optimize()
