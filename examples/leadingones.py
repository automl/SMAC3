"""
An example for the usage of SMAC within Python. The function 'leading_ones' is
optimized. Conditionals are explained.
"""

import logging
import numpy as np

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

def leading_ones(cfg):
    """ Leading Ones
    score is the number of 1 starting from the first parameter
    e.g., 111001 -> 3; 0110111 -> 0

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the individual numbers.
        (e.g. param1: 1, param2: 1, param3: 0, ...)
        Configurations are indexable.
    """
    arr_ = [None] * len(cfg.keys())
    logger.info(arr_)
    for p in cfg:
        logger.error(p)
        logger.error(cfg[p])
        arr_[int(p[5:])] = cfg[p]

    logger.info(arr_)

    count = 0
    for v in arr_:
        if v and int(v) == 1:
            count += 1
        else:
            break

    logger.error(count)
    return -count

logger = logging.getLogger("LeadingOnesExample")  # Enable to show Debug outputs
logging.basicConfig(level=logging.DEBUG)

# build Configuration Space which defines all parameters and their ranges
n_params = 16
# using conditionals should help a lot in this example
# TODO major bug in exhausted ConfigSpace (https://github.com/automl/SMAC3/issues/25)
use_conditionals = True

cs = ConfigurationSpace()
previous_param = None
for n in range(n_params):
    # Build general parameters {param1:"0", param2:"1", etc.}
    p = CategoricalHyperparameter("param%d" % (n), ["0", "1"], default="0")
    cs.add_hyperparameter(p)

    # Limit the search-space using conditionals
    if n > 0 and use_conditionals:
        cond = InCondition(
            child=p, parent=previous_param, values=["1"])
        cs.add_condition(cond)

    previous_param = p

# scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                     "runcount-limit": n_params*10,  # at most 32 function evaluations
                     "cs": cs,  # configuration space
                     "deterministic": "true"
                     })

# register function to be optimize using a Target Algorithm Function Evaluator
taf = ExecuteTAFuncDict(leading_ones)

# example call of the function
# it returns: Status, Cost, Runtime, Additional Infos
def_value = taf.run(cs.get_default_configuration())[1]
print("Default Value: %.2f" % (def_value))

# Optimize
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=taf)

try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = taf.run(incumbent)[1]
print("Optimized Value: %.2f" % (inc_value))
