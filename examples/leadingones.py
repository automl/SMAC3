import logging

import numpy as np

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

def leading_ones(cfg, seed):
    """ Leading ones
    score is the number of 1 starting from the first parameter
    e.g., 111001 -> 3; 0110111 -> 0
    """

    arr_ = [0] * len(cfg.keys())
    for p in cfg:
        arr_[int(p)] = cfg[p]

    count = 0
    for v in arr_:
        if v == 1:
            count += 1
        else:
            break

    return -count

logger = logging.getLogger("Optimizer")  # Enable to show Debug outputs
logging.basicConfig(level=logging.DEBUG)

# build Configuration Space which defines all parameters and their ranges
n_params = 16
use_conditionals = True # using conditionals should help a lot in this example

cs = ConfigurationSpace()
previous_param = None
for n in range(n_params):
    p = CategoricalHyperparameter("%d" % (n), [0, 1], default=0)
    cs.add_hyperparameter(p)

    if n > 0 and use_conditionals:
        cond = InCondition(
            child=p, parent=previous_param, values=[1])
        cs.add_condition(cond)
        
    previous_param = p

# SMAC scenario object
scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                     "runcount-limit": n_params*2,  # at most 200 function evaluations
                     "cs": cs,  # configuration space
                     "deterministic": "true"
                     })

# register function to be optimize
taf = ExecuteTAFuncDict(leading_ones)

# example call of the function
# it returns: Status, Cost, Runtime, Additional Infos
def_value = taf.run(cs.get_default_configuration())[1]
print("Default Value: %.2f" % (def_value))
 
# Optimize
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
            tae_runner=taf)
try:
    incumbent = smac.optimize()
finally:
    incumbent = smac.solver.incumbent

inc_value = taf.run(incumbent)[1]
print("Optimized Value: %.2f" % (inc_value))
