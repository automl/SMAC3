"""
Example for the use of SMAC4AC, a basic SMAC-wrapper to run the algorithm configuration with a scenario file.
We optimize the spear-qcp algorithm.
It requires the scenario object for the initialization, which can be provided either through a `scenario.txt` file,
or by creating a scenario object in the code.
This example also allows you to use "Successive Halving" as an alternate intensification procedure to
SMAC's own "intensifier" approach.
"""

import logging

from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.successive_halving import SuccessiveHalving
from smac.scenario.scenario import Scenario

if __name__ == '__main__':
    logging.basicConfig(level=20)  # 10: debug; 20: info

    scenario = Scenario('scenario.txt')

    # provide arguments for the intensifier like this
    intensifier_kwargs = {'n_seeds': 2,  # specify the number of seeds to evaluate for a
                          # non-deterministic target algorithm
                          'initial_budget': 1,
                          'eta': 3,
                          'min_chall': 1  # because successive halving cannot handle min_chall > 1
                          }

    smac = SMAC4AC(scenario=scenario,  # scenario object
                   intensifier_kwargs=intensifier_kwargs,  # arguments for Successive Halving
                   intensifier=SuccessiveHalving  # change intensifier to successive halving by passing the class.
                   # it must implement `AbstractRacer`
                   )

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print("Optimized configuration %s" % str(incumbent))
