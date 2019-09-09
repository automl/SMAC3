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
from smac.scenario.scenario import Scenario

from smac.intensification.successive_halving import SuccessiveHalving

if __name__ == '__main__':
    logging.basicConfig(level=20)  # 10: debug; 20: info

    scenario = Scenario('scenario.txt')
    intensifier_kwargs = {'min_budget': 1, 'eta': 2}
    smac = SMAC4AC(scenario=scenario,  # scenario object
                   intensifier_kwargs=intensifier_kwargs,  # parameters for Successive Halving
                   intensifier=SuccessiveHalving)  # change intensifier to successive halving by passing the class

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    # inc_value = smac.get_tae_runner().run(incumbent)[1]
    print("Optimized configuration %s" % str(incumbent))
