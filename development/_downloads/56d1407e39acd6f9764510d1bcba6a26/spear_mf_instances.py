"""
SPEAR-QCP with Multi-Fidelity on Instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We optimize the SPEAR algorithm on QCP to demonstrate the powerful SMAC4AC facade. Algorithm and
instance definition is done inside scenario file.

Moreover, we present you an alternative :term:`intensification<Intensification>` procedure "Successive Halving".
"""

import logging

logging.basicConfig(level=logging.INFO)

from smac.facade.smac_ac_facade import SMAC4AC
from smac.intensification.successive_halving import SuccessiveHalving
from smac.scenario.scenario import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


if __name__ == "__main__":
    scenario = Scenario("examples/commandline/spear_qcp/scenario.txt")

    # provide arguments for the intensifier like this
    intensifier_kwargs = {
        "n_seeds": 2,  # specify the number of seeds to evaluate for a non-deterministic target algorithm
        "initial_budget": 1,
        "eta": 3,
        "min_chall": 1,  # because successive halving cannot handle min_chall > 1
    }

    smac = SMAC4AC(
        scenario=scenario,  # scenario object
        intensifier_kwargs=intensifier_kwargs,  # arguments for Successive Halving
        # change intensifier to successive halving by passing the class.
        # it must implement `AbstractRacer`.
        intensifier=SuccessiveHalving,
    )

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    print("Optimized configuration %s" % str(incumbent))
