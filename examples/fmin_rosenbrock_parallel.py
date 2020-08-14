"""
============================================
Parallel Intensifier with No Intensification
============================================

This example showcases how to use dask to
launch parallel configurations via n_workers
"""

import logging

from smac.intensification.random_search import RandomSearcher
from smac.facade.func_facade import fmin_smac

# --------------------------------------------------------------
# We need to provide a pickable function and use __main__
# to be compliant with multiprocessing API
# Below is a WA to have a packaged function called rosenbrock_2d
# --------------------------------------------------------------
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from rosenbrock_2d_delayed import rosenbrock_2d  # noqa: E402
# --------------------------------------------------------------

if __name__ == '__main__':

    # debug output
    logging.basicConfig(level=20)
    logger = logging.getLogger("Optimizer")  # Enable to show Debug outputs

    # fmin_smac assumes that the function is deterministic
    # and uses under the hood the SMAC4HPO
    # n_workers tells the SMBO loop to execute in parallel
    x, cost, smac = fmin_smac(func=rosenbrock_2d,
                              intensifier=RandomSearcher,
                              scenario_args={'n_workers': 4},
                              x0=[-3, -4],
                              bounds=[(-5, 10), (-5, 10)],
                              maxfun=20,
                              rng=3)  # Passing a seed makes fmin_smac determistic
    print("Best x: %s; with cost: %f" % (str(x), cost))
