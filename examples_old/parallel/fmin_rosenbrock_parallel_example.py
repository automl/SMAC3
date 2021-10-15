"""
============================================
Parallel Intensifier with No Intensification
============================================

This example showcases how to use dask to
launch parallel configurations via n_workers
"""

import logging
logging.basicConfig(level=logging.INFO)

from smac.intensification.simple_intensifier import SimpleIntensifier
from smac.facade.func_facade import fmin_smac

# --------------------------------------------------------------
# We need to provide a pickable function and use __main__
# to be compliant with multiprocessing API
# Below is a work around to have a packaged function called
# rosenbrock_2d
# --------------------------------------------------------------
import os
import sys

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


# https://sphinx-gallery.github.io/stable/faq.html#why-is-file-not-defined-what-can-i-use
cwd = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
sys.path.append(os.path.join(cwd))
from rosenbrock_2d_delayed_func import rosenbrock_2d  # noqa: E402
# --------------------------------------------------------------


if __name__ == '__main__':
    # fmin_smac assumes that the function is deterministic
    # and uses under the hood the SMAC4HPO
    # n_workers tells the SMBO loop to execute in parallel
    x, cost, smac = fmin_smac(
        func=rosenbrock_2d,
        intensifier=SimpleIntensifier,
        x0=[-3, -4],
        bounds=[(-5, 10), (-5, 10)],
        maxfun=25,
        rng=3,
        n_jobs=4,
    )  # Passing a seed makes fmin_smac determistic

    print("Best x: %s; with cost: %f" % (str(x), cost))
