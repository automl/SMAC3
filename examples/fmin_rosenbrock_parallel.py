"""
==========================
scipy-style fmin interface
==========================
"""

import logging
import time

from smac.facade.func_facade import fmin_smac

if __name__ == '__main__':

    # Encapsulate the function within the main to have a self
    # Isolated example. Functions to be parallelized must be pickle-compatible
    def rosenbrock_2d(x):
        """ The 2 dimensional Rosenbrock function as a toy model
        The Rosenbrock function is well know in the optimization community and
        often serves as a toy problem. It can be defined for arbitrary
        dimensions. The minimium is always at x_i = 1 with a function value of
        zero. All input parameters are continuous. The search domain for
        all x's is the interval [-5, 10].
        """
        x1 = x[0]
        x2 = x[1]

        val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
        time.sleep(1)
        return val

    # debug output
    logging.basicConfig(level=20)
    logger = logging.getLogger("Optimizer")  # Enable to show Debug outputs

    # fmin_smac assumes that the function is deterministic
    # and uses under the hood the SMAC4HPO
    x, cost, _ = fmin_smac(func=rosenbrock_2d,
                           scenario_args={'n_workers': 4},
                           x0=[-3, -4],
                           bounds=[(-5, 10), (-5, 10)],
                           maxfun=10,
                           rng=3)  # Passing a seed makes fmin_smac determistic

    print("Best x: %s; with cost: %f" % (str(x), cost))
