"""
==========================
scipy-style fmin interface
==========================
"""

import logging
logging.basicConfig(level=20)

from smac.facade.func_facade import fmin_smac

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
    x1 = x[0]
    x2 = x[1]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val


if __name__ == "__main__":
    # fmin_smac assumes that the function is deterministic
    # and uses under the hood the SMAC4HPO
    x, cost, _ = fmin_smac(func=rosenbrock_2d,
                           x0=[-3, -4],
                           bounds=[(-5, 10), (-5, 10)],
                           maxfun=10,
                           rng=3)  # Passing a seed makes fmin_smac determistic

    print("Best x: %s; with cost: %f" % (str(x), cost))
