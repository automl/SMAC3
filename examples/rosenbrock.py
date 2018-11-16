from __future__ import division, print_function

import logging

from smac.facade.func_facade import fmin_smac


def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 5].
    """
    x1 = x[0]
    x2 = x[1]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val

# debug output
logging.basicConfig(level=20)
logger = logging.getLogger("Optimizer") # Enable to show Debug outputs

x, cost, _ = fmin_smac(func=rosenbrock_2d,
                       x0=[-3, -4],
                       bounds=[(-5, 5), (-5, 5)],
                       maxfun=50,
                       rng=3)  # Passing a seed makes fmin_smac determistic

print("Best x: %s; with cost: %f"% (str(x), cost))
