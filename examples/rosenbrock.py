from __future__ import division, print_function

import logging
import numpy as np

from smac.facade.func_facade import FuncSMAC


def rosenbrock_2d(cfg, seed):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 5].
    """
    x1 = cfg["x1"]
    x2 = cfg["x2"]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val

# debug output
logging.basicConfig(level=10)
logger = logging.getLogger("Optimizer") # Enable to show Debug outputs


smac = FuncSMAC(func=rosenbrock_2d,
                x0=[-3, -4],
                upper_bounds=[5, 5],
                lower_bounds=[-5, -5],
                maxfun=325)

incumbent = smac.optimize()
