from __future__ import division, print_function

#TODO: remove really ugly boilerplate
import logging
import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0,cmd_folder)
    
import numpy as np

from smac.facade.func_facade import FuncSMAC

def rosenbrock_4d(cfg, seed):
    """ The 4 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous, but we will pretent that 
    x2, x3, and x4 can only take integral values. The search domain for
    all x's is the interval [-5, 5].
    """
    x1 = cfg["x1"]
    x2 = cfg["x2"]
    x3 = cfg["x3"]
    x4 = cfg["x4"]

    val = (100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2 +
           100 * (x3 - x2 ** 2) ** 2 + (x2 - 1) ** 2 +
           100 * (x4 - x3 ** 2) ** 2 + (x3 - 1) ** 2)
    
    return(val)

# debug output
logging.basicConfig(level=10)
logger = logging.getLogger("Optimizer") # Enable to show Debug outputs


smac = FuncSMAC(func=rosenbrock_4d,
                 x0 = [-1,0,1,0],
                 upper_bounds= [1,1,1,1],
                 lower_bounds= [-1,-1,-1,-1],
                 maxfun=100)

incumbent = smac.optimize()