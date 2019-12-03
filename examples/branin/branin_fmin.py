import logging

from branin import branin

from smac.facade.func_facade import fmin_smac

"""
Example for the use of fmin_smac, a basic SMAC-wrapper to optimize a
function.
We optimize the branin-function, which has two parameters: x1 and x2.
The fmin_smac needs neither a scenario-file, nor a configuration space.
All relevant information is directly passed to the function.
"""

if __name__ == '__main__':
    logging.basicConfig(level=20)  # 10: debug; 20: info
    x, cost, _ = fmin_smac(func=branin,  # function
                           x0=[0, 0],  # default configuration
                           bounds=[(-5, 10), (0, 15)],  # limits
                           maxfun=10,  # maximum number of evaluations
                           rng=3)  # random seed
    print("Optimum at {} with cost of {}".format(x, cost))
