#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:31:15 2019

@author: antonio
"""

import logging

import numpy as np

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_bo_facade import SMAC4BO

from smac.optimizer.acquisition import probCLCB

import  matplotlib.pyplot as plt

def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """
    x1 = x["x0"]
    x2 = x["x1"]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    
    check = x1**2 + x2**2

    if(check <= 2):
        return(val)
    else:
        return(np.NAN)
    return val


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
cs.add_hyperparameters([x0, x1])

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 20,   # max. number of function evaluations; for this example set to a low number
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = rosenbrock_2d(cs.get_default_configuration())
print("Default Value: %.2f" % def_value)

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4BO(scenario=scenario,
               rng=np.random.RandomState(42),
               tae_runner=rosenbrock_2d,
               acquisition_function = probCLCB,
               )

smac.optimize()

X, Y, feasible = smac.get_X_y()
feasible = feasible.astype("int")
 
x = np.arange(-5, 5, 0.05).flatten()
y = np.arange(-5, 5, 0.05).flatten()
z = []
for i in np.arange(len(x)):
    for j in np.arange(len(y)):
        z = z + [rosenbrock_2d({"x0": x[i], "x1": y[j]})]
z = np.reshape(np.array(z),newshape=(len(x),len(y)))
h = plt.contourf(x,y,z)
plt.plot(X[:,0],X[:,1], "gx", markersize=5.0)
plt.plot(1.0,1.0, "ro", markersize=5.0)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
  