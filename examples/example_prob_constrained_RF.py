#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:19:05 2019

@author: antonio
"""


import logging

from smac.facade.func_facade import fmin_smac
from smac.optimizer.acquisition import probCLCB
import  matplotlib.pyplot as plt
import numpy as np


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
    check = x[0]**2 + x[1]**2

    if(check <= 2):
        return(val)
    else:
        return(np.NAN)

if __name__ == '__main__':
    x0 = np.random.uniform(low=-5,high=10,size=(1,2)).tolist()[0]
#    x0 = np.random.uniform(low=-1.5,high=1.5,size=(1,2)).tolist()[0]
    logging.basicConfig(level=20)  # 10: debug; 20: info
    x, cost, opt = fmin_smac(func=rosenbrock_2d,  # function
                           x0=x0,#[-1.5, 5],    # default configuration
                           bounds=[(-5,10),(-5,10)], #[(-5, 10), (0, 15)],  # limits
#                           bounds=[(-1.5,1.5),(-1.5,1.5)], #[(-5, 10), (0, 15)],  # limits
#                           maxfun= 20,
                           maxfun=200,   # maximum number of evaluations
                           rng=5, acquisition_function = probCLCB)# random seed
    print("Optimum at {} with cost of {}".format(x, cost))
    X, Y, feasible = opt.get_X_y()
    feasible = feasible.astype("int")
 
    x = np.arange(-5, 5, 0.05).flatten()
    y = np.arange(-5, 5, 0.05).flatten()
    z = []
    for i in np.arange(len(x)):
        for j in np.arange(len(y)):
            z = z + [rosenbrock_2d([x[i],y[j]])]
    z = np.reshape(np.array(z),newshape=(len(x),len(y)))
    h = plt.contourf(x,y,z)
    plt.plot(X[:,0],X[:,1], "gx", markersize=5.0)
    plt.plot(1.0,1.0, "ro", markersize=5.0)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
