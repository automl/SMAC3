# SMAC v3 Project

Copyright (C) 2016  [ML4AAD Group](http://www.ml4aad.org/)

__Attention__: This package is under heavy development and subject to change. 
A stable release of SMAC (v2) in Java can be found [here](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/).

The documentation can be found [here](https://automl.github.io/SMAC3/).

Status for master branch:

[![Build Status](https://travis-ci.org/automl/SMAC3.svg?branch=master)](https://travis-ci.org/automl/SMAC3)
[![Code Health](https://landscape.io/github/automl/SMAC3/master/landscape.svg?style=flat)](https://landscape.io/github/automl/SMAC3/master)
[![Coverage Status](https://coveralls.io/repos/automl/auto-sklearn/badge.svg?branch=master&service=github)](https://coveralls.io/github/automl/SMAC3?branch=master)

Status for development branch

[![Build Status](https://travis-ci.org/automl/SMAC3.svg?branch=development)](https://travis-ci.org/automl/SMAC3)
[![Code Health](https://landscape.io/github/automl/SMAC3/development/landscape.svg?style=flat)](https://landscape.io/github/automl/SMAC3/development)
[![Coverage Status](https://coveralls.io/repos/automl/SMAC3/badge.svg?branch=development&service=github)](https://coveralls.io/github/automl/SMAC3?branch=development)

#OVERVIEW

SMAC is a tool for algorithm configuration 
to optimize the parameters of arbitrary algorithms across a set of instances.
This also includes hyperparameter optimization of ML algorithms.
The main core consists of Bayesian Optimization in combination with a simple racing mechanism to
efficiently decide which of two configuration performs better.

For a detailed description of its main idea,
we refer to

    Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
    Sequential Model-Based Optimization for General Algorithm Configuration
    In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)


SMAC v3 is written in python3 and continuously tested with python3.4 and python3.5. 
Its [Random Forest](https://bitbucket.org/aadfreiburg/random_forest_run) is written in C++.

#Installation:

    cat requirements.txt | xargs -n 1 -L 1 pip install
    
    python setup.py install
    
# License

This program is free software: you can redistribute it and/or modify
it under the terms of the 3-clause BSD license (please see the LICENSE file).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the 3-clause BSD license 
along with this program (see LICENSE file). 
If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# USAGE

The usage of SMAC v3 is mainly the same as provided with [SMAC v2.08](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/v2.08.00/manual.pdf).
It supports the same parameter configuration space syntax and interface to target algorithms.
Please note that we do not support the extended parameter configuration syntax introduced in SMACv2.10.

# Examples

See examples/

  * examples/rosenbrock.py - example on how to optimize a Python function (REQUIRES [PYNISHER](https://github.com/sfalkner/pynisher) )
  * examples/spear_qcp/run.sh - example on how to optimize the SAT solver Spear on a set of SAT formulas
 
# Contact
 
SMAC v3 is developed by the [ML4AAD Group of the University of Freiburg](http://www.ml4aad.org/).

If you found a bug, please report to https://github.com/automl/SMAC3
