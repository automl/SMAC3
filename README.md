# SMAC v3 Project

Copyright (C) 2016-2020  [AutoML Group](http://www.automl.org/)

__Attention__: This package is a reimplementation of the original SMAC tool
(see reference below).
However, the reimplementation slightly differs from the original SMAC.
For comparisons against the original SMAC, we refer to a stable release of SMAC (v2) in Java
which can be found [here](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/).

The documentation can be found [here](https://automl.github.io/SMAC3/).

Status for master branch:

[![Build Status](https://travis-ci.org/automl/SMAC3.svg?branch=master)](https://travis-ci.org/automl/SMAC3)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/58f47a4bd25e45c9a4901ebca68118ff?branch=master)](https://www.codacy.com/app/automl/SMAC3?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=automl/SMAC3&amp;utm_campaign=Badge_Grade)
[![codecov Status](https://codecov.io/gh/automl/SMAC3/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/SMAC3)

Status for the development branch

[![Build Status](https://travis-ci.org/automl/SMAC3.svg?branch=development)](https://travis-ci.org/automl/SMAC3)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/58f47a4bd25e45c9a4901ebca68118ff?branch=development)](https://www.codacy.com/app/automl/SMAC3?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=automl/SMAC3&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/automl/SMAC3/branch/development/graph/badge.svg)](https://codecov.io/gh/automl/SMAC3)

Try SMAC directly in your Browser

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v0ZH5S9Sfift30GxHAp96e0yZZUFS0Ah)

# OVERVIEW

SMAC is a tool for algorithm configuration to optimize the parameters of
arbitrary algorithms across a set of instances. This also includes
hyperparameter optimization of ML algorithms. The main core consists of
Bayesian Optimization in combination with an aggressive racing mechanism to
efficiently decide which of two configurations performs better.

For a detailed description of its main idea,
we refer to

    Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
    Sequential Model-Based Optimization for General Algorithm Configuration
    In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)


SMAC v3 is written in Python3 and continuously tested with Python 3.6 and
python3.6. Its [Random Forest](https://github.com/automl/random_forest_run)
is written in C++.

# Installation

## Requirements

Besides the listed requirements (see `requirements.txt`), the random forest
used in SMAC3 requires SWIG (>= 3.0, <4.0) as a build dependency:

```apt-get install swig```

On Arch Linux (or any distribution with swig4 as default implementation):

```
pacman -Syu swig3
ln -s /usr/bin/swig-3 /usr/bin/swig
```

## Installation via pip

SMAC3 is available on PyPI.

```pip install smac```

## Manual Installation

```
git clone https://github.com/automl/SMAC3.git && cd SMAC3
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install .
```

## Installation in Anaconda

If you use Anaconda as your Python environment, you have to install three
packages **before** you can install SMAC:

```conda install gxx_linux-64 gcc_linux-64 swig```

## Optional dependencies

SMAC3 comes with a set of optional dependencies that can be installed using
[setuptools extras](https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies):

- `lhd`: Latin hypercube design
- `gp`: Gaussian process models

These can be installed from PyPI or manually:

```
# from PyPI
pip install smac[gp]

# manually
pip install .[gp,lhd]
```

For convenience, there is also an `all` meta-dependency that installs all optional dependencies:
```
pip install smac[all]
```

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
It supports the same parameter configuration space syntax
(except for extended forbidden constraints) and interface to
target algorithms.

# Examples

We provide a bunch of examples in the [examples folder](examples), such as:

  * Optimization of a python function directly with SMAC
    * [branin/branin_fmin.py](examples/branin/branin_fmin.py)
    * [fmin_rosenbrock.py](examples/fmin_rosenbrock.py) - Optimization of the 2D Rosenbrock function
    * [fmin_rosenbrock_parallel.py](examples/fmin_rosenbrock_parallel.py) - Example of parallel SMAC using dask
  * Optimization of a black-box function with SMAC
    * [SMAC4BO_rosenbrock.py](examples/SMAC4BO_rosenbrock.py) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v0ZH5S9Sfift30GxHAp96e0yZZUFS0Ah)
    * [SMAC4BO_acq_rosenbrock.py](examples/SMAC4BO_acq_rosenbrock.py) - Example to select the acquisition function
  * Hyperparameter Optimization with SMAC
    * [SMAC4HPO_rosenbrock.py](examples/SMAC4BO_rosenbrock.py) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v0ZH5S9Sfift30GxHAp96e0yZZUFS0Ah)
    * [SMAC4HPO_rf.py](examples/SMAC4BO_rf.py) - Optimization of a random forest
    * [SMAC4HPO_svm.py](examples/SMAC4HPO_svm.py) - Optimization of an SVM [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v0ZH5S9Sfift30GxHAp96e0yZZUFS0Ah)
  * Optimization of a SAT solver across problem instances with SMAC
    * [spear_qcp/run_SMAC.sh](examples/spear_qcp/run_SMAC.sh)
  * Optimization of an MLP
    * [parallel_sh_mlp.py](examples/parallel_sh_mlp.py) - Parallel Successive Halving
    * [hyperband_mlp.py](examples/hyperband_mlp.py) - Hyperband
    * [BOHB4HPO_mlp.py](examples/BOHB4HPO_mlp.py) - BOHB
    * [BOHB4HPO_sgd_instances.py](examples/BOHB4HPO_sgd_instances.py) - BOHB across instances

An overview of all examples can be seen in our [documentation](https://automl.github.io/SMAC3/master/examples/index.html).

# Contact

SMAC3 is developed by the [AutoML Group of the University of Freiburg](http://www.automl.org/).

If you found a bug, please report to <https://github.com/automl/SMAC3/issues>.

Our guidelines for contributing to this package can be found [here](https://github.com/automl/SMAC3/blob/master/.github/CONTRIBUTING.md)
