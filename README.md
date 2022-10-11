# Sequential Model Algorithm Configuration (SMAC)


[![Tests](https://github.com/automl/SMAC3/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/automl/SMAC3/actions/workflows/pytest.yml)
[![Docs](https://github.com/automl/SMAC3/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/automl/SMAC3/actions/workflows/docs.yml)
[![Examples](https://github.com/automl/SMAC3/actions/workflows/examples.yml/badge.svg?branch=main)](https://github.com/automl/SMAC3/actions/workflows/examples.yml)
[![codecov
Status](https://codecov.io/gh/automl/SMAC3/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/SMAC3)

SMAC is a tool for algorithm configuration to optimize the parameters of arbitrary algorithms, including hyperparameter 
optimization of Machine Learning algorithms. The main core consists of Bayesian Optimization in combination with an 
aggressive racing mechanism to efficiently decide which of two configurations performs better.

SMAC3 is written in Python3 and continuously tested with Python 3.8, 3.9, and 3.10. Its Random
Forest is written in C++. In further texts, SMAC is representatively mentioned for SMAC3.

> [Documention](https://automl.github.io/SMAC3)

> [Roadmap](https://github.com/orgs/automl/projects/5/views/2)


## Important: Changes in v2.0

With the next big major release of SMAC, we drastically boosted the user experience by improving the APIs and how the 
pipelining is done (see [changelog](CHANGELOG.md)). However, cleaning the code base, we removed the command-line 
interface (calling a target function from a script is still supported), runtime optimization, and pSMAC. Also,
 python 3.7 is not supported anymore. If you depend on those functionalities, we kindly ask you to keep using v1.4.

We are excited to introduce the new major release and look forward to developing new features on the new code base. 
We hope you enjoy this new user experience as much as we do. 🚀


## Installation

This is for the installation on a Linux system, for Windows and Mac and further information see the [documention](https://automl.github.io/SMAC3/main/installation.html).
Create a new environment with python 3.10 and make sure swig is installed either on your system or
inside the environment. We demonstrate the installation via anaconda in the following:

Create and activate environment:
```
conda create -n SMAC python=3.10
conda activate SMAC
```

Install swig:
```
conda install gxx_linux-64 gcc_linux-64 swig
```

Install SMAC via PyPI:
```
pip install smac
```

Or alternatively, clone the environment:
```
git clone https://github.com/automl/SMAC3.git && cd SMAC3
pip install -e .[dev]
```


## Minimal Example

```py
from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)


configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=200)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train)
incumbent = smac.optimize()
```

More examples can be found in the [documention](https://automl.github.io/SMAC3/main/examples/).


## License

This program is free software: you can redistribute it and/or modify
it under the terms of the 3-clause BSD license (please see the LICENSE file).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the 3-clause BSD license
along with this program (see LICENSE file).
If not, see [here](https://opensource.org/licenses/BSD-3-Clause).


## Miscellaneous

SMAC3 is developed by the [AutoML Groups of the Universities of Hannover and
Freiburg](http://www.automl.org/).

If you have found a bug, please report to [issues](https://github.com/automl/SMAC3/issues). Moreover, we are 
appreciating any kind of help. Find our guidlines for contributing to this package 
[here](https://github.com/automl/SMAC3/blob/master/.github/CONTRIBUTING.md).

If you use SMAC in one of your research projects, please cite our 
[JMLR paper](https://jmlr.org/papers/v23/21-0888.html):
```
@article{JMLR:v23:21-0888,
  author  = {Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and Difan Deng and Carolin Benjamins and Tim Ruhkopf and René Sass and Frank Hutter},
  title   = {SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {54},
  pages   = {1--9},
  url     = {http://jmlr.org/papers/v23/21-0888.html}
}
```

Copyright (C) 2016-2022  [AutoML Group](http://www.automl.org).
