# Sequential Model Algorithm Configuration (SMAC)


[![Tests](https://github.com/automl/SMAC3/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/automl/SMAC3/actions/workflows/pytest.yml)
[![Docs](https://github.com/automl/SMAC3/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/automl/SMAC3/actions/workflows/docs.yml)
[![Examples](https://github.com/automl/SMAC3/actions/workflows/examples.yml/badge.svg?branch=main)](https://github.com/automl/SMAC3/actions/workflows/examples.yml)
[![codecov
Status](https://codecov.io/gh/automl/SMAC3/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/SMAC3)

SMAC is a tool for algorithm configuration to optimize the parameters of
arbitrary algorithms, including hyperparameter optimization of Machine Learning algorithms. The main core consists of
Bayesian Optimization in combination with an aggressive racing mechanism to
efficiently decide which of two configurations performs better.

SMAC3 is written in Python3 and continuously tested with Python 3.7, 3.8, 3.9, and 3.10. Its Random
Forest is written in C++. In further texts, SMAC is representatively mentioned for SMAC3.

> [Documention](https://automl.github.io/SMAC3)

> [Roadmap](https://github.com/orgs/automl/projects/5/views/2)


## Installation

Create a new environment with python 3.9 and make sure swig is installed either on your system or
inside the environment. We demonstrate the installation via anaconda in the following:

Create and activate environment:
```
conda create -n SMAC python=3.9
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

We refer to the [documention](https://automl.github.io/SMAC3) for further installation options.


## Minimal Example

```py
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario


X_train, y_train = np.random.randint(2, size=(20, 2)), np.random.randint(2, size=20)
X_val, y_val = np.random.randint(2, size=(5, 2)), np.random.randint(2, size=5)


def train_random_forest(config):
    model = RandomForestClassifier(max_depth=config["depth"])
    model.fit(X_train, y_train)

    # Define the evaluation metric as return
    return 1 - model.score(X_val, y_val)


if __name__ == "__main__":
    # Define your hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformIntegerHyperparameter("depth", 2, 100))

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 10,  # Max number of function evaluations (the more the better)
        "cs": configspace,
    })

    smac = SMAC4BB(scenario=scenario, tae_runner=train_random_forest)
    best_found_config = smac.optimize()

```

More examples can be found in the [documention](https://automl.github.io/SMAC3).



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

If you have found a bug, please report to [issues](https://github.com/automl/SMAC3/issues). Moreover, we are appreciating any kind of help.
Find our guidlines for contributing to this package [here](https://github.com/automl/SMAC3/blob/master/.github/CONTRIBUTING.md).

If you use SMAC in one of your research projects, please cite our [JMLR paper](https://jmlr.org/papers/v23/21-0888.html):
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

Copyright (C) 2016-2022  [AutoML Group](http://www.automl.org/).
