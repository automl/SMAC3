# Sequential Model Algorithm Configuration (SMAC)


[![Tests](https://github.com/automl/SMAC3/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/automl/SMAC3/actions/workflows/pytest.yml)
[![Docs](https://github.com/automl/SMAC3/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/automl/SMAC3/actions/workflows/docs.yml)
[![examples](https://github.com/automl/SMAC3/actions/workflows/terminal_examples.yml/badge.svg?branch=master)](https://github.com/automl/SMAC3/actions/workflows/terminal_examples.yml)
[![codecov
Status](https://codecov.io/gh/automl/SMAC3/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/SMAC3)

SMAC is a tool for algorithm configuration to optimize the parameters of
arbitrary algorithms, including hyperparameter optimization of Machine Learning algorithms. The main core consists of
Bayesian Optimization in combination with an aggressive racing mechanism to
efficiently decide which of two configurations performs better.

For a detailed description of the main ideas, we refer to:

```
Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
Sequential Model-Based Optimization for General Algorithm Configuration
In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)
```

```
Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and Difan Deng and Carolin Benjamins and René Sass and Frank Hutter
SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization
```

SMAC3 is written in Python3 and continuously tested with Python 3.7, 3.8 and 3.9. Its Random
Forest is written in C++. In further texts, SMAC is representatively mentioned for SMAC3.


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
pip install -r requirements.txt
pip install .
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
    """ 
    Trains a random forest on the given hyperparameters, defined by config, and returns the accuracy
    on the validation data.

    Input:
        config (Configuration): Configuration object derived from ConfigurationSpace.

    Return:
        cost (float): Performance measure on the validation data.
    """
    model = RandomForestClassifier(max_depth=config["depth"])
    model.fit(X_train, y_train)

    # define the evaluation metric as return
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

If you use SMAC in one of your research projects, please cite us:
```
@misc{lindauer2021smac3,
      title={SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization}, 
      author={Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and Difan Deng and Carolin Benjamins and René Sass and Frank Hutter},
      year={2021},
      eprint={2109.09831},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


Copyright (C) 2016-2021  [AutoML Group](http://www.automl.org/)




<br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br /><br />















Copyright (C) 2016-2021  [AutoML Group](http://www.automl.org/)

__Attention__: This package is a reimplementation of the original SMAC tool
(see reference below).
However, the reimplementation slightly differs from the original SMAC.
For comparisons against the original SMAC, we refer to a stable release of SMAC (v2) in Java
which can be found [here](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/).

The documentation can be found [here](https://automl.github.io/SMAC3/).



Try SMAC directly in your Browser [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1v0ZH5S9Sfift30GxHAp96e0yZZUFS0Ah)

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


SMAC v3 is written in Python3 and continuously tested with Python 3.7, 3.8 and 3.9. 
Its [Random Forest](https://github.com/automl/random_forest_run) is written in C++.

# Installation

## Requirements

Besides the listed requirements (see `requirements.txt`), the random forest
used in SMAC3 requires SWIG as a build dependency:

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

Please see an overview of all examples in our [documentation](https://automl.github.io/SMAC3/master/examples/index.html).

# Contact

SMAC3 is developed by the [AutoML Groups of the Universities of Hannover and Freiburg](http://www.automl.org/).

If you found a bug, please report to <https://github.com/automl/SMAC3/issues>.

Our guidelines for contributing to this package can be found [here](https://github.com/automl/SMAC3/blob/master/.github/CONTRIBUTING.md)

# Citation

If you use the SMAC3 package, please cite our paper (in addition to the original SMAC paper by Hutter et al. 2011):

```
@inproceedings {lindauer-arxiv21a,
  author = {Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and Difan Deng and Carolin Benjamins and René Sass and Frank Hutter},
  title = {SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization},
  booktitle = {ArXiv: 2109.09831},
  year = {2021},
  url = {https://arxiv.org/abs/2109.09831}
}
```
