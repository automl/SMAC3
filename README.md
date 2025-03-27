# SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization


[![Tests](https://github.com/automl/SMAC3/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/automl/SMAC3/actions/workflows/pytest.yml)
[![Documentation](https://github.com/automl/SMAC3/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/automl/SMAC3/actions/workflows/docs.yml)
[![codecov
Status](https://codecov.io/gh/automl/SMAC3/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/SMAC3)

<img src="docs/images/logo.png" style="width: 50%;" />

SMAC offers a robust and flexible framework for Bayesian Optimization to support users in determining well-performing 
hyperparameter configurations for their (Machine Learning) algorithms, datasets and applications at hand. The main core 
consists of Bayesian Optimization in combination with an aggressive racing mechanism to efficiently decide which of two configurations performs better.

MO-SMAC is implemented directly into SMAC3. This repository is forked from the [SMAC3 repository](https://github.com/automl/SMAC3) and therefore contains references and copyright information to those authors. 
These do not align with the authors of MO-SMAC and, therefore, the anonymity for this repository remains intact.

## Installation

Create a new environment with Python 3.10 and make sure swig is installed either on your system or
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

Clone this repository and install locally:
```

If you want to contribute to SMAC, use the following steps instead:
```
git clone https://github.com/automl/SMAC3.git && cd SMAC3
make install-dev
cd SMAC3K
pip install -e .[dev]
```

## Minimal Example
To use MO-SMAC, there is a multi-objective facade that provides all the functionalities for MO-AAC. The example below shows how this facade can be accessed and used. 

```py
from ConfigSpace import Configuration, ConfigurationSpace

import time 
import numpy as np
from smac.facade.multi_objective_facade import MultiObjectiveFacade
from smac import Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    start_time = time.time()
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    run_time = time.time() - start_time
    return {"perf": 1 - np.mean(scores), "runtime": run_time}


configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, 
                    deterministic=True, 
                    n_trials=200,
                    objectives=["perf", "runtime"])

# Use SMAC to find the best configuration/hyperparameters
smac = MultiObjectiveFacade(scenario, train)
incumbent = smac.optimize()
```
