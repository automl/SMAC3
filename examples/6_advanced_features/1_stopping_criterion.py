"""
Use Stopping Criterion to stop the Optimization Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to use the stopping criterion by Makarova et al. (2021) [1] to stop the optimization process. It
is implemented in SMAC as a callback function.

[1] Makarova, Anastasia, et al. "Automatic Termination for Hyperparameter Optimization." First Conference on
    Automated Machine Learning (Main Track). 2022.
"""

import os

from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from smac.callback.stopping_callback import StoppingCallback

iris = datasets.load_iris()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores), {'std_crossval': 1 - np.std(scores), 'folds': 5, 'data_points': len(iris.data)}


configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=10)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train, callbacks=[StoppingCallback()])
incumbent = smac.optimize()

print(incumbent)

# delete smac output folder as I want to run this script multiple times
os.system("rm -rf smac3_output")