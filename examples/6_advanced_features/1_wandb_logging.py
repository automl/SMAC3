"""
Use Weights and Biases for logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to use Weights and Biases (WandB) for logging.

To use WandB, you need to install the package via pip:

.. code-block:: bash

        pip install wandb

Then you can use the WandBCallback to log the results of the optimization as well as intermediate information to WandB.
This is done by creating a WandBCallback object and passing it to the used Facade.

"""
from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import smac
from smac import HyperparameterOptimizationFacade, Scenario
from smac.callback import WandBCallback

iris = datasets.load_iris()


def train(config: Configuration, seed: int = 0) -> float:
    classifier = SVC(C=config["C"], random_state=seed)
    scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)


configspace = ConfigurationSpace({"C": (0.100, 1000.0)})

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=100, seed=3)

wandb_callback = WandBCallback(
    project="smac-dev",
    entity="benjamc",
    config=Scenario.make_serializable(scenario),
)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train, callbacks=[wandb_callback], overwrite=True)
incumbent = smac.optimize()
