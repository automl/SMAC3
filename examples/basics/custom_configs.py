"""
Start with Custom Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to incorporate evaluated configurations into SMAC.
"""

import copy
import time
import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from ConfigSpace.conditions import InCondition
from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score

from smac import HyperparameterFacade, Scenario
from smac.runhistory import RunHistory, StatusType

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()


class SVM:
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        kernel = Categorical("kernel", ["linear", "poly", "rbf", "sigmoid"], default="poly")
        C = Float("C", (0.001, 1000.0), default=1.0, log=True)
        shrinking = Categorical("shrinking", [True, False], default=True)
        degree = Integer("degree", (1, 5), default=3)
        coef = Float("coef0", (0.0, 10.0), default=0.0)
        gamma = Categorical("gamma", ["auto", "value"], default="auto")
        gamma_value = Float("gamma_value", (0.0001, 8.0), default=1.0, log=True)

        # Then we create dependencies
        use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
        use_coef = InCondition(child=coef, parent=kernel, values=["poly", "sigmoid"])
        use_gamma = InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"])
        use_gamma_value = InCondition(child=gamma_value, parent=gamma, values=["value"])

        # Add hyperparameters and conditions to our configspace
        cs.add_hyperparameters([kernel, C, shrinking, degree, coef, gamma, gamma_value])
        cs.add_conditions([use_degree, use_coef, use_gamma, use_gamma_value])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        config_dict = config.get_dictionary()
        if "gamma" in config:
            config_dict["gamma"] = config_dict["gamma_value"] if config_dict["gamma"] == "value" else "auto"
            config_dict.pop("gamma_value", None)

        classifier = svm.SVC(**config_dict, random_state=seed)
        scores = cross_val_score(classifier, iris.data, iris.target, cv=5)
        cost = 1 - np.mean(scores)

        return cost


if __name__ == "__main__":
    classifier = SVM()
    configspace = classifier.configspace
    default_config = configspace.get_default_configuration()
    seed = 99

    # Example call of the target algorithm
    default_value = classifier.train(default_config)
    print(f"Default value: {round(default_value, 2)}")

    # Now we create an empty runhistory to add random configurations with our values to start with
    runhistory = RunHistory()
    for config in configspace.sample_configuration(10):
        start_time = time.time()
        cost = classifier.train(copy.deepcopy(config), seed)
        train_time = time.time() - start_time
        runhistory.add(config, cost=cost, time=train_time, seed=seed, status=StatusType.SUCCESS)

    # Next, we create an object, holding general information about the run
    scenario = Scenario(
        configspace,
        # We want to run max 100 trials (combination of config and seed) on top of what's already in the runhistory
        n_trials=100,
    )

    intensifier = HyperparameterFacade.get_intensifier(
        scenario,
        max_config_calls=2,  # We only want to use two seeds per config
    )

    # We use the hyperparameter facade to run SMAC
    smac = HyperparameterFacade(
        scenario,
        classifier.train,
        intensifier=intensifier,
        runhistory=runhistory,
        overwrite=True,  # Disables to continue the run
    )
    incumbent = smac.optimize()

    incumbent_value = classifier.train(incumbent)
    print(f"Incumbent value: {round(incumbent_value, 2)}")
