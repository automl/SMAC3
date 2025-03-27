"""
Stochastic Gradient Descent On Multiple Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example for optimizing a Multi-Layer Perceptron (MLP) across multiple (dataset) instances.

Alternative to budgets, here wlog. we consider instances as a fidelity type. An instance represents a specific
scenario/condition (e.g. different datasets, subsets, transformations) for the algorithm to run. SMAC then returns the
algorithm that had the best performance across all the instances. In this case, an instance is a binary dataset i.e.,
digit-2 vs digit-3.

If we use instance as our fidelity, we need to initialize scenario with argument instance. In this case the argument
budget is no longer required by the target function. But due to the scenario instance argument,
the target function now is required to have an instance argument.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class DigitsDataset:
    def __init__(self) -> None:
        self._data = datasets.load_digits()

    def get_instances(self) -> list[str]:
        """Create instances from the dataset which include two classes only."""
        return [f"{classA}-{classB}" for classA, classB in itertools.combinations(self._data.target_names, 2)]

    def get_instance_features(self) -> dict[str, list[int | float]]:
        """Returns the mean and variance of all instances as features."""
        features = {}
        for instance in self.get_instances():
            data, _ = self.get_instance_data(instance)
            features[instance] = [np.mean(data), np.var(data)]

        return features

    def get_instance_data(self, instance: str) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve data from the passed instance."""
        # We split the dataset into two classes
        classA, classB = instance.split("-")
        indices = np.where(np.logical_or(int(classA) == self._data.target, int(classB) == self._data.target))

        data = self._data.data[indices]
        target = self._data.target[indices]

        return data, target


class SGD:
    def __init__(self, dataset: DigitsDataset) -> None:
        self.dataset = dataset

    @property
    def configspace(self) -> ConfigurationSpace:
        """Build the configuration space which defines all parameters and their ranges for the SGD classifier."""
        cs = ConfigurationSpace()

        # We define a few possible parameters for the SGD classifier
        alpha = Float("alpha", (0, 1), default=1.0)
        l1_ratio = Float("l1_ratio", (0, 1), default=0.5)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        eta0 = Float("eta0", (0.00001, 1), default=0.1, log=True)
        # Add the parameters to configuration space
        cs.add_hyperparameters([alpha, l1_ratio, learning_rate, eta0])

        return cs

    def train(self, config: Configuration, instance: str, seed: int = 0) -> float:
        """Creates a SGD classifier based on a configuration and evaluates it on the
        digits dataset using cross-validation."""

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # SGD classifier using given configuration
            clf = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=config["alpha"],
                l1_ratio=config["l1_ratio"],
                learning_rate=config["learning_rate"],
                eta0=config["eta0"],
                max_iter=30,
                early_stopping=True,
                random_state=seed,
            )

            # get instance
            data, target = self.dataset.get_instance_data(instance)

            cv = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)  # to make CV splits consistent
            scores = cross_val_score(clf, data, target, cv=cv)

        return 1 - np.mean(scores)


if __name__ == "__main__":
    dataset = DigitsDataset()
    model = SGD(dataset)

    scenario = Scenario(
        model.configspace,
        walltime_limit=30,  # We want to optimize for 30 seconds
        n_trials=5000,  # We want to try max 5000 different trials
        min_budget=1,  # Use min one instance
        max_budget=45,  # Use max 45 instances (if we have a lot of instances we could constraint it here)
        instances=dataset.get_instances(),
        instance_features=dataset.get_instance_features(),
    )

    # Create our SMAC object and pass the scenario and the train method
    smac = MFFacade(
        scenario,
        model.train,
        overwrite=True,
    )

    # Now we start the optimization process
    incumbent = smac.optimize()

    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
