from __future__ import annotations

import warnings

import numpy as np
import pytest
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class Rosenbrock2D:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-3)
        x1 = Float("x1", (-5, 10), default=-4)
        cs.add_hyperparameters([x0, x1])

        return cs

    def train(self, config: Configuration, seed: int = 0, budget: float = None, instance: str = None) -> float:
        """The 2-dimensional Rosenbrock function as a toy model.
        The Rosenbrock function is well-known in the optimization community and
        often serves as a toy problem. It can be defined for arbitrary
        dimensions. The minimum is always at x_i = 1 with a function value of
        zero. All input parameters are continuous. The search domain for
        all x's is the interval [-5, 10].
        """
        x1 = config["x0"]
        x2 = config["x1"]

        cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
        return cost


@pytest.fixture
def rosenbrock() -> Rosenbrock2D:
    return Rosenbrock2D()


class SGD:
    def __init__(self, dataset: "Dataset") -> None:
        self.dataset = dataset

    @property
    def configspace(self) -> ConfigurationSpace:
        """Build the configuration space which defines all parameters and their ranges for the SGD classifier."""
        cs = ConfigurationSpace(seed=0)

        # We define a few possible parameters for the SGD classifier
        alpha = Float("alpha", (0, 1), default=1.0)
        l1_ratio = Float("l1_ratio", (0, 1), default=0.5)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        eta0 = Float("eta0", (0.00001, 1), default=0.1, log=True)

        # Add the parameters to configuration space
        cs.add_hyperparameters([alpha, l1_ratio, learning_rate, eta0])

        return cs

    def train(self, config: Configuration, instance: str = "0-1", budget: float = 1, seed: int = 0) -> float:
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
                max_iter=budget,
                early_stopping=True,
                random_state=seed,
            )

            # Get instance
            data, target = self.dataset.get_instance_data(instance)

            cv = StratifiedKFold(n_splits=4, random_state=seed, shuffle=True)  # to make CV splits consistent
            scores = cross_val_score(clf, data, target, cv=cv)

        return 1 - np.mean(scores)


@pytest.fixture
def make_sgd():
    def create(dataset: "Dataset") -> SGD:
        return SGD(dataset)

    return create
