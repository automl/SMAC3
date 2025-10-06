from __future__ import annotations

import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    Integer,
)
from ConfigSpace.conditions import InCondition
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from src.models.model import Model


class MLPModel(Model):
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types, we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical("activation", ["logistic", "tanh", "relu"], default="tanh")
        solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
        batch_size = Integer("batch_size", (30, 300), default=200)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        learning_rate_init = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)

        # Add all hyperparameters at once:
        cs.add([n_layer, n_neurons, activation, solver, batch_size, learning_rate, learning_rate_init])

        # Adding conditions to restrict the hyperparameter space...
        # ... since learning rate is used when solver is 'sgd'.
        use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
        # ... since learning rate initialization will only be accounted for when using 'sgd' or 'adam'.
        use_lr_init = InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
        # ... since batch size will not be considered when optimizer is 'lbfgs'.
        use_batch_size = InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

        # We can also add multiple conditions on hyperparameters at once:
        cs.add([use_lr, use_batch_size, use_lr_init])

        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
        assert self.dataset is not None
        # For deactivated parameters (by virtue of the conditions),
        # the configuration stores None-values.
        # This is not accepted by the MLP, so we replace them with placeholder values.
        lr = config["learning_rate"] if config["learning_rate"] else "constant"
        lr_init = config["learning_rate_init"] if config["learning_rate_init"] else 0.001
        batch_size = config["batch_size"] if config["batch_size"] else 200

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            classifier = MLPClassifier(
                hidden_layer_sizes=[config["n_neurons"]] * config["n_layer"],
                solver=config["solver"],
                batch_size=batch_size,
                activation=config["activation"],
                learning_rate=lr,
                learning_rate_init=lr_init,
                max_iter=int(np.ceil(budget)),
                random_state=seed,
            )

            # Returns the 5-fold cross validation accuracy
            cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)  # to make CV splits consistent
            score = cross_val_score(classifier, self.dataset.get_X(), self.dataset.get_Y(), cv=cv, error_score="raise")

        return 1 - np.mean(score)
