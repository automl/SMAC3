"""
Multi-Layer Perceptron Using Multiple Epochs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example for optimizing a Multi-Layer Perceptron (MLP) using multiple budgets.
Since we want to take advantage of Multi-Fidelity, the ``MultiFidelityFacade`` is a good choice. By default,
``MultiFidelityFacade`` internally runs with `hyperband <https://arxiv.org/abs/1603.06560>`_, which is a combination of an
aggressive racing mechanism and successive halving. Crucially, the target algorithm function
must accept a budget variable, detailing how much fidelity smac wants to allocate to this
configuration.

MLP is a deep neural network, and therefore, we choose epochs as fidelity type. This implies,
that ``budget`` specifies the number of epochs smac wants to allocate. The digits dataset
is chosen to optimize the average accuracy on 5-fold cross validation.

.. note::

    This example uses the ``MultiFidelityFacade`` facade, which is the closest implementation to
    `BOHB <https://github.com/automl/HpBandSter>`_.
"""

import logging

logging.basicConfig(level=logging.INFO)

import warnings

import numpy as np
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac import MultiFidelityFacade, Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


digits = load_digits()


class MLP:
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
        cs.add_hyperparameters(
            [
                n_layer,
                n_neurons,
                activation,
                solver,
                batch_size,
                learning_rate,
                learning_rate_init,
            ]
        )

        # Adding conditions to restrict the hyperparameter space...
        # ... since learning rate is used when solver is 'sgd'.
        use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
        # ... since learning rate initialization will only be accounted for when using 'sgd' or 'adam'.
        use_lr_init = InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
        # ... since batch size will not be considered when optimizer is 'lbfgs'.
        use_batch_size = InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

        # We can also add multiple conditions on hyperparameters at once:
        cs.add_conditions([use_lr, use_batch_size, use_lr_init])

        return cs

    def train(self, config, seed=0, budget=25) -> float:
        # For deactivated parameters (by virtue of the conditions),
        # the configuration stores None-values.
        # This is not accepted by the MLP, so we replace them with placeholder values.
        lr = config["learning_rate"] if config["learning_rate"] else "constant"
        lr_init = config["learning_rate_init"] if config["learning_rate_init"] else 0.001
        batch_size = config["batch_size"] if config["batch_size"] else 200

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
            score = cross_val_score(classifier, digits.data, digits.target, cv=cv, error_score="raise")

        return 1 - np.mean(score)


if __name__ == "__main__":
    mlp = MLP()
    default_config = mlp.configspace.get_default_configuration()

    # Example call of the target algorithm (for debugging)
    default_value = mlp.train(default_config)
    print(f"Default value: {round(default_value, 2)}")

    # Define our environment variables
    scenario = Scenario(
        mlp.configspace,
        walltime_limit=100,  # After 100 seconds, we stop the optimization
        n_trials=200,  # Evaluate max 100 different trials
        min_budget=5,  # Train the MLP for at least 5 epochs
        max_budget=25,  # Train the MLP for at most 25 epochs
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = MultiFidelityFacade.get_initial_design(scenario, n_configs=5)

    # Create our SMAC object and pass the scenario and the train method
    smac = MultiFidelityFacade(
        scenario,
        mlp.train,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbent = smac.optimize()

    incumbent_value = mlp.train(incumbent)
    print(f"Incumbent value: {round(incumbent_value, 2)}")
