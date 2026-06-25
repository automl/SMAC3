"""ParEGO
# Flags: doc-Runnable

An example of how to use multi-objective optimization with PHVI for an algorithm configuration scenario.
Both accuracy and run-time are going to be optimized on the digits dataset using an MLP, and the configurations are
shown in a plot, highlighting the best ones in a Pareto front. The red cross indicates the best configuration selected
by SMAC.

In the optimization, SMAC evaluates the configurations on two different seeds. Therefore, the plot shows the
mean accuracy and run-time of each configuration.
"""
from __future__ import annotations

import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.datasets import load_digits, load_iris, load_wine, load_diabetes
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac import Scenario
from smac.facade.multi_objective_facade import MultiObjectiveFacade as MOFacade

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

from smac.main.config_selector import ConfigSelector

digits = load_digits()

instances = {"digits": load_digits(), "iris":load_iris(), "wine": load_wine(), "diabetes": load_diabetes()}

class MLP:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        n_layer = Integer("n_layer", (1, 5), default=1)
        n_neurons = Integer("n_neurons", (8, 256), log=True, default=10)
        activation = Categorical("activation", ["logistic", "tanh", "relu"], default="tanh")
        solver = Categorical("solver", ["lbfgs", "sgd", "adam"], default="adam")
        batch_size = Integer("batch_size", (30, 300), default=200)
        learning_rate = Categorical("learning_rate", ["constant", "invscaling", "adaptive"], default="constant")
        learning_rate_init = Float("learning_rate_init", (0.0001, 1.0), default=0.001, log=True)

        cs.add([n_layer, n_neurons, activation, solver, batch_size, learning_rate, learning_rate_init])

        use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
        use_lr_init = InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
        use_batch_size = InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

        # We can also add multiple conditions on hyperparameters at once:
        cs.add([use_lr, use_batch_size, use_lr_init])

        return cs

    @staticmethod
    def train(config: Configuration, instance:str = "none", seed: int = 0, budget: int = 10) -> dict[str, float]:
        lr = config.get("learning_rate", "constant")
        lr_init = config.get("learning_rate_init", 0.001)
        batch_size = config.get("batch_size", 200)

        start_time = time.time()

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
            score = cross_val_score(classifier, instances[instance].data, instances[instance].target, cv=cv, error_score="raise")

        return {
            "1 - accuracy": 1 - np.mean(score),
            "time": time.time() - start_time,
        }


def plot_pareto(smac: AbstractFacade, incumbents: list[Configuration]) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    average_costs = []
    average_pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if config in incumbents:
            average_pareto_costs += [average_cost]
        else:
            average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)
    pareto_costs = np.vstack(average_pareto_costs)
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plt.scatter(costs_x, costs_y, marker="x", label="Configuration")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    plt.title("Pareto-Front")
    plt.xlabel(smac.scenario.objectives[0])
    plt.ylabel(smac.scenario.objectives[1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    mlp = MLP()
    objectives = ["1 - accuracy", "time"]

    # Define our environment variables
    scenario = Scenario(
        mlp.configspace,
        objectives=objectives,
        instances=list(instances.keys()),
        instance_features={inst: [float(num),]for num, inst in  enumerate(instances.keys())},
        deterministic=False,
        walltime_limit=30,  # After 30 seconds, we stop the hyperparameter optimization
        n_trials=2000,  # Evaluate max 200 different trials
        n_workers=1,
    )

    # We use the SMAC2 settings of balancing the retraining on a wallclock ratio
    config_selector = ConfigSelector(scenario, retrain_after=None, retrain_wallclock_ratio=0.5, min_configurations=2)

    # Create our SMAC object and pass the scenario and the train method
    smac = MOFacade(
        scenario,
        mlp.train,
        config_selector=config_selector,
        overwrite=True,
    )

    # Let's optimize
    incumbents = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(mlp.configspace.get_default_configuration())
    print(f"Validated costs from default config: \n--- {default_cost}\n")

    print("Validated costs from the Pareto front (incumbents):")
    for incumbent in incumbents:
        cost = smac.validate(incumbent)
        print("---", cost)

    # Let's plot a pareto front
    plot_pareto(smac, incumbents)
