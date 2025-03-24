"""
2D Schaffer Function with Objective Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple example on how to use multi-objective optimization is shown. The 2D Schaffer function is used. In the plot
you can see that all points are on the Pareto front. However, since we set the objective weights, you can notice that
SMAC prioritizes the second objective over the first one.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from matplotlib import pyplot as plt

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.facade import AbstractFacade

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def schaffer(x: float) -> Tuple[float, float]:
    f1 = np.square(x)
    f2 = np.square(np.sqrt(f1) - 2)

    return f1, f2


def target_function(config: Configuration, seed: int = 0) -> Dict[str, float]:
    f1, f2 = schaffer(config["x"])
    return {"metric1": f1, "metric2": f2}


def plot_from_smac(smac: AbstractFacade) -> None:
    plt.figure()
    configs = smac.runhistory.get_configs()
    incumbents = smac.intensifier.get_incumbents()

    for i, config in enumerate(configs):
        if config in incumbents:
            continue

        label = None
        if i == 0:
            label = "Configuration"

        x = config["x"]
        f1, f2 = schaffer(x)
        plt.scatter(f1, f2, c="blue", alpha=0.1, marker="o", zorder=3000, label=label)

    for i, config in enumerate(incumbents):
        label = None
        if i == 0:
            label = "Incumbent"

        x = config["x"]
        f1, f2 = schaffer(x)
        plt.scatter(f1, f2, c="red", alpha=1, marker="x", zorder=3000, label=label)

    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title("Schaffer 2D")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # Simple configspace
    cs = ConfigurationSpace({"x": (-2.0, 2.0)})

    # Scenario object
    scenario = Scenario(
        configspace=cs,
        deterministic=True,  # Only one seed
        n_trials=150,
        objectives=["metric1", "metric2"],
    )

    smac = HPOFacade(
        scenario=scenario,
        target_function=target_function,
        multi_objective_algorithm=HPOFacade.get_multi_objective_algorithm(
            scenario,
            objective_weights=[1, 2],  # Weight metric2 twice as much as metric1
        ),
        overwrite=True,
    )
    incumbents = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Validated costs from default config: \n--- {default_cost}\n")

    print("Validated costs from the Pareto front (incumbents):")
    for incumbent in incumbents:
        cost = smac.validate(incumbent)
        print("---", cost)

    # Plot the evaluated points
    plot_from_smac(smac)
