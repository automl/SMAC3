"""
2D Schaffer Function with Objective Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple example on how to use multi-objective optimization is shown. The `schaffer` function is used.
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


def plot(all_x: list[float]) -> None:
    plt.figure()
    for x in all_x:
        f1, f2 = schaffer(x)
        plt.scatter(f1, f2, c="blue", alpha=0.1, zorder=3000)

    # plt.vlines([1], 0, 4, linestyles="dashed", colors=["red"])
    # plt.hlines([1], 0, 4, linestyles="dashed", colors=["red"])

    plt.show()


def plot_from_smac(smac: AbstractFacade) -> None:
    rh = smac.runhistory
    all_x = []
    for trial_key in rh:
        config = rh.ids_config[trial_key.config_id]
        all_x.append(config["x"])

    plot(all_x)


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
