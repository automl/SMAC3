"""
2D Schaffer Function
^^^^^^^^^^^^^^^^^^^^

A simple example on how to use multi-objective optimization is shown. The `schaffer` function is used.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt

from smac import HyperparameterFacade, Scenario
from smac.facade import Facade
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def schaffer(x: float) -> Tuple[float, float]:
    f1 = np.square(x)
    f2 = np.square(np.sqrt(f1) - 2)

    return f1, f2


def target_algorithm(config: Configuration, seed: int = 0) -> Dict[str, float]:
    f1, f2 = schaffer(config["x"])
    return {"metric1": f1, "metric2": f2}


def plot(all_x: list[float]) -> None:
    plt.figure()
    for x in all_x:
        f1, f2 = schaffer(x)
        plt.scatter(f1, f2, c="blue", alpha=0.1, zorder=3000)

    plt.vlines([1], 0, 4, linestyles="dashed", colors=["red"])
    plt.hlines([1], 0, 4, linestyles="dashed", colors=["red"])

    plt.show()


def plot_from_smac(smac: Facade) -> None:
    rh = smac.runhistory
    all_x = []
    for runkey in rh.data.keys():
        config = rh.ids_config[runkey.config_id]
        all_x.append(config["x"])

    plot(all_x)


if __name__ == "__main__":
    # Simple configspace
    cs = ConfigurationSpace({"x": (-2.0, 2.0)})

    # Scenario object
    scenario = Scenario(
        configspace=cs,
        n_trials=150,
        objectives=["metric1", "metric2"],
    )

    smac = HyperparameterFacade(
        scenario=scenario,
        target_algorithm=target_algorithm,
        multi_objective_algorithm=MeanAggregationStrategy(scenario),
        overwrite=True,
    )
    incumbent = smac.optimize()

    # Plot the evaluated points
    plot_from_smac(smac)
