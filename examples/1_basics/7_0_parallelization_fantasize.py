"""Example of using SMAC with parallelization and fantasization vs. no estimation for pending evaluations.

This example will take some time because the target function is artificially slowed down to demonstrate the effect of
fantasization. The example will plot the incumbent found by SMAC with and without fantasization.
"""
from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float

from matplotlib import pyplot as plt

from smac import BlackBoxFacade, Scenario
from smac.facade import AbstractFacade

from rich import inspect
import time

def plot_trajectory(facades: list[AbstractFacade], names: list[str]) -> None:
    # Plot incumbent
    cmap = plt.get_cmap("tab10")

    fig = plt.figure()
    axes = fig.subplots(1, 2)

    for ax_i, x_axis in zip(axes, ["walltime", "trial"]):
        for i, facade in enumerate(facades):
            X, Y = [], []
            inspect(facade.intensifier.trajectory)
            for item in facade.intensifier.trajectory:
                # Single-objective optimization
                assert len(item.config_ids) == 1
                assert len(item.costs) == 1

                y = item.costs[0]
                x = getattr(item, x_axis)

                X.append(x)
                Y.append(y)

            ax_i.plot(X, Y, label=names[i], color=cmap(i))
            ax_i.scatter(X, Y, marker="x", color=cmap(i))
            ax_i.set_xlabel(x_axis)
            ax_i.set_ylabel(facades[0].scenario.objectives)
            ax_i.set_yscale("log")
            ax_i.legend()

    plt.show()

class Branin():
    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        x1 = Float("x1", (-5, 10), default=0)
        x2 = Float("x2", (0, 15), default=0)

        # Add hyperparameters and conditions to our configspace
        cs.add([x1, x2])

        time.sleep(10)

        return cs

    def train(self, config: Configuration, seed: int) -> float:
        x1 = config["x1"]
        x2 = config["x2"]
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        cost = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        regret = cost - 0.397887

        return regret

if __name__ == "__main__":
    seed = 345455
    scenario = Scenario(n_trials=100, configspace=Branin().configspace, n_workers=4, seed=seed)
    facade = BlackBoxFacade

    smac_noestimation = facade(
        scenario=scenario,
        target_function=Branin().train,
        overwrite=True, 
    )
    smac_fantasize = facade(
        scenario=scenario,
        target_function=Branin().train,
        config_selector=facade.get_config_selector(
            scenario=scenario,
            batch_sampling_estimation_strategy="kriging_believer"
        ),
        overwrite=True,
        logging_level=0
    )
    
    incumbent_noestimation = smac_noestimation.optimize()
    incumbent_fantasize = smac_fantasize.optimize()

    plot_trajectory(facades=[
        smac_noestimation,
        smac_fantasize,
        ], names=["No Estimation", "Fantasize"])

    del smac_noestimation
    del smac_fantasize
