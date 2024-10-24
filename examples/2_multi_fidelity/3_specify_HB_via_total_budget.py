"""Specify Number of Trials via a Total Budget in Hyperband

This example uses a dummy function but illustrates how to setup Hyperband if you 
want to specify a total optimization budget in terms of fidelity units.

In Hyperband, normally SMAC calculates a typical Hyperband round.
If the number of trials is not used up by one single round, the next round is started.
Instead of specifying the number of trial beforehand, specify the total budget
in terms of the fidelity units and let SMAC calculate how many trials that would be.


"""
from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt

from smac import MultiFidelityFacade, RunHistory, Scenario
from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class QuadraticFunction:
    max_budget = 500

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (-5, 5), default=-5)
        cs.add([x])

        return cs

    def train(self, config: Configuration, seed: int = 0, budget: float | None = None) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = config["x"]

        if budget is None:
            multiplier = 1
        else:
            multiplier = 1 + budget / self.max_budget

        return x**2 * multiplier


def plot(runhistory: RunHistory, incumbent: Configuration) -> None:
    plt.figure()

    # Plot ground truth
    x = list(np.linspace(-5, 5, 100))
    y = [xi * xi for xi in x]
    plt.plot(x, y)

    # Plot all trials
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        x = config["x"]
        y = v.cost  # type: ignore
        plt.scatter(x, y, c="blue", alpha=0.1, zorder=9999, marker="o")

    # Plot incumbent
    plt.scatter(incumbent["x"], incumbent["x"] * incumbent["x"], c="red", zorder=10000, marker="x")

    plt.show()


if __name__ == "__main__":
    model = QuadraticFunction()

    min_budget = 10  # minimum budget per trial
    max_budget = 500  # maximum budget per trial
    eta = 3  # standard HB parameter influencing the number of stages

    # Let's calculate how many trials we need to exhaust the total optimization budget (in terms of
    # fidelity units)
    n_trials = get_n_trials_for_hyperband_multifidelity(
        total_budget=10000,  # this is the total optimization budget we specify in terms of fidelity units
        min_budget=min_budget,  # This influences the Hyperband rounds, minimum budget per trial
        max_budget=max_budget,  # This influences the Hyperband rounds, maximum budget per trial
        eta=eta,  # This influences the Hyperband rounds
        print_summary=True,
    )

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(
        model.configspace, deterministic=True, n_trials=n_trials, min_budget=min_budget, max_budget=max_budget
    )

    # Now we use SMAC to find the best hyperparameters
    smac = MultiFidelityFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        intensifier=MultiFidelityFacade.get_intensifier(scenario=scenario, eta=eta),
    )

    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    # Let's plot it too
    plot(smac.runhistory, incumbent)
