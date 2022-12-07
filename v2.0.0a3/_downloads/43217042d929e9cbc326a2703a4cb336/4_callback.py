"""
Custom Callback
^^^^^^^^^^^^^^^

Using callbacks is the easieast way to integrate custom code inside the Bayesian optimization loop.
In this example, we disable SMAC's default logging option and use the custom callback to log the evaluated trials.
Furthermore, we print some stages of the optimization process.
"""

from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace, Float

import smac
from smac import Callback
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory import TrialInfo, TrialValue

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

    def train(self, config: Configuration, seed: int = 0) -> float:
        x1 = config["x0"]
        x2 = config["x1"]

        cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
        return cost


class CustomCallback(Callback):
    def __init__(self) -> None:
        self.trials_counter = 0

    def on_start(self, smbo: smac.main.smbo.SMBO) -> None:
        print("Let's start!")
        print("")

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        self.trials_counter += 1
        if self.trials_counter % 10 == 0:
            print(f"Evaluated {self.trials_counter} trials so far.")

            incumbent = smbo.intensifier.get_incumbent()
            assert incumbent is not None
            print(f"Current incumbent: {incumbent.get_dictionary()}")
            print(f"Current incumbent value: {smbo.runhistory.get_cost(incumbent)}")
            print("")

        if self.trials_counter == 50:
            print(f"We just triggered to stop the optimization after {smbo.runhistory.finished} finished trials.")
            return False

        return None


if __name__ == "__main__":
    model = Rosenbrock2D()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, n_trials=200)

    # Now we use SMAC to find the best hyperparameters
    HPOFacade(
        scenario,
        model.train,
        overwrite=True,
        callbacks=[CustomCallback()],
        logging_level=999999,
    ).optimize()
