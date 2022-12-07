"""
Continue an Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

SMAC offers the possibility to continue an optimization. In this example, an optimization of a simple quadratic function
is continued. We use a custom callback, to artificially stop the first optimization.
"""

from __future__ import annotations
from ConfigSpace import Configuration, ConfigurationSpace, Float


from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario, Callback
from smac.runhistory import TrialInfo, TrialValue
from smac.main.smbo import SMBO

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class StopCallback(Callback):
    def __init__(self, stop_after: int):
        self._stop_after = stop_after

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Called after the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization.
        """
        if smbo.runhistory.finished == self._stop_after:
            return False

        return None


class QuadraticFunction:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (-5, 5), default=-5)
        cs.add_hyperparameters([x])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum at x=0."""
        x = config["x"]
        return x * x


if __name__ == "__main__":
    model = QuadraticFunction()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, deterministic=True, n_trials=50)
    stop_after = 10

    # Now we use SMAC to find the best hyperparameters
    smac = HPOFacade(
        scenario,
        model.train,  # We pass the target function here
        callbacks=[StopCallback(stop_after=stop_after)],
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )

    incumbent = smac.optimize()
    assert smac.runhistory.finished == stop_after

    # Now, we want to continue the optimization
    # Make sure, we don't overwrite the last run
    smac2 = HPOFacade(scenario, model.train, overwrite=False)

    # Check whether we get the same incumbent
    assert smac.intensifier.get_incumbent() == smac2.intensifier.get_incumbent()
    assert smac2.runhistory.finished == stop_after

    # And now we finish the optimization
    incumbent2 = smac2.optimize()

    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost of first run: {incumbent_cost}")

    incumbent_cost = smac2.validate(incumbent2)
    print(f"Incumbent cost of continued run: {incumbent_cost}")
