"""
Ask-and-Tell
^^^^^^^^^^^^

This examples shows how to dynamically adapt the configuration space during the optimization process.
"""

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac.utils.configspace import modify_hyperparameter, recreate_configspace, update_configspace
from smac import HyperparameterOptimizationFacade, Scenario
from smac.acquisition.maximizer import RandomSearch
from smac.acquisition.function import EI
from smac.runhistory.dataclasses import TrialValue

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class Rosenbrock2D:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-3)
        x1 = Float("x1", (-5, 10), default=-4)
        cs.add([x0, x1])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """The 2-dimensional Rosenbrock function as a toy model.
        The Rosenbrock function is well know in the optimization community and
        often serves as a toy problem. It can be defined for arbitrary
        dimensions. The minimium is always at x_i = 1 with a function value of
        zero. All input parameters are continuous. The search domain for
        all x's is the interval [-5, 10].
        """
        x1 = config["x0"]
        x2 = config["x1"]

        cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
        return cost


if __name__ == "__main__":
    model = Rosenbrock2D()

    # Scenario object
    scenario = Scenario(model.configspace, deterministic=False, n_trials=100)

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,  # We basically use one seed per config only
    )

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        model.train,
        intensifier=intensifier,
        overwrite=True,
        acquisition_maximizer=RandomSearch(scenario.configspace, EI()),
    )

    # Run for 20 trials, the initial design
    for _ in range(20):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost, time=0.5)

        smac.tell(info, value)

    # Modify the configuration space
    hyperparameters = dict(scenario.configspace)
    hyperparameter_name = "x0"
    modifications = {"lower": 3, "upper": 4, "default_value": 3.5}
    new_hyperparameters = modify_hyperparameter(
        hyperparameters, hyperparameter_name, **modifications
    )
    new_configspace = recreate_configspace(scenario.configspace.name, new_hyperparameters)
    update_configspace(smac, new_configspace)

    # Sample configurations step by step. Watch that x0 is now in [3, 4]
    for _ in range(10):
        info = smac.ask()
        assert info.seed is not None
        assert 3 <= info.config["x0"] <= 4

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost, time=0.5)

        smac.tell(info, value)

    # After calling ask+tell, we can still optimize
    # Note: SMAC will optimize the next 90 trials because 10 trials already have been evaluated
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
