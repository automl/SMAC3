"""Quadratic Function
# Flags: doc-Runnable

An example of adding additional arguments to the target function either using a class, or a partial function.

This example extends the quadratic function example at examples/1_basics/1_quadratic_function.py.
"""

from functools import partial

from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt

from smac import RunHistory, Scenario
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class AdditionalArgumentsClass:
    def __init__(self, bias:int) -> None:
        self.bias = bias

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (-5, 5), default=-5)
        cs.add([x])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = config["x"]
        return x**2 + self.bias
    
class PartialFunctionClass:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (-5, 5), default=-5)
        cs.add([x])

        return cs

    def train(self, config: Configuration, seed: int = 0, bias:int=0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = config["x"]
        return x**2 + bias

def plot(runhistory: RunHistory, incumbent: Configuration, incumbent_cost:float, bias: int, incumbent_color:str, color:str) -> None:
    # Get all configurations and costs
    configs = [config["x"] for config in runhistory.get_configs()]
    costs = [config.cost for config in runhistory.values()]

    # Plot all trials
    plt.scatter(configs, costs, c=color, alpha=0.4, zorder=9999, marker="o", label=f"Model with bias {bias}")

    # Plot incumbent
    plt.scatter(incumbent["x"], incumbent_cost, c=incumbent_color,s = 100, zorder=10000, marker="x", label=f"Incumbent with bias {bias}")


if __name__ == "__main__":
    for model, bias, color, incumbent_color in [(AdditionalArgumentsClass(bias=2), 2, "green", "red"), (PartialFunctionClass(), -2, "blue", "orange")]:
        # Scenario object specifying the optimization "environment"
        seed = 0 if isinstance(model, AdditionalArgumentsClass) else 1
        scenario = Scenario(model.configspace, deterministic=True, n_trials=100, seed=seed)

        if isinstance(model, PartialFunctionClass):
            model.train = partial(model.train, bias=-2)

        # Now we use SMAC to find the best hyperparameters
        smac = HPOFacade(
            scenario,
            model.train,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        )

        incumbent_config = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(model.configspace.get_default_configuration())
        print(f"Default cost: {default_cost}")

        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent_config)
        print(f"Incumbent cost: {incumbent_cost}")

        # Let's plot it too
        plot(smac.runhistory, incumbent_config, incumbent_cost, bias=bias, color=color, incumbent_color=incumbent_color)
    plt.legend()
    plt.show()
