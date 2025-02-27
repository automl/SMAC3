"""Ask-and-Tell
# Flags: doc-Runnable

This examples show how to use the Ask-and-Tell interface.

Notice, that the ask-and-tell interface will still use the initial design specified in the facade.
Should you which to add your own evaluated configurations instead or deactivate the initial
design all together, please refer to the warmstarting example in conjunction with this one.
"""

from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
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
        target_function=model.train,
        intensifier=intensifier,
        overwrite=True,
    )

    # We can ask SMAC which trials should be evaluated next
    for _ in range(10):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        value = TrialValue(cost=cost, time=0.5)

        smac.tell(info, value)

    # After calling ask+tell, we can still optimize
    # Note: SMAC will optimize the next 90 trials because 10 trials already have been evaluated.
    # If we however choose not to call optimize; e.g. because we want to manage heavy
    # computation of model.train completely outside smac, but still use it to suggest new
    # configurations, then n_trails will only be relevant for the initial design in combination
    # with initial design max_ratio! In fact in an only ask+tell case, we could even set
    # target_function=None in the constructor, because smac wouldn't even need to know
    # what the target function is. But that will prevent us from calling optimize and validate later
    # on.
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
