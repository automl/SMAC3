"""
Ask-and-Tell Interface
^^^^^^^^^^^^^^^^^^

This examples show how to use the Ask-and-Tell interface.
"""

from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import HyperparameterFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue

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
    scenario = Scenario(
        model.configspace,
        deterministic=False,
        n_trials=100,
    )

    # Example call of the target function
    default_value = model.train(model.configspace.get_default_configuration())
    print(f"Default value: {round(default_value, 2)}")

    intensifier = HyperparameterFacade.get_intensifier(
        scenario,
        max_config_calls=1,  # We basically use one seed only
    )

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterFacade(
        scenario,
        model.train,
        intensifier=intensifier,
        overwrite=True,
    )

    # We can provide SMAC with custom configurations first
    for config in model.configspace.sample_configuration(10):
        for seed in smac.get_target_function_seeds():
            cost = model.train(config, seed=seed)

            trial_info = TrialInfo(config, seed=seed)
            trial_value = TrialValue(cost=cost, time=0.5)

            smac.tell(trial_info, trial_value)

    # NOTE: Using only the tell method does not increase the number of submitted trials.
    # Hence, the optimize method will process additional 100 trials.
    smac.optimize()

    # Now we retrieve the best configuration using our runhistory
    incumbent = smac.runhistory.get_incumbent()

    incumbent_value = model.train(incumbent)
    print(f"Incumbent value: {round(incumbent_value, 2)}")
