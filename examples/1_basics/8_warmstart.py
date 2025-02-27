"""Warmstarting SMAC
# Flags: doc-Runnable

With the ask and tell interface, we can support warmstarting SMAC. We can communicate rich
information about the previous trials to SMAC using `TrialInfo` and `TrialValue` instances.
For more details on ask and tell consult the [info page ask-and-tell](../../../advanced_usage/5_ask_and_tell).
"""
from __future__ import annotations

from smac.scenario import Scenario
from smac.facade import HyperparameterOptimizationFacade
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac.runhistory.dataclasses import TrialValue, TrialInfo


class Rosenbrock2D:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-3)
        x1 = Float("x1", (-5, 10), default=-4)
        cs.add([x0, x1])

        return cs

    def evaluate(self, config: Configuration, seed: int = 0) -> float:
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
    SEED = 12345
    task = Rosenbrock2D()

    # Previous evaluations
    # X vectors need to be connected to the configuration space
    configurations = [
        Configuration(task.configspace, {'x0':1, 'x1':2}),
        Configuration(task.configspace, {'x0':-1, 'x1':3}),
        Configuration(task.configspace, {'x0':5, 'x1':5}),
    ]
    costs = [task.evaluate(c, seed=SEED) for c in configurations]

    # Define optimization problem and budget
    scenario = Scenario(task.configspace, deterministic=False, n_trials=30)
    intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=1)
    smac = HyperparameterOptimizationFacade(
        scenario,
        target_function=task.evaluate,
        intensifier=intensifier,
        overwrite=True,

        # Modify the initial design to use our custom initial design
        initial_design=HyperparameterOptimizationFacade.get_initial_design(
            scenario, 
            n_configs=0,  # Do not use the default initial design at all

            # You can pass the configurations as additional_configs, which will specify their
            # origin to be the initial design. However, this is not necessary and we can just
            # smac.tell the configurations.
            # additional_configs=configurations  # Use the configurations previously evaluated as initial design
                                               # This only passes the configurations but not the cost!
                                               # So in order to actually use the custom, pre-evaluated initial design
                                               # we need to tell those trials, like below.
        )
    )

    # Convert previously evaluated configurations into TrialInfo and TrialValue instances to pass to SMAC
    trial_infos = [TrialInfo(config=c, seed=SEED) for c in configurations]
    trial_values = [TrialValue(cost=c) for c in costs]

    # Warmstart SMAC with the trial information and values
    for info, value in zip(trial_infos, trial_values):
        smac.tell(info, value)

    # Optimize as usual
    # Notice, that since we added three configurations, n_trials for the remaining optimization
    # is effectively 27 in optimize().
    smac.optimize()
