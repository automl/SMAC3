"""
Adaptive Capping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adaptive capping is often used in optimization algorithms, particularly in
scenarios where the time taken to evaluate solutions can vary significantly.

"""
import math
import time

import signal
from contextlib import contextmanager

from smac.runhistory import InstanceSeedBudgetKey, TrialInfo

import warnings

from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float


class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutException(f"Function call exceeded timeout of {seconds} seconds")

    # Set the signal handler for the alarm signal
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)  # Schedule an alarm after the given number of seconds

    try:
        yield
    finally:
        # Cancel the alarm if the block finishes before timeout
        signal.alarm(0)


class CappedProblem:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (0, 5), default=5, log=False)
        x1 = Float("x1", (0, 7), default=7, log=False)
        cs.add_hyperparameters([x0, x1])
        return cs

    def train(self, config: Configuration, instance:str, budget, seed: int = 0) -> float:
        x0 = config["x0"]
        x1 = config["x1"]

        try:
            with timeout(int(math.ceil(budget))):
                runtime = 0.5 * x1 + 0.5 * x0 * int(instance)
                time.sleep(runtime)
                return runtime
        except TimeoutException as e:
            print(f"Timeout for configuration {config} with runtime budget {budget}")
            return budget  # FIXME: what should be returned here?


if __name__ == '__main__':
    from smac import HyperparameterOptimizationFacade, RunHistory
    from smac import Scenario

    from smac.intensifier import Intensifier

    capped_problem = CappedProblem()

    scenario = Scenario(
        capped_problem.configspace,
        walltime_limit=3600,  # After 200 seconds, we stop the hyperparameter optimization
        n_trials=500,  # Evaluate max 500 different trials
        instances=['1', '2', '3'],
        instance_features={'1': [1], '2': [2], '3': [3]}
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

    intensifier = Intensifier(scenario, runtime_cutoff=10, adaptive_capping_slackfactor=1.2)

    # Create our SMAC object and pass the scenario and the train method
    smac = HyperparameterOptimizationFacade(
        scenario,
        capped_problem.train,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    # Let's optimize
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(capped_problem.configspace.get_default_configuration())
    print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")
