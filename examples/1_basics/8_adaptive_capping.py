"""Adaptive Capping
# Flags: doc-Runnable

Adaptive capping is often used in optimization algorithms, particularly in
scenarios where the time taken to evaluate solutions can vary significantly.
For more details on adaptive capping, consult the [info page adaptive capping](../../advanced_usage/13_adaptive_capping.md).

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

    def train(self, config: Configuration, instance:str, cutoff, seed: int = 0) -> float:
        x0 = config["x0"]
        x1 = config["x1"]

        try:
            with timeout(int(math.ceil(cutoff))):
                runtime = 0.5 * x1 + 0.5 * x0 * int(instance)
                time.sleep(runtime)
                return runtime
        except TimeoutException as e:
            print(f"Timeout for configuration {config} with runtime cutoff {cutoff}")
            return cutoff + 1


if __name__ == '__main__':
    from smac import AlgorithmConfigurationFacade
    from smac import Scenario

    capped_problem = CappedProblem()

    scenario = Scenario(
        capped_problem.configspace,
        walltime_limit=3600,  # After 3600 seconds, we stop the hyperparameter optimization
        n_trials=500,  # Evaluate max 500 different trials
        instances=['1', '2', '3'],
        instance_features={'1': [1], '2': [2], '3': [3]},
        adaptive_capping=True,
        runtime_cutoff=200 # We allow an algorithm at maximum 200s to solve all instances
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = AlgorithmConfigurationFacade.get_initial_design(scenario, n_configs=5)

    # Create our SMAC object and pass the scenario and the train method
    smac = AlgorithmConfigurationFacade(
        scenario,
        capped_problem.train,
        initial_design=initial_design,
        overwrite=True,
    )

    # Let's optimize
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(capped_problem.configspace.get_default_configuration())

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
