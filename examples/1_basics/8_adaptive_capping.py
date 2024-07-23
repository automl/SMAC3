"""
Adaptive Capping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adaptive capping is often used in optimization algorithms, particularly in
scenarios where the time taken to evaluate solutions can vary significantly.

"""

import time
from ConfigSpace import ConfigurationSpace, Configuration, Float
import signal
from contextlib import contextmanager

from smac.runhistory import InstanceSeedBudgetKey, TrialInfo
from smac.runner import AbstractRunner


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

    def train(self, config: Configuration, budget, seed: int = 0) -> float:
        x0 = config["x0"]
        x1 = config["x1"]

        try:
            with timeout(budget):
                runtime = 1.5 * x1 + x0
                time.sleep(runtime)
                return runtime
        except TimeoutException as e:
            print(f"Timeout for configuration {config} with budget {budget}")
            return budget  # FIXME: what should be returned here?


if __name__ == '__main__':

    from smac import HyperparameterOptimizationFacade
    from smac import Scenario

    from smac.intensifier import Intensifier

    capped_problem = CappedProblem()

    scenario = Scenario(
        capped_problem.configspace,
        walltime_limit=200,  # After 200 seconds, we stop the hyperparameter optimization
        n_trials=500,  # Evaluate max 500 different trials
        # FIXME: we will want to have an active capping option, that renders the budgets useless?

    )

    # We want to run five random configurations before starting the optimization.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)


    class CappedIntensifier(Intensifier):
        def __init__(self, scenario, runtime_cutoff: int | None = None, *args, **kwargs):
            """
            Intensifier that caps the runtime of a configuration to a given value.

            Parameters
            ----------
            scenario : Scenario
                Scenario object
            runtime_cutoff : int, defaults to None: Initial runtime budget in seconds for adaptive capping.
                A non-None value will trigger adaptive capping and require the target algorithm to accept
                a budget argument that is the number of seconds to maximally run the target algorithm.
            """
            super().__init__(scenario, *args, **kwargs)
            self.runtime_cutoff = runtime_cutoff

        def _get_next_trials(
                self,
                config: Configuration,
                *,
                N: int | None = None,
                from_keys: list[InstanceSeedBudgetKey] | None = None,
                shuffle: bool = True,
        ) -> list[TrialInfo]:
            trials = super()._get_next_trials(config, N=N, from_keys=from_keys, shuffle=shuffle)
            if self.runtime_cutoff is not None:
                # We need to adapt the budget to the runtime cutoff
                trials = [
                    TrialInfo(config=t.config, seed=t.seed, budget=self.runtime_cutoff)
                    for t in trials
                ]

            return trials

        def uses_budgets(self) -> bool:
            return True


    # Create our intensifier
    intensifier = CappedIntensifier(scenario, runtime_cutoff=15)

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

    # An example of the problem, using analoge random sampling:
    # capped_problem = CappedProblem()
    # cs = capped_problem.configspace
    #
    # config = cs.sample_configuration()
    #
    # next_runtime = 99999 # some very high number
    # while next_runtime > 2 :
    #     config = cs.sample_configuration()
    #     try: # TODO move try except to train method!
    #         runtime = capped_problem.train(config, int(next_runtime))
    #         next_runtime = runtime
    #         print(f"incumbent: {config}, with runtime {config['x0'] + 1.5 * config['x1']}")
    #     except TimeoutException as e:
    #         print(f'failed config: {config} with budget {next_runtime}, evaluates to a runtime of {config["x0"] + 1.5 * config["x1"]}')
    #
    #
    # print(f"Finished with a budget of {next_runtime} seconds")
