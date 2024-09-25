"""
Adaptive Capping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adaptive capping is often used in optimization algorithms, particularly in
scenarios where the time taken to evaluate solutions can vary significantly.

"""
import math
import time
import warnings
from typing import List

from ConfigSpace import ConfigurationSpace, Configuration, Float
import signal
from contextlib import contextmanager

from smac.runhistory import InstanceSeedBudgetKey, TrialInfo
import itertools
import warnings

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score



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

    # FIXME AC facade instead of HPO
    from smac import HyperparameterOptimizationFacade, RunHistory
    from smac import Scenario

    from smac.intensifier import Intensifier

    capped_problem = CappedProblem()

    scenario = Scenario(
        capped_problem.configspace,
        walltime_limit=200,  # After 200 seconds, we stop the hyperparameter optimization
        n_trials=500,  # Evaluate max 500 different trials
        instances=['1', '2', '3'],
        instance_features={'1': [1], '2': [2], '3': [3]}
    )

    # We want to run five random configurations before starting the optimization.
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)


    class CappedIntensifier(Intensifier):
        def __init__(self, scenario, runtime_cutoff: int | None = None,
                     adaptive_capping_slackfactor: float = 1.2,
                     *args, **kwargs):
            """
            Intensifier that caps the runtime of a configuration to a given value.

            Parameters
            ----------
            scenario : Scenario
                Scenario object
            runtime_cutoff : int, defaults to None: Initial runtime budget in seconds for.
            adaptive capping.
                A non-None value will trigger adaptive capping and require the target algorithm to accept
                a budget argument that is the number of seconds to maximally run the target algorithm.
            """
            super().__init__(scenario, *args, **kwargs)
            self.runtime_cutoff = runtime_cutoff
            self.adaptive_capping_slackfactor = adaptive_capping_slackfactor

        def get_trials_of_interest(
                self,
                config: Configuration,
                *,
                validate: bool = False,
                seed: int | None = None,
        ) -> list[TrialInfo]:
            """Returns the trials of interest for a given configuration.
            Expands the keys from ``get_instance_seed_keys_of_interest`` with the config.
            """
            is_keys = self.get_instance_seed_keys_of_interest(validate=validate, seed=seed)

            trials = []
            for key in is_keys:
                # FIXME: trialinfo needs budget key
                trials.append(TrialInfo(config=config, instance=key.instance, seed=key.seed))

            return trials

        def _get_next_trials(
                self,
                config: Configuration,
                *,
                N: int | None = None,
                from_keys: list[InstanceSeedBudgetKey] | None = None,
                shuffle: bool = True,
        ) -> list[TrialInfo]:
            trials = super()._get_next_trials(
                config,
                N=N,
                from_keys=from_keys,
                shuffle=shuffle
            )
            if self.runtime_cutoff is not None and bool(trials):
                # We need to adapt the budget to the runtime cutoff
                budgets = [self._get_adaptivecapping_budget(t.config) for t in trials]

                trials = [
                    TrialInfo(config=t.config, instance=t.instance, seed=t.seed, budget=b)
                    for b, t in zip(budgets, trials)
                ]

            return trials

        def uses_budgets(self) -> bool:
            return True

        def _get_adaptivecapping_budget(
                self,
                challenger: Configuration,

        ) -> float:
            """Adaptive capping: Compute cutoff based on time so far used for incumbent and reduce
            cutoff for next run of challenger accordingly.

            Warning:
            For concurrent runs, the budget will be determined for a challenger x instance
            combination at the moment the challenger is considered for the instance, ignorant of
            the runtime cost of the currently running instances of the same configuration.

            !Only applicable if self.run_obj_time

            !runs on incumbent should be superset of the runs performed for the
             challenger

            Parameters
            ----------
            challenger : Configuration
                Configuration which challenges incumbent

            inc_sum_cost: float
                Sum of runtimes of all incumbent runs

            Returns
            -------
            cutoff: float
                Adapted cutoff
            """

            # cost used by challenger for going over all its runs
            # should be subset of runs of incumbent (not checked for efficiency
            # reasons)
            incumbents = self.get_incumbents(sort_by="num_trials")
            if len(incumbents) == 0:
                return  self.runtime_cutoff

            if len(incumbents) > 1:
                warnings.warn("Adaptive capping is only supported for single incumbent scenarios")

            inc_sum_cost = self.runhistory.sum_cost(
                config=incumbents[0],
                instance_seed_budget_keys=None,
                normalize=True
            )

            # if len(challenger) == 0:
            #     # we only got the
            #     return

            # original logic for get_runs_for_config:
            # https://github.com/automl/SMAC3/blob/f1d2aa2ea3b6ad4075550af69e3300f19411a5ea/smac/runhistory/runhistory.py#L772
            # TODAY: runhistory.get_trials?
            chall_inst_seeds = self.runhistory.get_trials(
                challenger,
                highest_observed_budget_only=True
            )
            # fixme: for each challenger, we need to compute its total cost!
            #  and then we need to return a per config based budget!
            chal_sum_cost = self.runhistory.sum_cost(
                config=challenger, instance_seed_budget_keys=chall_inst_seeds, normalize=True
            )
            assert type(chal_sum_cost) == float

            cutoff = min(
                self.runtime_cutoff,
                inc_sum_cost * self.adaptive_capping_slackfactor - chal_sum_cost
            )

            return cutoff


    # Create our intensifier
    intensifier = CappedIntensifier(scenario, runtime_cutoff=10)

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
