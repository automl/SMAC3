from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, cast

from collections import Counter
import random

from ConfigSpace import Configuration

from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.runhistory import (
    InstanceSeedBudgetKey,
    TrialInfo,
    TrialInfoIntent,
    TrialValue,
)
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import format_array, get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class Intensifier(AbstractIntensifier):
    def __init__(
        self,
        scenario: Scenario,
        min_config_calls: int = 1,
        max_config_calls: int = 2000,
        min_challenger: int = 2,
        intensify_percentage: float = 0.5,
        race_against: Configuration | None = None,
        seed: int | None = None,
    ):
        if scenario.deterministic:
            if min_challenger != 1:
                logger.info("The number of minimal challengers is set to one for deterministic algorithms.")

            min_challenger = 1

        super().__init__(scenario=scenario)
        
        # Internal variables
        self._max_config_calls = max_config_calls
        self._intensify_percentage = intensify_percentage

    def __next__(self) -> Iterator[TrialInfo]:
        rh = self.runhistory
        old_incumbent: Configuration | None = None
        
        # TODO: How to handle this if we continue optimization run? Just start from scratch?
        rejected: list[Configuration] = []
        pending: list[tuple[Configuration, int]] = []
        
        while True:
            # We select a new incumbent, which is based on the average of instance/seed pairs
            # A new incumbent is chosen only if the new configuration is evaluated as on all instance/seed pairs
            # the incumbent has
            incumbent = rh.get_incumbent()
            
            # Clear the pending queue if incumbent changed
            if old_incumbent != incumbent:
                old_incumbent = incumbent
                pending = []
            
            # We get ``self._max_config_calls`` == maxR trials to evaluate for the incumbent
            # First the instances are "filled-up" before a new seed is started
            # We don't yield the trials if the trials are marked as running in the runhistory
            initial_incumbent_trials = self._get_missing_trials(incumbent, N=self._max_config_calls)
            for trial in initial_incumbent_trials:
                # Keep in mind: The generator keeps track of the state so the next time __next__ is called, 
                # we start directly after the yield again
                yield trial
    
            # Percentage parameter: We decide whether to intensify or to evaluate a fresh configuration
            if random.rand() < self._intensify_percentage:
                config = next(self.config_selector)
                N = 1
            else:
                # Continue pending runs with the latest N
                config = None
                if len(pending) > 0:
                    pending_config, pending_N = pending.pop()
                    
                    # Select config and N from the queue
                    config = pending_config
                    N = pending_N
                
                # If there are no pending configs, we select a [???] config from the runhistory?
                # Which configuration should be intensified?
                # - Random
                # - We choose the configuration with the second lowest cost after incumbent?
                # - We choose the configuration with the least trials?
                # Also, how do we select the N? If we select N == 1, then the configuration goes
                # into the pending queue and might be chosen the next iteration. Should work?
                # However, if the configuration is already rejected or the incumbent or else,
                # then the intensification in this iteration is basically skipped.
                if config is None:
                    config = rh.get_random_config()
                    N = 1

            # We don't want to evaluate a rejected configuration or an incumbent
            if config is None or config in rejected or config == incumbent:
                continue
            
            # Don't return missing trials if marked as ``RUNNING`` in the runhistory
            # Basically, trials which haven't been run yet
            # TODO: Remember to use the same random seed!
            initial_missing_trials = self._get_missing_trials(config, N)
            for trial in initial_missing_trials:
                yield trial
            
            # Trials which are evaluated already
            # TODO: Remember to use the same random seed as above (otherwise we get different trials)
            evaluated_trials = self._get_evaluated_trials(config, N)
            
            # We only go here if all trials have been evaluated
            if len(initial_missing_trials) == 0:
                # Now we have all trials evaluated and we can do a comparison
                if rh.get_cost(config, evaluated_trials) > rh.get_cost(incumbent, evaluated_trials):
                    # New config is worse than incumbent so we reject the configuration forever
                    rejected.append(config)
                # If we evaluated as much trials as we evaluated the incumbent
                elif len(evaluated_trials) == self._max_config_calls:
                    # New configuration is the new incumbent
                    # However, since the incumbent is evaluated in each iteration, we skip it here
                    pass
                else:
                    # In the original paper, we would double N: In our case, we mark it as pending so it could 
                    # be intensified in the next iteration.
                    pending.append((config, N * 2))
            # Trials have not been evaluated yet
            else:
                # We append the current N to the pending, so in the next iteration we check again
                # if the trials have been evaluated
                pending.append((config, N))
                
    def _get_missing_trials(self, config: Configuration, N: int | None=None) -> list[TrialInfo]:
        """Returns unevaluated trials for the passed configuration. Returns ``max_config_calls`` trials if ``N`` is 
        None. Prioritizes instances/seeds found in the runhistory. Seed change happens after all instances are
        added first.
        """
        trials = []
        
        # TODO: Respect deterministic
        for seed in self._tf_seeds:
            for instance in self._tf_instances:
                pass
                
                
    def _get_evaluated_trials(self):
        pass
        
        
        
        
        
    
                    
                
                