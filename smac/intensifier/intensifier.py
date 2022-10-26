from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, cast

from collections import Counter

from ConfigSpace import Configuration

from smac.constants import MAXINT
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.intensifier.stages import IntensifierStage
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

    def __next__(self) -> Iterator[TrialInfo]:
        rh = self.runhistory
        
        while True:
            incumbent = rh.get_incumbent()
            # incumbent_trials = rh.get_trials(incumbent)
            
            for trial in self._get_missing_trials(incumbent):
                yield trial
            
            # if R (== config_calls) contains less than maxR (== max_config_calls) runs with incumbent
            #for i in range(self._max_config_calls - len(incumbent_trials)):
            #    new_instance, new_seed = self._get_next_instance_seed(i)
                
            #    # TODO: Make sure we cache the new instance/seed
                
            #    yield TrialInfo(incumbent, new_instance, new_seed)
            
            # Start iteration with next config
            N = 1
            config = next(self.config_selector)
            while True:
                initial_missing_trials = self._get_missing_trials(incumbent, config, N)
                for trial in initial_missing_trials:
                    yield trial
                    
                # Now we basically have to wait till all missing trials are available
                # If we wait but the user requests a new trial, we have to grab a new config
                while True:
                    missing_trials = self._get_missing_trials(incumbent, config, N)
                    if len(missing_trials) == 0:
                        break
                    
                    # Now we grab a new config and yield a new trial info with the first instance and first seed#
                    # TODO: Intensify?
                    # Maybe we just evaluate a random trial?
                    # But which 
                    
                    yield TrialInfo(next(self.config_selector), self._tf_instances[0], self._tf_seeds[0])
                    
                # Now we have all trials evaluated
                if rh.get_cost(incumbent, initial_missing_trials) > rh.get_cost(config, initial_missing_trials):
                    # New config is worse than incumbent
                    break
                elif len(initial_missing_trials) == 0:
                    # New configuration is the new incumbent
                    # However, since the incumbent is evaluated differently, we skip it here
                    pass
                else:
                    N = 2 * N
                    
    def _get_missing_trials(self, config: Configuration, N: int|None=None) -> list[TrialInfo]:
        """Returns unevaluated trials for the passed configuration. Returns ``max_config_calls`` trials if ``N`` is 
        None. Prioritizes instances/seeds found in the runhistory. Seed change happens after all instances are
        added first.
        """
        trials = []
        
        # TODO: Respect deterministic
        for seed in self._tf_seeds:
            for instance in self._tf_instances:
                
                
        
        
        
        
        
    
                    
                
                