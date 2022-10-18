from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Iterator, Optional

import time

import numpy as np
from ConfigSpace import Configuration

from smac.runhistory import TrialInfo, TrialInfoIntent, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.stats import Stats
from smac.utils.logging import format_array, get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class NewAbstractIntensifier:
    def __init__(self, scenario: Scenario):
        
        # Some caching; will be also set when runhistory is set
        self._used_configs = []
        
    
    @property
    def runhistory(self) -> RunHistory:
        return self._runhistory
    
    @runhistory.setter
    def runhistory(self, runhistory: RunHistory) -> None:
        # Validate runhistory
        # ...
        
        # Add all configs from runhistory to cache
        # ...
        
        # Set it global
        self._runhistory = runhistory
        
    
    @property
    @abstractmethod
    def uses_seeds(self) -> bool:
        """If the intensifier needs to make use of seeds."""
        raise NotImplementedError

    @property
    @abstractmethod
    def uses_budgets(self) -> bool:
        """If the intensifier needs to make use of budgets."""
        raise NotImplementedError

    @property
    @abstractmethod
    def uses_instances(self) -> bool:
        """If the intensifier needs to make use of instances."""
        raise NotImplementedError

    @abstractmethod
    def get_target_function_seeds(self) -> list[int]:
        """Which seeds are used to call the target function."""
        raise NotImplementedError

    @abstractmethod
    def get_target_function_budgets(self) -> list[float | None]:
        """Which budgets are used to call the target function."""
        raise NotImplementedError

    @abstractmethod
    def get_target_function_instances(self) -> list[str | None]:
        """Which instances are used to call the target function."""
        raise NotImplementedError

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }
    
    @abstractmethod
    def __next__(self) -> Iterator[TrialInfo | None]:
        """Main loop of the intensifier. This method always returns a TrialInfo object, although the intensifier
        algorithm may need to wait for the result of the trial."""
        raise NotImplementedError
        
    def next_configuration(self) -> Iterator[Configuration]:
        for config in self._initial_configs:
            yield config
            
        # ... more code
        # check if config was already used
            
        

    ##################

    def get_next_trial(
        self,
        challengers: list[Configuration] | None,
        incumbent: Configuration,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
        n_workers: int = 1,
    ) -> tuple[TrialInfoIntent, TrialInfo]:
        """Abstract method for choosing the next challenger. If no more challengers are available, the method should
        issue a SKIP via RunInfoIntent.SKIP, so that a new iteration can sample new configurations.

        Parameters
        ----------
        challengers : list[Configuration] | None
            Promising configurations.
        incumbent : Configuration
            Incumbent configuration.
        get_next_configurations : Callable[[], Iterator[Configuration]] | None, defaults to none
            Function that generates next configurations to use for racing.
        runhistory : RunHistory
        repeat_configs : bool, defaults to true
            If false, an evaluated configuration will not be generated again.
        n_workers : int, optional, defaults to 1
            The maximum number of workers available.

        Returns
        -------
        TrialInfoIntent
            Indicator of how to consume the TrialInfo object.
        TrialInfo
            An object that encapsulates necessary information of the trial.
        """
        raise NotImplementedError()

    def _next_challenger(
        self,
        challengers: list[Configuration] | None,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
    ) -> Configuration | None:
        """Retuns the next challenger to use in intensification. If challenger is none, then the
        optimizer will be used to generate the next challenger.

        Parameters
        ----------
        challengers : list[Configuration] | None
            Promising configurations to evaluate next.
        get_next_configurations : Callable[[], Iterator[Configuration]] | None, defaults to none
            Function that generates next configurations to use for racing.
        runhistory : RunHistory
        repeat_configs : bool, defaults to true
            If false, an evaluated configuration will not be generated again.

        Returns
        -------
        configuration : Configuration | None
            Next challenger to use. If no challenger was found, none is returned.
        """
        start_time = time.time()

        chall_gen: Iterator[Optional[Configuration]]
        if challengers:
            # iterate over challengers provided
            logger.debug("Using provied challengers.")
            chall_gen = (c for c in challengers)
        elif get_next_configurations:
            # generating challengers on-the-fly if optimizer is given
            logger.debug("Generating new challenger from optimizer.")
            chall_gen = get_next_configurations()
        else:
            raise ValueError("No configurations (function) provided. Can not generate challenger!")

        logger.debug("Time spend to select next challenger: %.4f" % (time.time() - start_time))

        # Select challenger from the generators
        assert chall_gen is not None
        for challenger in chall_gen:
            # Repetitions allowed
            if repeat_configs:
                return challenger

            used_configs = runhistory.get_configs()  # set(runhistory.get_configs())

            # Otherwise, select only a unique challenger
            if challenger not in used_configs:
                return challenger

        logger.debug("No valid challenger was generated!")
        return None

    def compare_configs(
        self,
        incumbent: Configuration,
        challenger: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> Configuration | None:
        """Compare two configuration wrt the runhistory and return the one which performs better (or
        None if the decision is not safe).

        Decision strategy to return x as being better than y:
        * x has at least as many trials as y.
        * x performs better than y on the intersection of trials on x and y.

        Note
        ----
        Implicit assumption: Challenger was evaluated on the same instance-seed pairs as incumbent.

        Returns
        -------
        configuration : Configuration | None
            The better configuration. If the decision is not sure, none is returned.
        """
        inc_trials = runhistory.get_trials(incumbent, only_max_observed_budget=True)
        chall_trials = runhistory.get_trials(challenger, only_max_observed_budget=True)
        to_compare_trials = set(inc_trials).intersection(chall_trials)

        # Performance on challenger trials, the challenger only becomes incumbent if it dominates the incumbent
        chal_perf = runhistory.average_cost(challenger, to_compare_trials, normalize=True)
        inc_perf = runhistory.average_cost(incumbent, to_compare_trials, normalize=True)

        assert type(chal_perf) == float
        assert type(inc_perf) == float

        # Line 15
        if np.any(chal_perf > inc_perf) and len(chall_trials) >= self._min_config_calls:
            chal_perf_format = format_array(chal_perf)
            inc_perf_format = format_array(inc_perf)

            # Incumbent beats challenger
            logger.debug(
                f"Incumbent ({inc_perf_format}) is better than challenger "
                f"({chal_perf_format}) on {len(chall_trials)} trials."
            )

            return incumbent

        # Line 16
        # This statement is true if both incumbent trials and challenger trials are the same
        if not set(inc_trials) - set(chall_trials):
            # No plateau walks
            if np.any(chal_perf >= inc_perf):
                chal_perf_format = format_array(chal_perf)
                inc_perf_format = format_array(inc_perf)

                logger.debug(
                    f"Incumbent ({inc_perf_format}) is at least as good as the "
                    f"challenger ({chal_perf_format}) on {len(chall_trials)} trials."
                )

                if log_trajectory and self._stats.incumbent_changed == 0:
                    assert self._stats
                    self._stats.add_incumbent(cost=chal_perf, incumbent=incumbent)

                return incumbent

            # Challenger is better than incumbent and has at least the same trials as incumbent.
            # -> Change incumbent
            n_samples = len(chall_trials)
            chal_perf_format = format_array(chal_perf)
            inc_perf_format = format_array(inc_perf)

            logger.info(
                f"Challenger ({chal_perf_format}) is better than incumbent ({inc_perf_format}) "
                f"on {n_samples} trials."
            )
            self.print_incumbent_changes(incumbent, challenger)

            if log_trajectory:
                assert self._stats
                self._stats.add_incumbent(cost=chal_perf, incumbent=challenger)

            return challenger

        # Undecided
        return None

    def print_incumbent_changes(
        self,
        incumbent: Configuration | None,
        challenger: Configuration | None,
    ) -> None:
        if incumbent is None or challenger is None:
            return

        params = sorted([(param, incumbent[param], challenger[param]) for param in challenger.keys()])
        logger.info("Changes in incumbent:")
        for param in params:
            if param[1] != param[2]:
                logger.info("--- %s: %r -> %r" % param)
            else:
                logger.debug("--- %s remains unchanged: %r", param[0], param[1])
