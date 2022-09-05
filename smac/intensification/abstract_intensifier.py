from __future__ import annotations
from abc import abstractmethod

from typing import Any, Callable, Iterator, Optional, Tuple

import time

import numpy as np

from ConfigSpace import Configuration
from smac.runhistory import TrialInfo, TrialInfoIntent, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import format_array, get_logger
from smac.stats import Stats

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractIntensifier:
    """Base class for all racing methods.

    The "intensification" is designed to output a RunInfo object with enough information
    to run a given configuration (for example, the run info contains the instance/seed
    pair, as well as the associated resources).

    A worker can execute this RunInfo object and produce a RunValue object with the
    execution results. Each intensifier process the RunValue object and updates it's
    internal state in preparation for the next iteration.

    Note
    ----
    Do not use  this class directly.

    Parameters
    ----------
    stats: Stats
        stats object
    rng : np.random.RandomState
    instances : List[str]
        list of all instance ids
    instance_specifics : Mapping[str, str]
        mapping from instance name to instance specific string
    algorithm_walltime_limit : float
        runtime algorithm_walltime_limit of TA runs
    deterministic: bool
        whether the TA is deterministic or not
    min_config_calls : int
        Minimum number of run per config (summed over all calls to
        intensify).
    max_config_calls : int
        Maximum number of runs per config (summed over all calls to
        intensifiy).
    min_challenger: int
        minimal number of challengers to be considered (even if time_bound is exhausted earlier)
    intensify_percentage : float, defaults to 0.5
        How much percentage of the time should configurations be intensified (evaluated on higher budgets or
        more instances).
    """

    def __init__(
        self,
        scenario: Scenario,
        min_config_calls: int = 1,
        max_config_calls: int = 2000,
        min_challenger: int = 1,
        intensify_percentage: float = 0.5,
        seed: int | None = None,
    ):
        self.scenario = scenario
        self.stats: Stats | None = None

        if seed is None:
            seed = scenario.seed

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.deterministic = scenario.deterministic
        self.min_config_calls = min_config_calls
        self.max_config_calls = max_config_calls
        self.min_challenger = min_challenger
        self.intensify_percentage = intensify_percentage

        # Intensify percentage must be between 0 and 1
        assert intensify_percentage >= 0.0 and intensify_percentage <= 1.0

        # Set the instances
        self.instances: list[str | None]
        if scenario.instances is None:
            # We need to include None here to tell whether None instance was evaluated or not
            self.instances = [None]
        else:
            # Removing duplicates here
            # Fun fact: When using a set here, it always includes randomness
            self.instances = list(dict.fromkeys(scenario.instances))

        # General attributes
        self.num_run = 0  # Number of runs done in an iteration so far
        self._challenger_id = 0
        self._target_algorithm_time = 0.0

        # attributes for sampling next configuration
        # Repeating configurations is discouraged for parallel runs
        self.repeat_configs = False
        # to mark the end of an iteration
        self.iteration_done = False

    @property
    @abstractmethod
    def uses_seeds(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def uses_budgets(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def uses_instances(self) -> bool:
        raise NotImplementedError

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def get_next_run(
        self,
        challengers: list[Configuration] | None,
        incumbent: Configuration,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
        n_workers: int = 1,
    ) -> tuple[TrialInfoIntent, TrialInfo]:
        """Abstract method for choosing the next challenger, to allow for different selections
        across intensifiers uses ``_next_challenger()`` by default.

        If no more challengers are available, the method should issue a SKIP via
        RunInfoIntent.SKIP, so that a new iteration can sample new configurations.

        Parameters
        ----------
        challengers : List[Configuration]
            promising configurations
        incumbent: Configuration
             incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        runhistory : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again
        n_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        trial_info: RunInfo
            An object that encapsulates necessary information for a config run
        intent: RunInfoIntent
            Indicator of how to consume the RunInfo object
        """
        raise NotImplementedError()

    def process_results(
        self,
        trial_info: TrialInfo,
        trial_value: TrialValue,
        incumbent: Configuration | None,
        runhistory: RunHistory,
        time_bound: float,
        log_trajectory: bool = True,
    ) -> Tuple[Configuration, float]:
        """The intensifier stage will be updated based on the results/status of a configuration
        execution. Also, a incumbent will be determined.

        Parameters
        ----------
        trial_info : RunInfo
               A RunInfo containing the configuration that was evaluated
        incumbent : Optional[Configuration]
            Best configuration seen so far
        runhistory : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float
            time in [sec] available to perform intensify
        result: RunValue
            Contain the result (status and other methadata) of exercising
            a challenger/incumbent.
        log_trajectory: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
        raise NotImplementedError()

    def _next_challenger(
        self,
        challengers: list[Configuration] | None,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
    ) -> Configuration | None:
        """Retuns the next challenger to use in intensification If challenger is None, then
        optimizer will be used to generate the next challenger.

        Parameters
        ----------
        challengers : List[Configuration]
            promising configurations to evaluate next
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            a sampler that generates next configurations to use for racing
        runhistory : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again

        Returns
        -------
        Configuration
            next challenger to use
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
            raise ValueError("No configurations/ask function provided. Can not generate challenger!")

        logger.debug("Time to select next challenger: %.4f" % (time.time() - start_time))

        # Select challenger from the generators
        assert chall_gen is not None
        for challenger in chall_gen:
            # Repetitions allowed
            if repeat_configs:
                return challenger

            used_configs = set(runhistory.get_configs())

            # Otherwise, select only a unique challenger
            if challenger not in used_configs:
                return challenger

        logger.debug("No valid challenger was generated!")
        return None

    def _compare_configs(
        self,
        incumbent: Configuration,
        challenger: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> Configuration | None:
        """Compare two configuration wrt the runhistory and return the one which performs better (or
        None if the decision is not safe)

        Decision strategy to return x as being better than y:
            1. x has at least as many runs as y
            2. x performs better than y on the intersection of runs on x and y

        Implicit assumption:
            Challenger was evaluated on the same instance-seed pairs as
            incumbent

        Parameters
        ----------
        incumbent: Configuration
            Current incumbent
        challenger: Configuration
            Challenger configuration
        runhistory: smac.runhistory.runhistory.RunHistory
            Stores all runs we ran so far
        log_trajectory: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        None or better of the two configurations x,y
        """
        inc_runs = runhistory.get_runs_for_config(incumbent, only_max_observed_budget=True)
        chall_runs = runhistory.get_runs_for_config(challenger, only_max_observed_budget=True)
        to_compare_runs = set(inc_runs).intersection(chall_runs)

        # performance on challenger runs, the challenger only becomes incumbent
        # if it dominates the incumbent
        chal_perf = runhistory.average_cost(challenger, to_compare_runs, normalize=True)
        inc_perf = runhistory.average_cost(incumbent, to_compare_runs, normalize=True)

        assert type(chal_perf) == float
        assert type(inc_perf) == float

        # Line 15
        if np.any(chal_perf > inc_perf) and len(chall_runs) >= self.min_config_calls:
            chal_perf_format = format_array(chal_perf)
            inc_perf_format = format_array(inc_perf)
            # Incumbent beats challenger
            logger.debug(
                f"Incumbent ({inc_perf_format}) is better than challenger "
                f"({chal_perf_format}) on {len(chall_runs)} runs."
            )
            return incumbent

        # Line 16
        if not set(inc_runs) - set(chall_runs):
            # no plateau walks
            if np.any(chal_perf >= inc_perf):
                chal_perf_format = format_array(chal_perf)
                inc_perf_format = format_array(inc_perf)

                logger.debug(
                    f"Incumbent ({inc_perf_format}) is at least as good as the "
                    f"challenger ({chal_perf_format}) on {len(chall_runs)} runs."
                )
                assert self.stats
                if log_trajectory and self.stats.incumbent_changed == 0:
                    self.stats.add_incumbent(cost=chal_perf, incumbent=incumbent)

                return incumbent

            # Challenger is better than incumbent
            # and has at least the same runs as inc
            # -> change incumbent
            n_samples = len(chall_runs)
            chal_perf_format = format_array(chal_perf)
            inc_perf_format = format_array(inc_perf)

            logger.info(
                f"Challenger ({chal_perf_format}) is better than incumbent ({inc_perf_format}) " f"on {n_samples} runs."
            )
            self._log_incumbent_changes(incumbent, challenger)

            if log_trajectory:
                assert self.stats
                self.stats.add_incumbent(cost=chal_perf, incumbent=challenger)

            return challenger

        # undecided
        return None

    def _log_incumbent_changes(
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
