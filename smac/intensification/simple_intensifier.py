from __future__ import annotations

from typing import Any, Callable, Iterator

from smac.configspace import Configuration
from smac.constants import MAXINT
from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.runhistory import RunInfo, RunValue, RunInfoIntent, RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class SimpleIntensifier(AbstractIntensifier):
    """Performs the traditional Bayesian Optimization loop, without instance/seed intensification.

    Parameters
    ----------
    stats: smac.stats.stats.Stats
        stats object
    rng : np.random.RandomState
    instances : List[str]
        list of all instance ids
    instance_specifics : Mapping[str, str]
        mapping from instance name to instance specific string
    algorithm_walltime_limit : Optional[int]
        algorithm_walltime_limit of TA runs
    deterministic : bool
        whether the TA is deterministic or not
    """

    def __init__(
        self,
        scenario: Scenario,
        seed: int | None = None,
    ) -> None:

        super().__init__(
            scenario=scenario,
            min_challenger=1,
            seed=seed,
        )

        # We want to control the number of runs that are sent to
        # the workers. At any time, we want to make sure that if there
        # are just W workers, there should be at max W active runs
        # Below variable tracks active runs not processed
        self.run_tracker: dict[tuple[Configuration, str | None, int, float], bool] = {}

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def process_results(
        self,
        run_info: RunInfo,
        run_value: RunValue,
        incumbent: Configuration | None,
        runhistory: RunHistory,
        time_bound: float,
        log_trajectory: bool = True,
    ) -> tuple[Configuration, float]:
        """The intensifier stage will be updated based on the results/status of a configuration
        execution. Also, a incumbent will be determined.

        Parameters
        ----------
        run_info : RunInfo
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
        # Mark the fact that we processed this configuration
        self.run_tracker[(run_info.config, run_info.instance, run_info.seed, run_info.budget)] = True

        # If The incumbent is None we use the challenger
        if not incumbent:
            logger.info("First run, no incumbent provided; challenger is assumed to be the incumbent")
            incumbent = run_info.config

        self.num_run += 1

        incumbent = self._compare_configs(
            challenger=run_info.config,
            incumbent=incumbent,
            runhistory=runhistory,
            log_trajectory=log_trajectory,
        )
        # get incumbent cost
        inc_perf = runhistory.get_cost(incumbent)

        return incumbent, inc_perf

    def get_next_run(
        self,
        challengers: list[Configuration] | None,
        incumbent: Configuration,
        ask: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
        n_workers: int = 1,
    ) -> tuple[RunInfoIntent, RunInfo]:
        """Selects which challenger to be used. As in a traditional BO loop, we sample from the EPM,
        which is the next configuration based on the acquisition function. The input data is read
        from the runhistory.

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
        intent: RunInfoIntent
               Indicator of how to consume the RunInfo object
        run_info: RunInfo
            An object that encapsulates the minimum information to
            evaluate a configuration
        """

        # We always sample from the configs provided or the EPM
        challenger = self._next_challenger(
            challengers=challengers,
            ask=ask,
            runhistory=runhistory,
            repeat_configs=repeat_configs,
        )

        # Run tracker is a dictionary whose values indicate if a run has been
        # processed. If a value in this dict is false, it means that a worker
        # should still be processing this configuration.
        total_active_runs = len([v for v in self.run_tracker.values() if not v])
        if total_active_runs >= n_workers:
            # We only submit jobs if there is an idle worker
            return RunInfoIntent.WAIT, RunInfo(
                config=None,
                instance=None,
                seed=0,
                budget=0.0,
            )

        run_info = RunInfo(
            config=challenger,
            instance=None if self.instances is None else self.instances[-1],
            seed=0 if self.deterministic else int(self.rng.randint(low=0, high=MAXINT, size=1)[0]),
            budget=0.0,
        )

        self.run_tracker[(run_info.config, run_info.instance, run_info.seed, run_info.budget)] = False
        return RunInfoIntent.RUN, run_info
