from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Tuple

import os

from ConfigSpace import Configuration
from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.runhistory import TrialInfo, TrialInfoIntent, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class ParallelScheduler(AbstractIntensifier):
    """Common Racer class for Intensifiers that will schedule configurations on a parallel fashion.

    This class instantiates intensifier objects on a need basis, that is, to
    prevent workers from being idle. This intensifier objects will give configurations
    to run.

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
    min_budget : Optional[float]
        minimum budget allowed for 1 run of successive halving
    max_budget : Optional[float]
        maximum budget allowed for 1 run of successive halving
    eta : float
        'halving' factor after each iteration in a successive halving run. Defaults to 3
    n_initial_challengers : Optional[int]
        number of challengers to consider for the initial budget. If None, calculated internally
    n_seeds : Optional[int]
        Number of seeds to use, if TA is not deterministic. Defaults to None, i.e., seed is set as 0
    instance_order : Optional[str]
        how to order instances. Can be set to: [None, shuffle_once, shuffle]
        * None - use as is given by the user
        * shuffle_once - shuffle once and use across all SH run (default)
        * shuffle - shuffle before every SH run
    instance_seed_pairs : List[Tuple[str, int]], optional
        Do not set this argument, it will only be used by hyperband!
    min_challenger: int
        minimal number of challengers to be considered (even if time_bound is exhausted earlier). This class will
        raise an exception if a value larger than 1 is passed.
    incumbent_selection: str
        How to select incumbent in successive halving. Only active for real-valued budgets.
        Can be set to: [highest_executed_budget, highest_budget, any_budget]
        * highest_executed_budget - incumbent is the best in the highest budget run so far (default)
        * highest_budget - incumbent is selected only based on the highest budget
        * any_budget - incumbent is the best on any budget i.e., best performance regardless of budget
    """

    def __init__(
        self,
        scenario: Scenario,
        min_challenger: int = 1,
        intensify_percentage: float = 0.5,
        seed: int | None = None,
    ) -> None:

        super().__init__(
            scenario=scenario,
            min_challenger=min_challenger,
            intensify_percentage=intensify_percentage,
            seed=seed,
        )

        # We have a pool of instances that yield configurations ot run
        self.intensifier_instances: dict[int, AbstractIntensifier] = {}
        self.print_worker_warning = True

    @property
    def uses_seeds(self) -> bool:
        return True

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
        repeat_configs: bool = False,
        n_workers: int = 1,
    ) -> tuple[TrialInfoIntent, TrialInfo]:
        """This procedure decides from which instance to pick a config, in order to determine the
        next run.

        To prevent having idle workers, this procedure creates new instances
        up to the maximum number of workers available.

        If no new intensifier instance can be created and all intensifier
        objects need to wait for more data, this procedure sends a wait request to smbo.

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
        trial_info: RunInfo
            An object that encapsulates the minimum information to
            evaluate a configuration
        """
        if n_workers <= 1 and self.print_worker_warning:
            logger.warning(
                f"{self.__class__.__name__} is executed with {n_workers} worker(s) only. "
                f"However, your system supports up to {os.cpu_count()} workers. Consider increasing the workers "
                "in the scenario."
            )
            self.print_worker_warning = False

        # If repeat_configs is True, that means that not only self can repeat
        # configurations, but also in the context of multiprocessing, N
        # intensifier instances will also share configurations. The later
        # is not supported
        if repeat_configs:
            raise ValueError("repeat_configs is not supported for parallel execution")

        # First get a config to run from a SH instance
        for i in self._sort_instances_by_stage(self.intensifier_instances):
            intent, trial_info = self.intensifier_instances[i].get_next_run(
                challengers=challengers,
                incumbent=incumbent,
                get_next_configurations=get_next_configurations,
                runhistory=runhistory,
                repeat_configs=repeat_configs,
            )

            # if asked to wait, the intensifier cannot come up
            # with a new configuration, so we continue
            if intent == TrialInfoIntent.WAIT:
                continue

            return intent, trial_info

        # If gotten to this point, we might look into adding a new
        # intensifier
        if self._add_new_instance(n_workers):
            return self.intensifier_instances[len(self.intensifier_instances) - 1].get_next_run(
                challengers=challengers,
                incumbent=incumbent,
                get_next_configurations=get_next_configurations,
                runhistory=runhistory,
                repeat_configs=repeat_configs,
            )

        # If got to this point, no new instance can be added as
        # there are no idle workers and all running instances have to
        # wait, so we return a wait intent
        return TrialInfoIntent.WAIT, TrialInfo(
            config=None,
            instance=None,
            seed=0,
            budget=0.0,  # TODO: None?
        )

    def process_results(
        self,
        trial_info: TrialInfo,
        trial_value: TrialValue,
        incumbent: Configuration | None,
        runhistory: RunHistory,
        time_bound: float,
        log_trajectory: bool = True,
    ) -> tuple[Configuration, float]:
        """The intensifier stage will be updated based on the results/status of a configuration
        execution.

        To do so, this procedures redirects the result argument, to the
        respective intensifier object that generated the original config.

        Also, an incumbent will be determined. This determination is done
        using the complete run history, so we rely on the current intensifier
        choice of incumbent. That is, no need to go over each instance to
        get the incumbent, as there is no local runhistory

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
        incumbent: Configuration
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
        return self.intensifier_instances[trial_info.source].process_results(
            trial_info=trial_info,
            trial_value=trial_value,
            incumbent=incumbent,
            runhistory=runhistory,
            time_bound=time_bound,
            log_trajectory=log_trajectory,
        )

    def _add_new_instance(self, n_workers: int) -> bool:
        """Decides if it is possible to add a new intensifier instance, and adds it. If a new
        intensifier instance is added, True is returned, else False.

        Parameters
        ----------
        n_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
            Whether or not a successive halving instance was added
        """
        raise NotImplementedError()

    def _get_intensifier_ranking(self, intensifier: AbstractIntensifier) -> Tuple[int, int]:
        """Given a intensifier, returns how advance it is. This metric will be used to determine
        what priority to assign to the intensifier.

        Parameters
        ----------
        intensifier: AbstractRacer
            Intensifier to rank based on run progress

        Returns
        -------
        ranking: int
            the higher this number, the faster the intensifier will get
            the running resources. For hyperband we can use the
            sh_intensifier stage, for example
        tie_breaker: int
            The configurations that have been launched to break ties. For
            example, in the case of Successive Halving it can be the number
            of configurations launched
        """
        raise NotImplementedError()

    def _sort_instances_by_stage(self, instances: Dict[int, AbstractIntensifier]) -> List[int]:
        """This procedure dictates what SH to prioritize in launching jobs. It prioritizes resource
        allocation to SH instances that have higher stages. In case of tie, we prioritize the SH
        instance with more launched configs.

        Parameters
        ----------
        instances: Dict[int, AbstractRacer]
            Dict with the instances to prioritize

        Returns
        -------
        List:
            The order in which to query for new jobs
        """
        # This function might be called when no intensifier instances
        # exist (first iteration), so we return an empty list in that case
        if len(instances) == 0:
            return []

        # We want to prioritize runs that are close to finishing an iteration.
        # In the context of successive halving for example, an iteration has stages
        # that are composed of # of configs and each configs has # of instance-seed pairs
        # so ranking will be the stage (the higher the stage, the more we want this run
        # to be finished earlier). Also, in case of tie (runs at same stage) we need a
        # tie breaker, which can be the number of configs already launched
        preference = []
        for i, sh in instances.items():
            ranking, tie_breaker = self._get_intensifier_ranking(sh)
            preference.append(
                (i, ranking, tie_breaker),
            )

        # First we sort by config/instance/seed as the less important criteria
        preference.sort(key=lambda x: x[2], reverse=True)
        # Second by stage. The more advanced the stage is, the more we want
        # this intensifier to finish early
        preference.sort(key=lambda x: x[1], reverse=True)
        return [i for i, s, c in preference]
