import logging
import typing
import warnings

import numpy as np

from smac.intensification.abstract_racer import AbstractRacer, RunInfoIntent
from smac.intensification.successive_halving import SuccessiveHalving
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.stats.stats import Stats
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.utils.io.traj_logging import TrajLogger


class ParallelSuccessiveHalving(AbstractRacer):

    """Races multiple challengers against an incumbent using Successive Halving method,
    in a parallel fashion

    This class instantiates SuccessiveHalving objects on a need basis, that is, to
    prevent workers from being idle.

    Parameters
    ----------
    stats: smac.stats.stats.Stats
        stats object
    traj_logger: smac.utils.io.traj_logging.TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : typing.Optional[int]
        cutoff of TA runs
    deterministic : bool
        whether the TA is deterministic or not
    initial_budget : typing.Optional[float]
        minimum budget allowed for 1 run of successive halving
    max_budget : typing.Optional[float]
        maximum budget allowed for 1 run of successive halving
    eta : float
        'halving' factor after each iteration in a successive halving run. Defaults to 3
    num_initial_challengers : typing.Optional[int]
        number of challengers to consider for the initial budget. If None, calculated internally
    run_obj_time : bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    n_seeds : typing.Optional[int]
        Number of seeds to use, if TA is not deterministic. Defaults to None, i.e., seed is set as 0
    instance_order : typing.Optional[str]
        how to order instances. Can be set to: [None, shuffle_once, shuffle]
        * None - use as is given by the user
        * shuffle_once - shuffle once and use across all SH run (default)
        * shuffle - shuffle before every SH run
    adaptive_capping_slackfactor : float
        slack factor of adpative capping (factor * adaptive cutoff)
    inst_seed_pairs : typing.List[typing.Tuple[str, int]], optional
        Do not set this argument, it will only be used by hyperband!
    min_chall: int
        minimal number of challengers to be considered (even if time_bound is exhausted earlier). This class will
        raise an exception if a value larger than 1 is passed.
    incumbent_selection: str
        How to select incumbent in successive halving. Only active for real-valued budgets.
        Can be set to: [highest_executed_budget, highest_budget, any_budget]
        * highest_executed_budget - incumbent is the best in the highest budget run so far (default)
        * highest_budget - incumbent is selected only based on the highest budget
        * any_budget - incumbent is the best on any budget i.e., best performance regardless of budget
    """

    def __init__(self,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: typing.Optional[float] = None,
                 deterministic: bool = False,
                 initial_budget: typing.Optional[float] = None,
                 max_budget: typing.Optional[float] = None,
                 eta: float = 3,
                 num_initial_challengers: typing.Optional[int] = None,
                 run_obj_time: bool = True,
                 n_seeds: typing.Optional[int] = None,
                 instance_order: typing.Optional[str] = 'shuffle_once',
                 adaptive_capping_slackfactor: float = 1.2,
                 inst_seed_pairs: typing.Optional[typing.List[typing.Tuple[str, int]]] = None,
                 min_chall: int = 1,
                 incumbent_selection: str = 'highest_executed_budget',
                 ) -> None:

        super().__init__(stats=stats,
                         traj_logger=traj_logger,
                         rng=rng,
                         instances=instances,
                         instance_specifics=instance_specifics,
                         cutoff=cutoff,
                         deterministic=deterministic,
                         run_obj_time=run_obj_time,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor,
                         min_chall=min_chall)

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # Successive Halving Hyperparameters
        self.n_seeds = n_seeds
        self.instance_order = instance_order
        self.inst_seed_pairs = inst_seed_pairs
        self.incumbent_selection = incumbent_selection
        self._instances = instances
        self._instance_specifics = instance_specifics
        self.initial_budget = initial_budget
        self.max_budget = max_budget
        self.eta = eta
        self.num_initial_challengers = num_initial_challengers

        # We have a pool of successive halving instances that yield work
        # Add sh number 0, because we are likely gonna need it
        self.sh_instances = {}  # type: typing.Dict[int, SuccessiveHalving]

        # We will have multiple SH. Below variable tracks which is the last
        # instance that launched a job
        self.last_active_instance = 0

    def process_results(self,
                        run_info: RunInfo,
                        incumbent: typing.Optional[Configuration],
                        run_history: RunHistory,
                        time_bound: float,
                        result: RunValue,
                        log_traj: bool = True,
                        ) -> \
            typing.Tuple[Configuration, float]:
        """
        The intensifier stage will be updated based on the results/status
        of a configuration execution.

        To do so, this procedures redirects the result argument, to the
        respective SuccessiveHalving object that generated the original config.

        Also, an incumbent will be determined. This determination is done
        using the complete run history, so we rely on the current SH
        choice of incumbent. That is, not need to go over each SH to
        get the incumbent, as there is no local runhistory

        Parameters
        ----------
        run_info : RunInfo
            A RunInfo containing the configuration that was evaluated
        incumbent : typing.Optional[Configuration]
            Best configuration seen so far
        run_history : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float
            time in [sec] available to perform intensify
        result: RunValue
            Contain the result (status and other methadata) of exercising
            a challenger/incumbent.
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """

        return self.sh_instances[run_info.source_id].process_results(
            run_info=run_info,
            incumbent=incumbent,
            run_history=run_history,
            time_bound=time_bound,
            result=result,
            log_traj=log_traj,
        )

    def get_next_run(self,
                     challengers: typing.Optional[typing.List[Configuration]],
                     incumbent: Configuration,
                     chooser: typing.Optional[EPMChooser],
                     run_history: RunHistory,
                     repeat_configs: bool = False,
                     num_workers: int = 1,
                     ) -> typing.Tuple[RunInfoIntent, RunInfo]:
        """
        This procedure decides from which successive halver instance to pick a config,
        in order to determine the next run. For details on how the configuration
        got selected, please check get_next_run of SuccesiveHalving.

        To prevent having idle workers, this procedure creates successive halvers
        up to the maximum number of workers available.

        If no new SH can be created and all SH objects need to wait for more data,
        this procedure sends a wait request to smbo.

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent: Configuration
            incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again
        num_workers: int
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

        if num_workers <= 1:
            warnings.warn("ParallelSuccesiveHalving is intended to be used "
                          "with more than 1 worker but num_workers={}".format(
                              num_workers
                          ))

        # If repeat_configs is True, that means that not only self can repeat
        # configurations, but also in the context of multiprocessing, N
        # successive halving instance will also share configurations. The later
        # is not supported
        if repeat_configs:
            raise ValueError(
                "repeat_configs==True is not supported for parallel successive halving")

        # First get a config to run from a SH instance
        for i in self._sh_scheduling():
            intent, run_info = self.sh_instances[i].get_next_run(
                challengers=challengers,
                incumbent=incumbent,
                chooser=chooser,
                run_history=run_history,
                repeat_configs=repeat_configs,
            )

            # if asked to wait, the successive halver cannot come up
            # with a new configuration, so we continue
            if intent == RunInfoIntent.WAIT:
                continue

            # The last instance that sent a job, for scheduling purposes
            self.last_active_instance = i

            return intent, run_info

        # If gotten to this point, we might look into adding a new SH
        if self._add_new_SH(num_workers):
            return self.sh_instances[len(self.sh_instances) - 1].get_next_run(
                challengers=challengers,
                incumbent=incumbent,
                chooser=chooser,
                run_history=run_history,
                repeat_configs=repeat_configs,
            )

        # If got to this point, there is no new SH addition viable
        # not a SH has something to launch, so we have to wait
        return RunInfoIntent.WAIT, RunInfo(
            config=None,
            instance="0",
            instance_specific="0",
            seed=0,
            cutoff=self.cutoff,
            capped=False,
            budget=0.0,
        )

    def _add_new_SH(self, num_workers: int) -> bool:
        """
        Decides if it is possible to add a new SH, and adds it.
        If a new SH is added, True is returned, else False.

        Parameters:
        -----------
        num_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
            Whether or not a successive halving instance was added
        """
        if len(self.sh_instances) >= num_workers:
            return False

        self.sh_instances[len(self.sh_instances)] = SuccessiveHalving(
            stats=self.stats,
            traj_logger=self.traj_logger,
            rng=self.rs,
            instances=self._instances,
            instance_specifics=self._instance_specifics,
            cutoff=self.cutoff,
            deterministic=self.deterministic,
            initial_budget=self.initial_budget,
            max_budget=self.max_budget,
            eta=self.eta,
            num_initial_challengers=self.num_initial_challengers,
            run_obj_time=self.run_obj_time,
            n_seeds=self.n_seeds,
            instance_order=self.instance_order,
            adaptive_capping_slackfactor=self.adaptive_capping_slackfactor,
            inst_seed_pairs=self.inst_seed_pairs,
            min_chall=self.min_chall,
            incumbent_selection=self.incumbent_selection,
            identifier=len(self.sh_instances),
        )

        return True

    def _sh_scheduling(self, strategy: str = 'more_advanced_iteration') -> typing.List[int]:
        """
        This procedure dictates what SH to prioritize in
        launching jobs.

        Parameters
        ----------
            strategy: str
                What type of scheduling to follow

        Returns
        -------
            List:
                The order in which to query for new jobs

        """

        # This function might be called when no SuccessiveHalving instances
        # exist (first iteration), so we return an empty list in that case
        if len(self.sh_instances) == 0:
            return []

        expected = self.last_active_instance + 1
        if strategy == 'ordinal' or expected == len(self.sh_instances):
            # This is hpbanster strategy, just pick on an ordered fashion
            return list(range(len(self.sh_instances)))
        elif strategy == 'serial':
            # All SH instances are copies launching the same type of
            # of configurations. This is a cheap way of doing FIFO
            # if last index was 3, we will expect to go to 4 onwards like:
            # --> next schedule: 4->5->0->1->2->3 for 6 SH
            # in above example, if last scheduler was 5, it is like ordinal
            # scheduling
            return list(range(expected, len(self.sh_instances))) + list(range(expected))
        elif strategy == 'more_advanced_iteration':
            # We want to prioritize runs that are close to finishing an iteration.
            # In the context of successive halving, an iteration has stages that are
            # composed of # of configs and each configs has # of instance-seed pairs
            preference = []
            for i, sh in self.sh_instances.items():
                # Each row of this matrix is id, stage, configs+instances for stage
                # We use sh.run_tracker as a cheap way to know how advanced the run is
                # in case of stage ties among successive halvers. sh.run_tracker is
                # also emptied each iteration
                stage = 0
                if hasattr(sh, 'stage'):
                    # Newly created SuccessiveHalving objects have no stage
                    stage = sh.stage
                preference.append(
                    (i, stage, len(sh.run_tracker))
                )

            # First we sort by config/instance/seed as the less important criteria
            preference.sort(key=lambda x: x[2], reverse=True)
            # Second by stage. The more advanced the stage is, the more we want
            # this SH to finish early
            preference.sort(key=lambda x: x[1], reverse=True)
            return [i for i, s, c in preference]
        else:
            raise ValueError("Unsupported strategy " + strategy)
