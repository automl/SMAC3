import time
import logging
import typing
from collections import Counter
from enum import Enum

import numpy as np

from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory, InstSeedBudgetKey
from smac.tae.execute_ta_run import BudgetExhaustedException, CappedRunException, ExecuteTARun
from smac.utils.io.traj_logging import TrajLogger
from smac.intensification.abstract_racer import AbstractRacer, _config_to_run_type
from smac.optimizer.epm_configuration_chooser import EPMChooser

__author__ = "Katharina Eggensperger, Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class IntensifierStage(Enum):
    """Class to define different stages of intensifier"""
    RUN_FIRST_CONFIG = 0  # to replicate the old initial design
    RUN_INCUMBENT = 1  # Lines 3-7
    RUN_CHALLENGER = 2  # Lines 8-17
    RUN_BASIS = 3


class Intensifier(AbstractRacer):
    """Races challengers against an incumbent (a.k.a. SMAC's intensification
    procedure).

    Parameters
    ----------
    tae_runner : tae.executre_ta_run_*.ExecuteTARun* Object
        target algorithm run executor
    stats: Stats
        stats object
    traj_logger: TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : int
        runtime cutoff of TA runs
    deterministic: bool
        whether the TA is deterministic or not
    run_obj_time: bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    always_race_against: Configuration
        if incumbent changes race this configuration always against new incumbent;
        can sometimes prevent over-tuning
    use_ta_time_bound: bool,
        if true, trust time reported by the target algorithms instead of
        measuring the wallclock time for limiting the time of intensification
    run_limit : int
        Maximum number of target algorithm runs per call to intensify.
    maxR : int
        Maximum number of runs per config (summed over all calls to
        intensifiy).
    minR : int
        Minimum number of run per config (summed over all calls to
        intensify).
    adaptive_capping_slackfactor: float
        slack factor of adpative capping (factor * adpative cutoff)
    min_chall: int
        minimal number of challengers to be considered (even if time_bound is exhausted earlier)
    """

    def __init__(self, tae_runner: ExecuteTARun,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: int = None,
                 deterministic: bool = False,
                 run_obj_time: bool = True,
                 always_race_against: Configuration = None,
                 run_limit: int = MAXINT,
                 use_ta_time_bound: bool = False,
                 minR: int = 1,
                 maxR: int = 2000,
                 adaptive_capping_slackfactor: float = 1.2,
                 min_chall: int = 2,):

        super().__init__(tae_runner=tae_runner,
                         stats=stats,
                         traj_logger=traj_logger,
                         rng=rng,
                         instances=instances,
                         instance_specifics=instance_specifics,
                         cutoff=cutoff,
                         deterministic=deterministic,
                         run_obj_time=run_obj_time,
                         minR=minR,
                         maxR=maxR,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor,
                         min_chall=min_chall,)

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # general attributes
        self.run_limit = run_limit
        self.always_race_against = always_race_against

        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")

        self.use_ta_time_bound = use_ta_time_bound
        self.elapsed_time = 0.

        # stage variables
        # the intensification procedure is divided into 4 'stages':
        # 0. run 1st configuration (only in the 1st run when incumbent=None)
        # 1. add incumbent run
        # 2. race challenger
        # 3. race against configuration for a new incumbent
        self.stage = IntensifierStage.RUN_FIRST_CONFIG
        self.n_iters = 0

        # challenger related variables
        self._chall_indx = 0
        self.current_challenger = None
        self.continue_challenger = False
        self.configs_to_run = iter([])  # type: _config_to_run_type
        self.update_configs_to_run = True

        # racing related variables
        self.to_run = []  # type: typing.List[InstSeedBudgetKey]
        self.inc_sum_cost = np.inf
        self.N = -1

    def eval_challenger(self,
                        challenger: Configuration,
                        incumbent: typing.Optional[Configuration],
                        run_history: RunHistory,
                        time_bound: float = float(MAXINT),
                        log_traj: bool = True) -> typing.Tuple[Configuration, float]:
        """Running intensification to determine the incumbent configuration.
        *Side effect:* adds runs to run_history

        Implementation of Procedure 2 in Hutter et al. (2011).

        Provide either ``challengers`` or ``optimizer`` and set the other to ``None``.
        If both arguments are given, then the optimizer will be used.

        Parameters
        ----------
        challenger : Configuration
            promising configurations
        incumbent : typing.Optional[Configuration]
            best configuration so far, None in 1st run
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        time_bound : float, optional (default=2 ** 31 - 1)
            time in [sec] available to perform intensify
        log_traj: bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """

        if time_bound < self._min_time:
            raise ValueError("time_bound must be >= %f" % self._min_time)

        start_time = time.time()

        # The "intensification" is designed to be spread across multiple ``eval_challenger()`` runs
        # Lines 1 + 2 happen in the optimizer (SMBO)

        # ensure incumbent is not evaluated as challenger again
        if challenger == incumbent and self.stage == IntensifierStage.RUN_CHALLENGER:
            self.logger.debug("Challenger was the same as the current incumbent; Skipping challenger")
            inc_perf = run_history.get_cost(incumbent)
            return incumbent, inc_perf

        # if first ever run, then assume current challenger to be the incumbent
        if self.stage == IntensifierStage.RUN_FIRST_CONFIG:
            if incumbent is None:
                self.logger.info("First run, no incumbent provided; challenger is assumed to be the incumbent")
                incumbent = challenger
            else:
                inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)
                if len(inc_runs) > 0:
                    self.logger.debug("Skipping RUN_FIRST_CONFIG stage since incumbent is already run before")
                    self.stage = IntensifierStage.RUN_INCUMBENT

        self.logger.debug("Intensify on %s", challenger)
        if hasattr(challenger, 'origin'):
            self.logger.debug("Configuration origin: %s", challenger.origin)

        try:
            # since it runs only 1 "ExecuteRun" per iteration,
            # we run incumbent once and then challenger in the next
            if self.stage == IntensifierStage.RUN_FIRST_CONFIG or \
                    self.stage == IntensifierStage.RUN_INCUMBENT:
                # Lines 3-7
                self._add_inc_run(incumbent=incumbent, run_history=run_history, log_traj=log_traj)
            elif self.stage == IntensifierStage.RUN_CHALLENGER or \
                    self.stage == IntensifierStage.RUN_BASIS:
                # Lines 8-17
                incumbent = self._race_challenger(challenger=challenger,
                                                  incumbent=incumbent,
                                                  run_history=run_history,
                                                  log_traj=log_traj)

        except BudgetExhaustedException:
            # We return incumbent, SMBO stops due to its own budget checks
            inc_perf = run_history.get_cost(incumbent)
            self.logger.debug("Budget exhausted; Return incumbent")
            return incumbent, inc_perf

        # time used to evaluate challenger
        self.elapsed_time += time.time() - start_time

        # check if 1 intensification run is complete - line 18
        if self.stage == IntensifierStage.RUN_INCUMBENT and self._chall_indx >= self.min_chall:
            if self.num_run > self.run_limit:
                self.logger.info("Maximum #runs for intensification reached")
                self._next_iteration()

            if not self.use_ta_time_bound and self.elapsed_time - time_bound >= 0:
                self.logger.info("Wallclock time limit for intensification reached (used: %f sec, available: %f sec)",
                                 self.elapsed_time, time_bound)
                self._next_iteration()

            elif self._ta_time - time_bound >= 0:
                self.logger.info("TA time limit for intensification reached (used: %f sec, available: %f sec)",
                                 self._ta_time, time_bound)
                self._next_iteration()

        inc_perf = run_history.get_cost(incumbent)

        return incumbent, inc_perf

    def _add_inc_run(self,
                     incumbent: Configuration,
                     run_history: RunHistory,
                     log_traj: bool = True) -> None:
        """Add new run for incumbent

        *Side effect:* adds runs to <run_history>

        Parameters
        ----------
        incumbent : Configuration
            best configuration so far
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        """
        inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)

        # Line 3
        # First evaluate incumbent on a new instance
        if len(inc_runs) < self.maxR:
            # Line 4
            # find all instances that have the most runs on the inc
            inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)
            inc_inst = [s.instance for s in inc_runs]
            inc_inst = list(Counter(inc_inst).items())
            inc_inst.sort(key=lambda x: x[1], reverse=True)
            try:
                max_runs = inc_inst[0][1]
            except IndexError:
                self.logger.debug("No run for incumbent found")
                max_runs = 0
            inc_inst = [x[0] for x in inc_inst if x[1] == max_runs]

            available_insts = list(sorted(set(self.instances) - set(inc_inst)))

            # if all instances were used n times, we can pick an instances
            # from the complete set again
            if not self.deterministic and not available_insts:
                available_insts = self.instances

            # Line 6 (Line 5 is further down...)
            if self.deterministic:
                next_seed = 0
            else:
                next_seed = self.rs.randint(low=0, high=MAXINT, size=1)[0]

            if available_insts:
                # Line 5 (here for easier code)
                next_instance = self.rs.choice(available_insts)
                # Line 7
                self.logger.debug("Add run of incumbent")
                status, cost, dur, res = self.tae_runner.start(
                    config=incumbent,
                    instance=next_instance,
                    seed=next_seed,
                    cutoff=self.cutoff,
                    instance_specific=self.instance_specifics.get(next_instance, "0"))
                self._ta_time += dur
                self.num_run += 1
            else:
                self.logger.debug("No further instance-seed pairs for "
                                  "incumbent available.")
                # stop incumbent runs
                self.stage = IntensifierStage.RUN_CHALLENGER

            # output estimated performance of incumbent
            inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)
            inc_perf = run_history.get_cost(incumbent)
            self.logger.info("Updated estimated cost of incumbent on %d runs: %.4f"
                             % (len(inc_runs), inc_perf))

            # if running first configuration, go to next stage after 1st run
            if self.stage == IntensifierStage.RUN_FIRST_CONFIG:
                self.stage = IntensifierStage.RUN_INCUMBENT
                self._next_iteration()
            else:
                # Termination condition; after each run, this checks
                # whether further runs are necessary due to minR
                if len(inc_runs) >= self.minR or len(inc_runs) >= self.maxR:
                    self.stage = IntensifierStage.RUN_CHALLENGER
        else:
            # maximum runs for incumbent reached, do not run incumbent
            self.stage = IntensifierStage.RUN_CHALLENGER

        self._compare_configs(
            incumbent=incumbent, challenger=incumbent,
            run_history=run_history,
            log_traj=log_traj)

    def _race_challenger(self,
                         challenger: Configuration,
                         incumbent: Configuration,
                         run_history: RunHistory,
                         log_traj: bool = True) -> Configuration:
        """Aggressively race challenger against incumbent

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        incumbent : Configuration
            Best configuration so far
        run_history : smac.runhistory.runhistory.RunHistory
            Stores all runs we ran so far
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        new_incumbent: Configuration
            Either challenger or incumbent
        """

        # if list of <instance, seed> to run is not available, compute it
        if not self.to_run:
            self.to_run, self.inc_sum_cost = self._get_instances_to_run(incumbent=incumbent,
                                                                        challenger=challenger,
                                                                        run_history=run_history,
                                                                        N=self.N)
        if len(self.to_run) == 0:
            self.logger.debug("No further runs for challenger available")

        else:
            # Line 12
            # Run challenger on all <instance, seed> to run
            instance, seed, _ = self.to_run.pop()

            cutoff = self.cutoff
            if self.run_obj_time:
                cutoff = self._adapt_cutoff(challenger=challenger,
                                            run_history=run_history,
                                            inc_sum_cost=self.inc_sum_cost)
                if cutoff is not None and cutoff <= 0:
                    # no time to validate challenger
                    self.logger.debug("Stop challenger itensification due "
                                      "to adaptive capping.")
                    # challenger performance is worse than incumbent
                    # move on to the next iteration
                    self.stage = IntensifierStage.RUN_INCUMBENT
                    return incumbent

            self.logger.debug('Cutoff for challenger: %s' % str(cutoff))

            self.logger.debug("Add run of challenger")
            try:
                status, cost, dur, res = self.tae_runner.start(
                    config=challenger,
                    instance=instance,
                    seed=seed,
                    cutoff=cutoff,
                    instance_specific=self.instance_specifics.get(instance, "0"),
                    # Cutoff might be None if self.cutoff is None, but then the first if statement prevents
                    # evaluation of the second if statement
                    capped=(self.cutoff is not None) and (cutoff < self.cutoff))  # type: ignore[operator] # noqa F821
                self.num_run += 1
                self._ta_time += dur

            except CappedRunException:
                self.logger.debug("Challenger itensification timed out due "
                                  "to adaptive capping.")
                # move on to the next iteration
                self.stage = IntensifierStage.RUN_INCUMBENT
                return incumbent

        chal_runs = run_history.get_runs_for_config(challenger, only_max_observed_budget=True)
        chal_perf = run_history.get_cost(challenger)

        # if all <instance, seed> have been run, compare challenger performance
        if not self.to_run:
            new_incumbent = self._compare_configs(
                incumbent=incumbent, challenger=challenger,
                run_history=run_history,
                log_traj=log_traj)

            # update intensification stage
            if new_incumbent == incumbent:
                # move on to the next iteration
                self.stage = IntensifierStage.RUN_INCUMBENT
                self.continue_challenger = False
                self.logger.debug('Estimated cost of challenger on %d runs: %.4f, but worse than incumbent',
                                  len(chal_runs), chal_perf)

            elif new_incumbent == challenger:
                # New incumbent found
                incumbent = challenger
                self.continue_challenger = False
                # compare against basis configuration if provided, else go to next iteration
                if self.always_race_against and \
                        self.always_race_against != challenger:
                    self.stage = IntensifierStage.RUN_BASIS
                else:
                    self.stage = IntensifierStage.RUN_INCUMBENT
                self.logger.debug('Estimated cost of challenger on %d runs: %.4f, becomes new incumbent',
                                  len(chal_runs), chal_perf)

            else:  # Line 17
                # challenger is not worse, continue
                self.N = 2 * self.N
                self.continue_challenger = True
                self.logger.debug('Estimated cost of challenger on %d runs: %.4f, adding %d runs to the queue',
                                  len(chal_runs), chal_perf, self.N / 2)
        else:
            self.logger.debug('Estimated cost of challenger on %d runs: %.4f, still %d runs to go (continue racing)',
                              len(chal_runs), chal_perf, len(self.to_run))

        return incumbent

    def _get_instances_to_run(self,
                              challenger: Configuration,
                              incumbent: Configuration,
                              run_history: RunHistory,
                              N: int) -> typing.Tuple[typing.List[InstSeedBudgetKey], float]:
        """
        Returns the minimum list of <instance, seed> pairs to run the challenger on
        before comparing it with the incumbent

        Parameters
        ----------
        incumbent: Configuration
            incumbent configuration
        challenger: Configuration
            promising configuration that is presently being evaluated
        run_history: smac.runhistory.runhistory.RunHistory
            Stores all runs we ran so far
        N: int
            number of <instance, seed> pairs to select

        Returns
        -------
        typing.List[InstSeedBudgetKey]
            list of <instance, seed, budget> tuples to run
        float
            total (runtime) cost of running the incumbent on the instances (used for adaptive capping while racing)
        """
        # get next instances left for the challenger
        # Line 8
        inc_inst_seeds = set(run_history.get_runs_for_config(incumbent, only_max_observed_budget=True))
        chall_inst_seeds = set(run_history.get_runs_for_config(challenger, only_max_observed_budget=True))
        # Line 10
        missing_runs = sorted(inc_inst_seeds - chall_inst_seeds)

        # Line 11
        self.rs.shuffle(missing_runs)
        if N < 0:
            raise ValueError('Argument N must not be smaller than zero, but is %s' % str(N))
        to_run = missing_runs[:min(N, len(missing_runs))]
        missing_runs = missing_runs[min(N, len(missing_runs)):]

        # for adaptive capping
        # because of efficiency computed here
        inst_seed_pairs = list(inc_inst_seeds - set(missing_runs))
        # cost used by incumbent for going over all runs in inst_seed_pairs
        inc_sum_cost = run_history.sum_cost(
            config=incumbent,
            instance_seed_budget_keys=inst_seed_pairs,
        )

        return to_run, inc_sum_cost

    def get_next_challenger(self,
                            challengers: typing.Optional[typing.List[Configuration]],
                            chooser: typing.Optional[EPMChooser],
                            run_history: typing.Optional[RunHistory] = None,
                            repeat_configs: bool = True) -> \
            typing.Tuple[typing.Optional[Configuration], bool]:
        """
        Selects which challenger to use based on the iteration stage and set the iteration parameters.
        First iteration will choose configurations from the ``chooser`` or input challengers,
        while the later iterations pick top configurations from the previously selected challengers in that iteration

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        run_history : typing.Optional[smac.runhistory.runhistory.RunHistory]
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again

        Returns
        -------
        typing.Optional[Configuration]
            next configuration to evaluate
        bool
            flag telling if the configuration is newly sampled or one currently being tracked
        """

        # sampling from next challenger marks the beginning of a new iteration
        self.iteration_done = False

        # if in RUN_BASIS stage, return the basis configuration (i.e., `always_race_against`)
        if self.stage == IntensifierStage.RUN_BASIS:
            self.logger.debug("Race against basis configuration after incumbent change.")
            return self.always_race_against, False

        # if the current challenger could not be rejected, it is run again on more instances
        if self.current_challenger and self.continue_challenger:
            return self.current_challenger, False

        # select new configuration when entering 'race challenger' stage
        # or for the first run
        if not self.current_challenger or \
                (self.stage == IntensifierStage.RUN_CHALLENGER and not self.to_run):

            # this is a new intensification run, get the next list of configurations to run
            if self.update_configs_to_run:
                configs_to_run = self._generate_challengers(challengers=challengers, chooser=chooser)
                self.configs_to_run = typing.cast(_config_to_run_type, configs_to_run)
                self.update_configs_to_run = False

            # pick next configuration from the generator
            try:
                challenger = next(self.configs_to_run)
            except StopIteration:
                # out of challengers for the current iteration, start next incumbent iteration
                self._next_iteration()
                return None, False

            if challenger:
                # reset instance index for the new challenger
                self._chall_indx += 1
                self.current_challenger = challenger
                self.N = max(1, self.minR)
                self.to_run = []

            return challenger, True

        # return currently running challenger
        return self.current_challenger, False

    def _generate_challengers(self,
                              challengers: typing.Optional[typing.List[Configuration]],
                              chooser: typing.Optional[EPMChooser]) \
            -> _config_to_run_type:
        """
        Retuns a sequence of challengers to use in intensification
        If challengers are not provided, then optimizer will be used to generate the challenger list

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations to evaluate next
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            a sampler that generates next configurations to use for racing

        Returns
        -------
        typing.Optional[typing.Generator[Configuration]]
            A generator containing the next challengers to use
        """

        if challengers:
            # iterate over challengers provided
            self.logger.debug("Using challengers provided")
            chall_gen = iter(challengers)  # type: _config_to_run_type
        elif chooser:
            # generating challengers on-the-fly if optimizer is given
            self.logger.debug("Generating new challenger from optimizer")
            chall_gen = chooser.choose_next()
        else:
            raise ValueError('No configurations/chooser provided. Cannot generate challenger!')

        return chall_gen

    def _next_iteration(self) -> None:
        """
        Updates tracking variables at the end of an intensification run
        """
        # track iterations
        self.n_iters += 1
        self.iteration_done = True
        self.configs_to_run = iter([])
        self.update_configs_to_run = True

        # reset for a new iteration
        self.num_run = 0
        self._chall_indx = 0
        self.elapsed_time = 0
        self._ta_time = 0.0

        self.stats.update_average_configs_per_intensify(
            n_configs=self._chall_indx)
