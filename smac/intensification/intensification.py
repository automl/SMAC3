import logging
import typing
from collections import Counter
from enum import Enum

import numpy as np

from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.configspace import Configuration
from smac.runhistory.runhistory import (
    InstSeedBudgetKey,
    RunInfo,
    RunHistory,
    RunValue,
    StatusType
)
from smac.utils.io.traj_logging import TrajLogger
from smac.intensification.abstract_racer import (
    AbstractRacer,
    IntensifierBehest,
    _config_to_run_type,
)
from smac.optimizer.epm_configuration_chooser import EPMChooser

__author__ = "Katharina Eggensperger, Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class NoMoreChallengers(Exception):
    """Indicates that no more challengers are available
    for the intensification to proceed"""
    pass


class IntensifierStage(Enum):
    """Class to define different stages of intensifier"""
    RUN_FIRST_CONFIG = 0  # to replicate the old initial design
    RUN_INCUMBENT = 1  # Lines 3-7
    RUN_CHALLENGER = 2  # Lines 8-17
    RUN_BASIS = 3

    # helpers to determine what type of run to process
    # A challenger is assumed to be processed if the stage
    # is not from first_config or incumbent
    PROCESS_FIRST_CONFIG_RUN = 4
    PROCESS_INCUMBENT_RUN = 5


class Intensifier(AbstractRacer):
    """Races challengers against an incumbent

    SMAC's intensification procedure is denoted below.
    This class honors bellow procedure and implements it's sequential
    nature via stages (IntensifierStage), which are also highlighted below

    Procedure 2: Intensify(Θ~new, θinc, M, R, tintensify, Π, cˆ)
    cˆ(θ, Π0) denotes the empirical cost of θ on the subset of instances
    Π0 ⊆ Π, based on the runs in R; maxR is a parameter
    ____________________________________________________________
    1 for i := 1, . . . , length(Θ~ new) do
    2 θnew ← Θ~ new[i]
                        _____________________
                        STAGE-->RUN_INCUMBENT
                        ---------------------
    3 if R contains less than maxR runs with configuration θinc then
    4 Π0 ← {π
    0 ∈ Π | R contains less than or equal number of runs using θinc and π
    0
    than using θinc and any other π
    00 ∈ Π}
    5 π ← instance sampled uniformly at random from Π0
    6 s ← seed, drawn uniformly at random
    7 R ← ExecuteRun(R, θinc, π, s)
    8 N ← 1
                        ______________________
                        STAGE-->RUN_CHALLENGER
                        ----------------------
    9 while true do
    10 Smissing ← hinstance, seedi pairs for which θinc was run before, but not θnew
    11 Storun ←random subset of Smissing of size min(N, |Smissing|)
    12 foreach (π, s) ∈ Storun do R ← ExecuteRun(R, θnew, π, s)
    13 Smissing ← Smissing \ Storun
    14 Πcommon ← instances for which we previously ran both θinc and θnew
    15 if cˆ(θnew, Πcommon) > cˆ(θinc, Πcommon) then break
    16 else if Smissing = ∅ then θinc ← θnew; break
    17 else N ← 2 · N
    18 if time spent in this call to this procedure exceeds tintensify and i ≥ 2 then break
    19 return [R, θinc]

    Parameters
    ----------
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
        minimal number of challengers to be considered
        (even if time_bound is exhausted earlier)
    """

    def __init__(self,
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

        super().__init__(stats=stats,
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
        self.num_chall_run = 0
        self.current_challenger = None
        self.continue_challenger = False
        self.configs_to_run = iter([])  # type: _config_to_run_type
        self.update_configs_to_run = True

        # racing related variables
        self.to_run = []  # type: typing.List[InstSeedBudgetKey]
        self.inc_sum_cost = np.inf
        self.N = -1

    def get_next_run(self,
                     challengers: typing.Optional[typing.List[Configuration]],
                     incumbent: Configuration,
                     chooser: typing.Optional[EPMChooser],
                     run_history: RunHistory,
                     repeat_configs: bool = True,
                     ) -> typing.Tuple[IntensifierBehest, RunInfo]:
        """
        This procedure is in charge of generating a RunInfo object to comply
        with lines 7 (in case stage is stage==RUN_INCUMBENT) or line 12
        (In case of stage==RUN_CHALLENGER)

        A RunInfo object encapsulates the necessary information for a worker
        to execute the job, nevertheless, a challenger is not always available.
        This could happen because no more configurations are available or the new
        configuration to try was already executed.

        To circumvent this, a behest is also returned:

        - (behest=RUN) Run the RunInfo object (Normal Execution
        - (behest=TERMINATE) No more challengers available (Line 19 should occur)
        - (behest=SKIP) Skip this iteration. No challenger is available, in particular
            because challenger is the same as incumbent


        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent: Configuration
            incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        run_history : typing.Optional[smac.runhistory.runhistory.RunHistory]
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again

        Returns
        -------
        behest: IntensifierBehest
            What should the smbo object do with the runinfo.
        run_info: RunInfo
            An object that encapsulates necessary information for a config run
        """

        # In case a crash happens, and FirstRunCrashedException prevents a
        # failure, revert back to running the incumbent
        # Challenger case is by construction ok, as there is no special
        # stage for its processing
        if self.stage == IntensifierStage.PROCESS_FIRST_CONFIG_RUN:
            self.stage = IntensifierStage.RUN_FIRST_CONFIG
        elif self.stage == IntensifierStage.PROCESS_INCUMBENT_RUN:
            self.stage = IntensifierStage.RUN_INCUMBENT

        try:

            # Line 2:
            # Understand who is the challenger.
            # Not necessarily a challenger is always a new configuration,
            # as this method might be called multiple times to
            # gain more confidence on incumbent, or even run the same
            # challenger on multiple N (line 17)
            challenger = self.query_challenger(
                challengers=challengers,
                chooser=chooser,
                run_history=run_history,
                repeat_configs=repeat_configs,
            )

            # if first ever run, then assume current challenger to be the incumbent
            # In this case, we need a branch new challenger
            if self.stage == IntensifierStage.RUN_FIRST_CONFIG:
                if incumbent is None:
                    self.logger.info(
                        "First run, no incumbent provided;"
                        " challenger is assumed to be the incumbent"
                    )
                    incumbent = challenger
                else:
                    inc_runs = run_history.get_runs_for_config(
                        incumbent,
                        only_max_observed_budget=True
                    )
                    if len(inc_runs) > 0:
                        self.logger.debug(
                            "Skipping RUN_FIRST_CONFIG stage since "
                            "incumbent is already run before"
                        )
                        self.stage = IntensifierStage.RUN_INCUMBENT

            # Before we proceed with the intensification loop,
            # we have to check if it makes sense to run the challenger
            # That is, if running lines 8-18 but there is no more time due
            # to adaptive caping or there are no more instances to run
            # then we can proactively assume we will do a new intensification
            # round
            if self.stage in [IntensifierStage.RUN_CHALLENGER, IntensifierStage.RUN_BASIS]:
                if not self.to_run:
                    self.to_run, self.inc_sum_cost = self._get_instances_to_run(
                        incumbent=incumbent,
                        challenger=challenger,
                        run_history=run_history,
                        N=self.N
                    )

                is_there_time_due_to_adaptive_cap = self._is_there_time_due_to_adaptive_cap(
                    challenger=challenger,
                    run_history=run_history,
                )

                if len(self.to_run) == 0 or not is_there_time_due_to_adaptive_cap:

                    # If no more time, stage transition is a must
                    if not is_there_time_due_to_adaptive_cap:
                        self.stage = IntensifierStage.RUN_INCUMBENT
                        self.logger.debug(
                            "Stop challenger itensification due "
                            "to adaptive capping."
                        )

                    # Nevertheless, if there are no more instances to run,
                    # we might need to comply with line 17 and keep running the
                    # same challenger
                    self.logger.debug("No further runs for challenger possible")
                    self._process_racer_results(
                        challenger=challenger,
                        incumbent=incumbent,
                        run_history=run_history,
                    )

                    # Remove the challenger
                    if challengers:
                        challengers = [c for c in challengers if c != challenger]
                    challenger = self.query_challenger(
                        challengers=challengers,
                        chooser=chooser,
                        run_history=run_history,
                        repeat_configs=repeat_configs,
                    )

            # LINES 3-7
            if self.stage in [IntensifierStage.RUN_FIRST_CONFIG, IntensifierStage.RUN_INCUMBENT]:

                # Line 3
                # A modified version, that not only checks for maxR
                # but also makes sure that there are runnable instances,
                # that is, instances has not been exhausted
                inc_runs = run_history.get_runs_for_config(
                    incumbent,
                    only_max_observed_budget=True
                )

                # Line 4
                available_insts = self._get_inc_available_inst(incumbent, run_history)
                if available_insts and len(inc_runs) < self.maxR:
                    # Lines 5-6-7
                    instance, seed, cutoff = self._get_next_inc_config(available_insts)

                    instance_specific = "0"
                    if instance is not None:
                        instance_specific = self.instance_specifics.get(instance, "0")

                    return IntensifierBehest.RUN, RunInfo(
                        config=incumbent,
                        instance=instance,
                        instance_specific=instance_specific,
                        seed=seed,
                        cutoff=cutoff,
                        capped=False,
                        budget=0.0,
                    )
                else:
                    # This point marks the transitions from lines 3-7
                    # to 8-18.

                    self.logger.debug(
                        "No further instance-seed pairs for incumbent available."
                    )

                    self.stage = IntensifierStage.RUN_CHALLENGER

                    self._process_inc_run(
                        incumbent=incumbent,
                        run_history=run_history,
                    )

                    # Because we are moving to lines 8-18
                    # make sure that the challenger is properly
                    # set
                    challenger = self.query_challenger(
                        challengers=challengers,
                        chooser=chooser,
                        run_history=run_history,
                        repeat_configs=repeat_configs,
                    )

            # Skip the iteration if the challenger was previously run
            if challenger == incumbent and self.stage == IntensifierStage.RUN_CHALLENGER:
                self.challenger_same_as_incumbent = True
                self.logger.debug(
                    "Challenger was the same as the current incumbent; Skipping challenger"
                )
                return IntensifierBehest.SKIP, RunInfo(
                    config=None,
                    instance=None,
                    instance_specific="0",
                    seed=0,
                    cutoff=self.cutoff,
                    capped=False,
                    budget=0.0,
                )

            self.logger.debug("Intensify on %s", challenger)
            if hasattr(challenger, 'origin'):
                self.logger.debug("Configuration origin: %s", challenger.origin)

            if self.stage in [IntensifierStage.RUN_CHALLENGER, IntensifierStage.RUN_BASIS]:
                # Lines 8-11
                incumbent, instance, seed, cutoff = self._get_next_racer(
                    challenger=challenger,
                    incumbent=incumbent,
                    run_history=run_history,
                )

                capped = False
                if (self.cutoff is not None) and (cutoff < self.cutoff):  # type: ignore[operator] # noqa F821
                    capped = True

                instance_specific = "0"
                if instance is not None:
                    instance_specific = self.instance_specifics.get(instance, "0")

                # Line 12
                return IntensifierBehest.RUN, RunInfo(
                    config=challenger,
                    instance=instance,
                    instance_specific=instance_specific,
                    seed=seed,
                    cutoff=cutoff,
                    capped=capped,
                    budget=0.0,
                )
            else:
                raise ValueError('No valid stage found!')

        except NoMoreChallengers:
            # No new challengers are available even after a request to draw from the
            # epm/initial configs. This means we exhausted the loop condition on line 1
            return IntensifierBehest.TERMINATE, RunInfo(
                config=None,
                instance=None,
                instance_specific="0",
                seed=0,
                cutoff=self.cutoff,
                capped=False,
                budget=0.0,
            )

    def process_results(self,
                        challenger: Configuration,
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

        During intensification, a the following can happen:
        -> Challenger raced against incumbent. self.added_challenger_run
           is set to true and that run result need to be processed
        -> Also, during a challenger run, a capped exception
           can be triggered, where no racer post processing is needed
        -> A run on the incumbent for more confidence needs to
           be processed, IntensifierStage.PROCESS_INCUMBENT_RUN

        At the end of any run, checks are done to move to a new iteration.

        Parameters
        ----------
        challenger : Configuration
            a configuration that was previously executed, and whose status
            will be used to define the next stage.
        incumbet : typing.Optional[Configuration]
            best configuration so far, None in 1st run
        run_history : typing.Optional[smac.runhistory.runhistory.RunHistory]
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float, optional (default=2 ** 31 - 1)
            time in [sec] available to perform intensify
        result: RunValue
             Contain the result (status and other methadata) of exercising
             a challenger/incumbent.
        log_traj: bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
        if self.stage == IntensifierStage.PROCESS_FIRST_CONFIG_RUN:
            if incumbent is None:
                self.logger.info(
                    "First run, no incumbent provided;"
                    " challenger is assumed to be the incumbent"
                )
                incumbent = challenger

        if self.stage in [IntensifierStage.PROCESS_INCUMBENT_RUN,
                          IntensifierStage.PROCESS_FIRST_CONFIG_RUN]:
            self._ta_time += result.time
            self.num_run += 1
            self._process_inc_run(
                incumbent=incumbent,
                run_history=run_history,
                log_traj=log_traj,
            )

        else:
            self.num_run += 1
            self.num_chall_run += 1
            if result.status == StatusType.CAPPED:
                # move on to the next iteration
                self.logger.debug(
                    "Challenger itensification timed out due "
                    "to adaptive capping."
                )
                self.stage = IntensifierStage.RUN_INCUMBENT
            else:

                self._ta_time += result.time
                incumbent = self._process_racer_results(
                    challenger=challenger,
                    incumbent=incumbent,
                    run_history=run_history,
                    log_traj=log_traj,
                )

        elapsed_time = result.endtime - result.starttime
        # check if 1 intensification run is complete - line 18
        # this is different to regular SMAC as it requires at least successful challenger run,
        # which is necessary to work on a fixed grid of configurations.
        if (
            self.stage == IntensifierStage.RUN_INCUMBENT
            and self._chall_indx >= self.min_chall
            and self.num_chall_run > 0
        ):
            if self.num_run > self.run_limit:
                self.logger.info("Maximum #runs for intensification reached")
                self._next_iteration()

            if not self.use_ta_time_bound and elapsed_time - time_bound >= 0:
                self.logger.info(
                    "Wallclock time limit for intensification reached "
                    "(used: %f sec, available: %f sec)",
                    elapsed_time,
                    time_bound
                )

                self._next_iteration()

            elif self._ta_time - time_bound >= 0:
                self.logger.info(
                    "TA time limit for intensification reached (used: %f sec, available: %f sec)",
                    self._ta_time, time_bound
                )

                self._next_iteration()

        inc_perf = run_history.get_cost(incumbent)

        return incumbent, inc_perf

    def _get_next_inc_config(self,
                             available_insts: typing.List[str],
                             ) -> typing.Tuple[str, int, typing.Optional[float]]:
        """Method to extract the next seed/instance in which a
        incumbent run most be evaluated.

        Parameters
        ----------
        available_insts : typing.List[str]
            A list of instances from which to extract the next incumbent run

        Returns
        -------
        instance: str
            Next instance to evaluate
        seed: float
            Seed in which to evaluate the instance
        cutoff: Optional[float]
            Max time for a given instance/seed pair

        """
        # Line 5
        next_instance = self.rs.choice(available_insts)

        # Line 6
        if self.deterministic:
            next_seed = 0
        else:
            next_seed = self.rs.randint(low=0, high=MAXINT, size=1)[0]

        # Line 7
        self.logger.debug(
            "Add run of incumbent for instance={}".format(
                next_instance
            )
        )
        if self.stage == IntensifierStage.RUN_FIRST_CONFIG:
            self.stage = IntensifierStage.PROCESS_FIRST_CONFIG_RUN
        else:
            self.stage = IntensifierStage.PROCESS_INCUMBENT_RUN

        return next_instance, next_seed, self.cutoff

    def _get_inc_available_inst(
        self,
        incumbent: Configuration,
        run_history: RunHistory,
        log_traj: bool = True,
    ) -> typing.List[str]:
        """
        Implementation of line 4 of Intensification

        This method queries the inc runs in the run history
        and return the pending instances if any is available

        Parameters
        ----------
        incumbent: Configuration
            Either challenger or incumbent
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        """
        # Line 4
        # find all instances that have the most runs on the inc
        inc_runs = run_history.get_runs_for_config(
            incumbent,
            only_max_observed_budget=True
        )
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
        return available_insts

    def _process_inc_run(self,
                         incumbent: Configuration,
                         run_history: RunHistory,
                         log_traj: bool = True,
                         ) -> None:
        """Method to process the results of a challenger that races
        an incumbent

        Parameters
        ----------
        incumbent: Configuration
            Either challenger or incumbent
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        """
        # output estimated performance of incumbent
        inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)
        inc_perf = run_history.get_cost(incumbent)
        self.logger.info("Updated estimated cost of incumbent on %d runs: %.4f"
                         % (len(inc_runs), inc_perf))

        # if running first configuration, go to next stage after 1st run
        if self.stage in [IntensifierStage.RUN_FIRST_CONFIG,
                          IntensifierStage.PROCESS_FIRST_CONFIG_RUN]:
            self.stage = IntensifierStage.RUN_INCUMBENT
            self._next_iteration()
        else:
            # Termination condition; after each run, this checks
            # whether further runs are necessary due to minR
            if len(inc_runs) >= self.minR or len(inc_runs) >= self.maxR:
                self.stage = IntensifierStage.RUN_CHALLENGER
            else:
                self.stage = IntensifierStage.RUN_INCUMBENT

        self._compare_configs(
            incumbent=incumbent,
            challenger=incumbent,
            run_history=run_history,
            log_traj=log_traj
        )

    def _get_next_racer(
        self,
        challenger: Configuration,
        incumbent: Configuration,
        run_history: RunHistory,
        log_traj: bool = True,
    ) -> typing.Tuple[Configuration, str, int, typing.Optional[float]]:
        """Method to return the next config setting to
        aggressively race challenger against incumbent.

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
        instance: str
            Next instance to evaluate
        seed: int
            Seed in which to evaluate the instance
        cutoff: Optional[float]
            Max time for a given instance/seed pair
        """

        # By the time this function is called, the run history might
        # have shifted. Re-populate the list if necessary
        if not self.to_run:

            # Lines 10/11
            self.to_run, self.inc_sum_cost = self._get_instances_to_run(
                incumbent=incumbent,
                challenger=challenger,
                run_history=run_history,
                N=self.N
            )

        # Run challenger on all <instance, seed> to run
        instance, seed, _ = self.to_run.pop()

        cutoff = self.cutoff
        if self.run_obj_time:
            cutoff = self._adapt_cutoff(challenger=challenger,
                                        run_history=run_history,
                                        inc_sum_cost=self.inc_sum_cost)

        self.logger.debug('Cutoff for challenger: %s' % str(cutoff))

        self.logger.debug("Add run of challenger")

        # Line 12
        return incumbent, instance, seed, cutoff

    def _is_there_time_due_to_adaptive_cap(
        self,
        challenger: Configuration,
        run_history: RunHistory,
    ) -> bool:
        """
        A check to see if there is no more time for a challenger
        given the fact, that we are optimizing time and the
        incumbent looks more promising
        Line 18

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        run_history : smac.runhistory.runhistory.RunHistory
            Stores all runs we ran so far
        Returns
        -------
        bool:
            whether or not there is more time for a challenger run
        """
        # If time is not objective, then there is always time!
        if not self.run_obj_time:
            return True

        cutoff = self._adapt_cutoff(
            challenger=challenger,
            run_history=run_history,
            inc_sum_cost=self.inc_sum_cost
        )
        if cutoff is not None and cutoff <= 0:
            return False
        else:
            return True

    def _process_racer_results(self,
                               challenger: Configuration,
                               incumbent: Configuration,
                               run_history: RunHistory,
                               log_traj: bool = True,
                               ) -> typing.Optional[Configuration]:
        """Process the result of a racing configuration against the
        current incumbent. Might propose a new incumbent.

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        incumbent : Configuration
            Best configuration so far
        run_history : smac.runhistory.runhistory.RunHistory
            Stores all runs we ran so far

        Returns
        -------
        new_incumbent: typing.Optional[Configuration]
            Either challenger or incumbent
        """
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
                              N: int,
                              run_history: RunHistory,
                              ) -> typing.Tuple[typing.List[InstSeedBudgetKey], float]:
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

    def query_challenger(self,
                         challengers: typing.Optional[typing.List[Configuration]],
                         chooser: typing.Optional[EPMChooser],
                         run_history: typing.Optional[RunHistory] = None,
                         repeat_configs: bool = True) -> Configuration:
        """
        Handy wrapper around _query challenger to try to get a new configuration.

        In case of failure, it raise a NoMoreChallengers exception, suggesting
        a termination of the intensification procedure as there are no more
        challengers to be extracted from the initial configs/epm

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

        Raises
        ------
        NoMoreChallengers:
            Indication that for this iterations, no more challengers are pertinent.
            The caller should catch this to proceed with a new iteration accordingly.
        """
        # Line 8-18
        # Query if a new challenger is required based on the stage
        # No new challenger is required if we continue running a challenger
        # or we gain confidence on the incumbent. In any other case, a new
        # challenger is generated, which marks a new iteration of the for in line 1
        try:
            challenger, self.new_challenger = self._query_challenger(
                challengers=challengers,
                chooser=chooser,
                run_history=run_history,
                repeat_configs=repeat_configs,
            )
        except StopIteration:
            # out of challengers for the current iteration, start next incumbent iteration
            self._next_iteration()
            try:
                challenger, self.new_challenger = self._query_challenger(
                    challengers=challengers,
                    chooser=chooser,
                    run_history=run_history,
                    repeat_configs=repeat_configs,
                )
            except StopIteration:
                raise NoMoreChallengers()

        return challenger

    def _query_challenger(self,
                          challengers: typing.Optional[typing.List[Configuration]],
                          chooser: typing.Optional[EPMChooser],
                          run_history: typing.Optional[RunHistory] = None,
                          repeat_configs: bool = True) -> \
            typing.Tuple[typing.Optional[Configuration], bool]:
        """
        The intensify procedure can be called multiple times for the same configuration.

        That is, not necessarily a new challenger will be produce, as following scenarios
        are relevant (in order of priority):

        -> RUN_BASIS:
            Prioritize running against a configuration
        -> Line 17:
            Not enough N runs available to decide if the challenger is better/worst than
            the incumbent, so continue challenger is honored.
        -> Line 2/11:
            Get a new configuration to intensify
        -> Line 3-7:
            Keep going over incumbent

        Regarding challengers, the first iteration will choose configurations from the
        ``chooser`` or input challengers, while the later iterations pick top configurations
        from the previously selected challengers in that iteration

        This proposed challenger can be None, which marks the start of a new iteration.

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

        Raises
        ------
        StopIteration:
            Indication that for this iterations, no more challengers are pertinent.
            The caller should catch this to proceed with a new iteration accordingly.
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
            challenger = next(self.configs_to_run)

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
        self.num_chall_run = 0
        self._chall_indx = 0
        self.elapsed_time = 0
        self._ta_time = 0.0

        self.stats.update_average_configs_per_intensify(
            n_configs=self._chall_indx)
