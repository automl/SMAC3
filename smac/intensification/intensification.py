from __future__ import annotations

from typing import Any, Callable, Iterator, List, Optional, Tuple, cast

from collections import Counter
from enum import Enum

import numpy as np

from smac.configspace import Configuration
from smac.constants import MAXINT
from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.intensification.stages import IntensifierStage
from smac.runhistory import InstanceSeedBudgetKey, RunInfo, RunInfoIntent, RunValue
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import format_array, get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


# class NoMoreChallengers(Exception):
#    """Indicates that no more challengers are available for the intensification to proceed."""

#   pass


class Intensifier(AbstractIntensifier):
    r"""Races challengers against an incumbent.
    SMAC's intensification procedure, in detail:

    Procedure 2: Intensify(Θ_new, θ_inc, M, R, t_intensify, Π, cˆ)
    cˆ(θ, Π') denotes the empirical cost of θ on the subset of instances
    Π' ⊆ Π, based on the runs in R; max_config_calls is a parameter
    where:
    Θ_new: Sequence of parameter settings to evaluate, challengers in this class.
    θ_inc: incumbent parameter setting, incumbent in this class.

    1 for i := 1, . . . , length(Θ_new) do
    2     θ_new ← Θ_new[i]

                            STAGE-->RUN_INCUMBENT

    3     if R contains less than max_config_calls runs with configuration θ_inc then
    4         Π' ← {π'∈ Π | R contains less than or equal number of runs using θ_inc and π'
    0         than using θ_inc and any other π''∈ Π}
    5         π ← instance sampled uniformly at random from Π'
    6         s ← seed, drawn uniformly at random
    7         R ← ExecuteRun(R, θ_inc, π, s)
    8     N ← 1

                            STAGE-->RUN_CHALLENGER

    9     while true do
    10        S_missing ← {instance, seed} pairs for which θ_inc was run before, but not θ_new
    11        S_torun ← random subset of S_missing of size min(N, size(S_missing))
    12        foreach (π, s) ∈ S_torun do R ← ExecuteRun(R, θ_new, π, s)
    13        S_missing ← S_missing \\ S_torun
    14        Π_common ← instances for which we previously ran both θ_inc and θ_new
    15        if cˆ(θ_new, Π_common) > cˆ(θ_inc, Π_common) then break
    16        else if S_missing = ∅ then θ_inc ← θ_new; break
    17        else N ← 2 · N
    18    if time spent in this call to this procedure exceeds t_intensify and i ≥ 2 then break
    19 return [R, θ_inc]

    Parameters
    ----------
    rng : np.random.RandomState
    instances : List[str]
        list of all instance ids
    instance_specifics : Mapping[str, str]
        mapping from instance name to instance specific string
    algorithm_walltime_limit : int
        runtime algorithm_walltime_limit of TA runs
    deterministic: bool
        whether the TA is deterministic or not
    race_against: Configuration
        if incumbent changes race this configuration always against new incumbent;
        can sometimes prevent over-tuning
    use_target_algorithm_time_bound: bool,
        if true, trust time reported by the target algorithms instead of
        measuring the wallclock time for limiting the time of intensification
    run_limit : int
        Maximum number of target algorithm runs per call to intensify.
    max_config_calls : int
        Maximum number of runs per config (summed over all calls to intensifiy). Each passed and evaluated instance and
        seed combination is considered a run. Seeds are drawn uniformly at random if the target algorithm is
        non-deterministic. Otherwise, the same seed is used.
    min_config_calls : int
        Minimum number of run per config (summed over all calls to intensify). Each passed and evaluated instance and
        seed combination is considered a run. Seeds are drawn uniformly at random if the target algorithm is
        non-deterministic. Otherwise, the same seed is used.
    min_challenger: int
        minimal number of challengers to be considered
        (even if time_bound is exhausted earlier)
    """

    def __init__(
        self,
        scenario: Scenario,
        # instances: List[str],
        # instance_specifics: Mapping[str, str] = None,
        # algorithm_walltime_limit: float | None = None,
        # deterministic: bool = False,
        race_against: Configuration | None = None,
        run_limit: int = MAXINT,
        use_target_algorithm_time_bound: bool = False,
        min_config_calls: int = 1,
        max_config_calls: int = 2000,
        min_challenger: int = 2,
        intensify_percentage: float = 0.5,
        seed: int | None = None,
    ):
        super().__init__(
            scenario=scenario,
            # instances=instances,
            # instance_specifics=instance_specifics,
            # algorithm_walltime_limit=algorithm_walltime_limit,
            # deterministic=deterministic,
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            min_challenger=min_challenger,
            intensify_percentage=intensify_percentage,
            seed=seed,
        )

        if scenario.deterministic:
            if min_challenger != 1:
                logger.info("The number of minimal challengers is set to one for deterministic algorithms.")

            min_challenger = 1

        # General attributes
        self.run_limit = run_limit
        self.race_against = race_against

        if self.run_limit < 1:
            raise ValueError("The argument `run_limit` must be greather than 1.")

        self.use_target_algorithm_time_bound = use_target_algorithm_time_bound
        self.elapsed_time = 0.0

        # Stage variables
        # the intensification procedure is divided into 4 'stages':
        # 0. run 1st configuration (only in the 1st run when incumbent=None)
        # 1. add incumbent run
        # 2. race challenger
        # 3. race against configuration for a new incumbent
        self.stage = IntensifierStage.RUN_FIRST_CONFIG
        self.n_iters = 0

        # Challenger related variables
        self._challenger_id = 0
        self.num_challenger_run = 0
        self.current_challenger = None
        self.continue_challenger = False
        self.configs_to_run: Iterator[Optional[Configuration]] = iter([])
        self.update_configs_to_run = True

        # Racing related variables
        self.to_run = []  # type: List[InstanceSeedBudgetKey]
        # self.inc_sum_cost = np.inf
        self.N = -1

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def get_next_run(
        self,
        challengers: list[Configuration] | None,
        incumbent: Configuration,
        ask: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
        n_workers: int = 1,
    ) -> tuple[RunInfoIntent, RunInfo]:
        """This procedure is in charge of generating a RunInfo object to comply with lines 7 (in
        case stage is stage==RUN_INCUMBENT) or line 12 (In case of stage==RUN_CHALLENGER)

        A RunInfo object encapsulates the necessary information for a worker
        to execute the job, nevertheless, a challenger is not always available.
        This could happen because no more configurations are available or the new
        configuration to try was already executed.

        To circumvent this, a intent is also returned:

        - (intent=RUN) Run the RunInfo object (Normal Execution
        - (intent=SKIP) Skip this iteration. No challenger is available, in particular
            because challenger is the same as incumbent

        Parameters
        ----------
        challengers : List[Configuration]
            promising configurations
        incumbent: Configuration
            incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        runhistory : RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again
        n_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        intent: RunInfoIntent
            What should the smbo object do with the runinfo.
        run_info: RunInfo
            An object that encapsulates necessary information for a config run
        """
        if n_workers > 1:
            raise ValueError("The selected intensifier does not support more than 1 worker.")

        # If this function is called, it means the iteration is
        # not complete (we can be starting a new iteration, or re-running a
        # challenger due to line 17). We evaluate if a iteration is complete or not
        # via _process_results
        self.iteration_done = False

        # In case a crash happens, and FirstRunCrashedException prevents a
        # failure, revert back to running the incumbent
        # Challenger case is by construction ok, as there is no special
        # stage for its processing
        if self.stage == IntensifierStage.PROCESS_FIRST_CONFIG_RUN:
            self.stage = IntensifierStage.RUN_FIRST_CONFIG
        elif self.stage == IntensifierStage.PROCESS_INCUMBENT_RUN:
            self.stage = IntensifierStage.RUN_INCUMBENT

        # if first ever run, then assume current challenger to be the incumbent
        # Because this is the first ever run, we need to sample a new challenger
        # This new challenger is also assumed to be the incumbent
        if self.stage == IntensifierStage.RUN_FIRST_CONFIG:
            if incumbent is None:
                logger.info("No incumbent provided in the first run. Sampling a new challenger...")
                challenger, self.new_challenger = self.get_next_challenger(
                    challengers=challengers,
                    ask=ask,
                )
                incumbent = challenger
            else:
                inc_runs = runhistory.get_runs_for_config(incumbent, only_max_observed_budget=True)
                if len(inc_runs) > 0:
                    logger.debug("Skipping RUN_FIRST_CONFIG stage since incumbent has already been ran.")
                    self.stage = IntensifierStage.RUN_INCUMBENT

        # LINES 3-7
        if self.stage in [IntensifierStage.RUN_FIRST_CONFIG, IntensifierStage.RUN_INCUMBENT]:
            # Line 3
            # A modified version, that not only checks for max_config_calls
            # but also makes sure that there are runnable instances,
            # that is, instances has not been exhausted
            inc_runs = runhistory.get_runs_for_config(incumbent, only_max_observed_budget=True)

            # Line 4
            pending_instances = self._get_pending_instances(incumbent, runhistory)
            if pending_instances and len(inc_runs) < self.max_config_calls:
                # Lines 5-7
                instance, seed = self._get_next_instance(pending_instances)

                # instance_specific = "0"
                # if instance is not None:
                #    instance_specific = self.instance_specifics.get(instance, "0")

                return RunInfoIntent.RUN, RunInfo(
                    config=incumbent,
                    instance=instance,
                    # instance_specific=instance_specific,
                    seed=seed,
                    budget=0.0,
                )
            else:
                # This point marks the transitions from lines 3-7 to 8-18
                logger.debug("No further instance-seed pairs for incumbent available.")

                self.stage = IntensifierStage.RUN_CHALLENGER

        # Understand who is the active challenger.
        if self.stage == IntensifierStage.RUN_BASIS:
            # if in RUN_BASIS stage,
            # return the basis configuration (i.e., `race_against`)
            logger.debug("Race against default configuration after incumbent change.")
            challenger = self.race_against
        elif self.current_challenger and self.continue_challenger:
            # if the current challenger could not be rejected,
            # it is run again on more instances
            challenger = self.current_challenger
        else:
            # Get a new challenger if all instance/pairs have
            # been completed. Else return the currently running
            # challenger
            challenger, self.new_challenger = self.get_next_challenger(
                challengers=challengers,
                ask=ask,
            )

        # No new challengers are available for this iteration,
        # Move to the next iteration. This can only happen
        # when all configurations for this iteration are exhausted
        # and have been run in all proposed instance/pairs.
        if challenger is None:
            return RunInfoIntent.SKIP, RunInfo(
                config=None,
                instance=None,
                # instance_specific="0",
                seed=0,
                budget=0.0,
            )

        # Skip the iteration if the challenger was previously run
        if challenger == incumbent and self.stage == IntensifierStage.RUN_CHALLENGER:
            self.challenger_same_as_incumbent = True
            logger.debug("Challenger was the same as the current incumbent. Challenger is skipped.")
            return RunInfoIntent.SKIP, RunInfo(
                config=None,
                instance=None,
                # instance_specific="0",
                seed=0,
                budget=0.0,
            )

        logger.debug("Intensify on %s.", challenger.get_dictionary())
        if hasattr(challenger, "origin"):
            logger.debug("Configuration origin: %s", challenger.origin)

        if self.stage in [IntensifierStage.RUN_CHALLENGER, IntensifierStage.RUN_BASIS]:

            if not self.to_run:
                self.to_run = self._get_missing_instances(
                    incumbent=incumbent, challenger=challenger, runhistory=runhistory, N=self.N
                )

            # If there is no more configs to run in this iteration, or no more
            # time to do so, change the current stage base on how the current
            # challenger performs as compared to the incumbent. This is done
            # via _process_racer_results
            if len(self.to_run) == 0:
                # Nevertheless, if there are no more instances to run,
                # we might need to comply with line 17 and keep running the
                # same challenger. In this case, if there is not enough information
                # to decide if the challenger is better/worst than the incumbent,
                # line 17 doubles the number of instances to run.
                logger.debug("No further runs for challenger possible.")
                self._process_racer_results(
                    challenger=challenger,
                    incumbent=incumbent,
                    runhistory=runhistory,
                )

                # Request SMBO to skip this run. This function will
                # be called again, after the _process_racer_results
                # has updated the intensifier stage
                return RunInfoIntent.SKIP, RunInfo(
                    config=None,
                    instance=None,
                    # instance_specific="0",
                    seed=0,
                    budget=0.0,
                )

            else:
                # Lines 8-11
                incumbent, instance, seed = self._get_next_racer(
                    challenger=challenger,
                    incumbent=incumbent,
                    runhistory=runhistory,
                )

                # instance_specific = "0"
                # if instance is not None:
                #    instance_specific = self.instance_specifics.get(instance, "0")

                # Line 12
                return RunInfoIntent.RUN, RunInfo(
                    config=challenger,
                    instance=instance,
                    # instance_specific=instance_specific,
                    seed=seed,
                    budget=0.0,
                )
        else:
            raise ValueError("No valid stage found!")

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
        execution.

        During intensification, the following can happen:

        *   Challenger raced against incumbent
        *   Also, during a challenger run, a capped exception
            can be triggered, where no racer post processing is needed
        *   A run on the incumbent for more confidence needs to
            be processed, IntensifierStage.PROCESS_INCUMBENT_RUN
        *   The first run results need to be processed
            (PROCESS_FIRST_CONFIG_RUN)

        At the end of any run, checks are done to move to a new iteration.

        Parameters
        ----------
        run_info : RunInfo
            A RunInfo containing the configuration that was evaluated
        incumbent : Optional[Configuration]
            best configuration so far, None in 1st run
        runhistory : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float
            time in [sec] available to perform intensify
        result: RunValue
             Contain the result (status and other methadata) of exercising
             a challenger/incumbent.
        log_trajectory: bool
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
                logger.info("First run and no incumbent provided. Challenger is assumed to be the incumbent.")
                incumbent = run_info.config

        if self.stage in [
            IntensifierStage.PROCESS_INCUMBENT_RUN,
            IntensifierStage.PROCESS_FIRST_CONFIG_RUN,
        ]:
            self._target_algorithm_time += run_value.time
            self.num_run += 1
            self._process_incumbent(
                incumbent=incumbent,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )
        else:
            self.num_run += 1
            self.num_challenger_run += 1
            self._target_algorithm_time += run_value.time
            incumbent = self._process_racer_results(
                challenger=run_info.config,
                incumbent=incumbent,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )

        self.elapsed_time += run_value.endtime - run_value.starttime
        # check if 1 intensification run is complete - line 18
        # this is different to regular SMAC as it requires at least successful challenger run,
        # which is necessary to work on a fixed grid of configurations.
        if (
            self.stage == IntensifierStage.RUN_INCUMBENT
            and self._challenger_id >= self.min_challenger
            and self.num_challenger_run > 0
        ):
            if self.num_run > self.run_limit:
                logger.debug("Maximum number of runs for intensification reached.")
                self._next_iteration()

            if not self.use_target_algorithm_time_bound and self.elapsed_time - time_bound >= 0:
                logger.debug(
                    "Wallclock time limit for intensification reached (used: %f sec, available: %f sec)",
                    self.elapsed_time,
                    time_bound,
                )

                self._next_iteration()

            elif self._target_algorithm_time - time_bound >= 0:
                logger.debug(
                    "TA time limit for intensification reached (used: %f sec, available: %f sec)",
                    self._target_algorithm_time,
                    time_bound,
                )

                self._next_iteration()

        inc_perf = runhistory.get_cost(incumbent)

        return incumbent, inc_perf

    def _get_next_instance(
        self,
        pending_instances: list[str],
    ) -> tuple[str, int]:
        """Method to extract the next seed/instance in which a incumbent run most be evaluated.

        Parameters
        ----------
        pending_instances : List[str]
            A list of instances from which to extract the next incumbent run

        Returns
        -------
        instance: str
            Next instance to evaluate
        seed: float
            Seed in which to evaluate the instance
        """
        # Line 5 - and avoid https://github.com/numpy/numpy/issues/10791
        _idx = self.rng.choice(len(pending_instances))
        next_instance = pending_instances[_idx]

        # Line 6
        if self.deterministic:
            next_seed = 0
        else:
            next_seed = int(self.rng.randint(low=0, high=MAXINT, size=1)[0])

        # Line 7
        logger.debug("Add run of incumbent for instance={}".format(next_instance))
        if self.stage == IntensifierStage.RUN_FIRST_CONFIG:
            self.stage = IntensifierStage.PROCESS_FIRST_CONFIG_RUN
        else:
            self.stage = IntensifierStage.PROCESS_INCUMBENT_RUN

        return next_instance, next_seed

    def _get_pending_instances(
        self,
        incumbent: Configuration,
        runhistory: RunHistory,
    ) -> list[str]:
        """Implementation of line 4 of Intensification.

        This method queries the inc runs in the run history
        and return the pending instances if any is available.

        Parameters
        ----------
        incumbent: Configuration
            Either challenger or incumbent
        runhistory : RunHistory
            stores all runs we ran so far
        log_trajectory: bool
            Whether to log changes of incumbents in trajectory
        """
        # Line 4
        # Find all instances that have the most runs on the inc
        inc_runs = runhistory.get_runs_for_config(incumbent, only_max_observed_budget=True)
        inc_inst = [s.instance for s in inc_runs]
        inc_inst = list(Counter(inc_inst).items())

        inc_inst.sort(key=lambda x: x[1], reverse=True)
        try:
            max_runs = inc_inst[0][1]
        except IndexError:
            logger.debug("No run for incumbent found.")
            max_runs = 0

        inc_inst = [x[0] for x in inc_inst if x[1] == max_runs]
        available_insts = list(sorted(set(self.instances) - set(inc_inst)))

        # If all instances were used n times, we can pick an instances
        # from the complete set again
        if not self.deterministic and not available_insts:
            available_insts = self.instances

        return available_insts

    def _process_incumbent(
        self,
        incumbent: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> None:
        """Method to process the results of a challenger that races an incumbent.

        Parameters
        ----------
        incumbent: Configuration
            Either challenger or incumbent
        runhistory : RunHistory
            stores all runs we ran so far
        log_trajectory: bool
            Whether to log changes of incumbents in trajectory
        """
        # output estimated performance of incumbent
        inc_runs = runhistory.get_runs_for_config(incumbent, only_max_observed_budget=True)
        inc_perf = runhistory.get_cost(incumbent)
        format_value = format_array(inc_perf)
        logger.info(f"Updated estimated cost of incumbent on {len(inc_runs)} runs: {format_value}")

        # if running first configuration, go to next stage after 1st run
        if self.stage in [
            IntensifierStage.RUN_FIRST_CONFIG,
            IntensifierStage.PROCESS_FIRST_CONFIG_RUN,
        ]:
            self.stage = IntensifierStage.RUN_INCUMBENT
            self._next_iteration()
        else:
            # Termination condition; after each run, this checks
            # whether further runs are necessary due to min_config_calls
            if len(inc_runs) >= self.min_config_calls or len(inc_runs) >= self.max_config_calls:
                self.stage = IntensifierStage.RUN_CHALLENGER
            else:
                self.stage = IntensifierStage.RUN_INCUMBENT

        self._compare_configs(
            incumbent=incumbent, challenger=incumbent, runhistory=runhistory, log_trajectory=log_trajectory
        )

    def _get_next_racer(
        self,
        challenger: Configuration,
        incumbent: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> Tuple[Configuration, str, int]:
        """Method to return the next config setting to aggressively race challenger against
        incumbent.

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        incumbent : Configuration
            Best configuration so far
        runhistory : RunHistory
            Stores all runs we ran so far
        log_trajectory: bool
               Whether to log changes of incumbents in trajectory

        Returns
        -------
        new_incumbent: Configuration
            Either challenger or incumbent
        instance: str
            Next instance to evaluate
        seed: int
            Seed in which to evaluate the instance
        """
        # By the time this function is called, the run history might
        # have shifted. Re-populate the list if necessary
        if not self.to_run:
            # Lines 10/11
            self.to_run = self._get_missing_instances(
                incumbent=incumbent, challenger=challenger, runhistory=runhistory, N=self.N
            )

        # Run challenger on all <instance, seed> to run
        instance_seed_budget_key = self.to_run.pop()
        instance = instance_seed_budget_key.instance
        seed = instance_seed_budget_key.seed

        # Line 12
        return incumbent, instance, seed

    def _process_racer_results(
        self,
        challenger: Configuration,
        incumbent: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> Configuration | None:
        """Process the result of a racing configuration against the current incumbent. Might propose
        a new incumbent.

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        incumbent : Configuration
            Best configuration so far
        runhistory : RunHistory
            Stores all runs we ran so far

        Returns
        -------
        new_incumbent: Optional[Configuration]
            Either challenger or incumbent
        """
        chal_runs = runhistory.get_runs_for_config(challenger, only_max_observed_budget=True)
        chal_perf = runhistory.get_cost(challenger)

        # If all <instance, seed> have been run, compare challenger performance
        if not self.to_run:
            new_incumbent = self._compare_configs(
                incumbent=incumbent,
                challenger=challenger,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )

            # update intensification stage
            if new_incumbent == incumbent:
                # move on to the next iteration
                self.stage = IntensifierStage.RUN_INCUMBENT
                self.continue_challenger = False
                logger.debug(
                    "Estimated cost of challenger on %d runs: %.4f, but worse than incumbent",
                    len(chal_runs),
                    chal_perf,
                )

            elif new_incumbent == challenger:
                # New incumbent found
                incumbent = challenger
                self.continue_challenger = False
                # compare against basis configuration if provided, else go to next iteration
                if self.race_against and self.race_against != challenger:
                    self.stage = IntensifierStage.RUN_BASIS
                else:
                    self.stage = IntensifierStage.RUN_INCUMBENT
                logger.debug(
                    "Estimated cost of challenger on %d runs: %.4f, becomes new incumbent",
                    len(chal_runs),
                    chal_perf,
                )

            else:  # Line 17
                # challenger is not worse, continue
                self.N = 2 * self.N
                self.continue_challenger = True
                logger.debug(
                    "Estimated cost of challenger on %d runs: %.4f, adding %d runs to the queue",
                    len(chal_runs),
                    chal_perf,
                    self.N / 2,
                )
        else:
            logger.debug(
                "Estimated cost of challenger on %d runs: %.4f, still %d runs to go (continue racing)",
                len(chal_runs),
                chal_perf,
                len(self.to_run),
            )

        return incumbent

    def _get_missing_instances(
        self,
        challenger: Configuration,
        incumbent: Configuration,
        N: int,
        runhistory: RunHistory,
    ) -> list[InstanceSeedBudgetKey]:
        """Returns the minimum list of <instance, seed> pairs to run the challenger on before
        comparing it with the incumbent.

        Parameters
        ----------
        incumbent: Configuration
            incumbent configuration
        challenger: Configuration
            promising configuration that is presently being evaluated
        runhistory: RunHistory
            Stores all runs we ran so far
        N: int
            number of <instance, seed> pairs to select

        Returns
        -------
        List[InstSeedBudgetKey]
            list of <instance, seed, budget> tuples to run
        # float
        #     total (runtime) cost of running the incumbent on the instances (used for adaptive capping while racing)
        """
        # Get next instances left for the challenger
        # Line 8
        inc_inst_seeds = set(runhistory.get_runs_for_config(incumbent, only_max_observed_budget=True))
        chall_inst_seeds = set(runhistory.get_runs_for_config(challenger, only_max_observed_budget=True))

        # Line 10
        missing_runs = sorted(inc_inst_seeds - chall_inst_seeds)

        # Line 11
        self.rng.shuffle(missing_runs)  # type: ignore
        if N < 0:
            raise ValueError("Argument N must not be smaller than zero, but is %s" % str(N))

        to_run = missing_runs[: min(N, len(missing_runs))]
        missing_runs = missing_runs[min(N, len(missing_runs)) :]

        # for adaptive capping
        # because of efficiency computed here
        # instance_seed_pairs = list(inc_inst_seeds - set(missing_runs))
        # cost used by incumbent for going over all runs in instance_seed_pairs
        # inc_sum_cost = runhistory.sum_cost(config=incumbent, instance_seed_budget_keys=instance_seed_pairs, normalize=True)
        # assert type(inc_sum_cost) == float

        return to_run  # , inc_sum_cost

    def get_next_challenger(
        self,
        challengers: list[Configuration] | None,
        ask: Callable[[], Iterator[Configuration]] | None,
    ) -> tuple[Configuration | None, bool]:
        """This function returns the next challenger, that should be exercised though lines 8-17.

        It does so by populating configs_to_run, which is a pool of configuration
        from which the racer will sample. Each configuration within configs_to_run,
        will be intensified on different instances/seed registered in self.to_run
        as stated in line 11.

        A brand new configuration should only be sampled, after all self.to_run
        instance seed pairs are exhausted.

        This method triggers a call to _next_iteration if there are no more configurations
        to run, for the current intensification loop. This marks the transition to Line 2,
        where a new configuration to intensify will be drawn from epm/initial challengers.


        Parameters
        ----------
        challengers : List[Configuration]
            promising configurations
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing

        Returns
        -------
        Optional[Configuration]
            next configuration to evaluate
        bool
            flag telling if the configuration is newly sampled or one currently being tracked
        """
        # select new configuration when entering 'race challenger' stage
        # or for the first run
        if not self.current_challenger or (self.stage == IntensifierStage.RUN_CHALLENGER and not self.to_run):

            # this is a new intensification run, get the next list of configurations to run
            if self.update_configs_to_run:
                configs_to_run = self._generate_challengers(challengers=challengers, ask=ask)
                self.configs_to_run = cast(Iterator[Optional[Configuration]], configs_to_run)
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
                self._challenger_id += 1
                self.current_challenger = challenger
                self.N = max(1, self.min_config_calls)
                self.to_run = []

            return challenger, True

        # return currently running challenger
        return self.current_challenger, False

    def _generate_challengers(
        self,
        challengers: list[Configuration] | None,
        ask: Callable[[], Iterator[Configuration]] | None,
    ) -> Iterator[Optional[Configuration]]:
        """Retuns a sequence of challengers to use in intensification If challengers are not
        provided, then optimizer will be used to generate the challenger list.

        Parameters
        ----------
        challengers : List[Configuration]
            promising configurations to evaluate next
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            a sampler that generates next configurations to use for racing

        Returns
        -------
        Optional[Generator[Configuration]]
            A generator containing the next challengers to use
        """
        chall_gen: Iterator[Optional[Configuration]]
        if challengers:
            # iterate over challengers provided
            logger.debug("Using provided challengers...")
            chall_gen = iter(challengers)
        elif ask:
            # generating challengers on-the-fly if optimizer is given
            logger.debug("Generating new challenger from optimizer...")
            chall_gen = ask()
        else:
            raise ValueError("No configurations/ask function provided. Can not generate challenger!")

        return chall_gen

    def _next_iteration(self) -> None:
        """Updates tracking variables at the end of an intensification run."""
        assert self.stats

        # track iterations
        self.n_iters += 1
        self.iteration_done = True
        self.configs_to_run = iter([])
        self.update_configs_to_run = True

        # reset for a new iteration
        self.num_run = 0
        self.num_challenger_run = 0
        self._challenger_id = 0
        self.elapsed_time = 0
        self._target_algorithm_time = 0.0

        self.stats.update_average_configs_per_intensify(n_configs=self._challenger_id)
