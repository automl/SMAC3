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
    r"""Races challengers against an incumbent.

    SMAC's intensification procedure, in detail:
    Intensify(Θ_new, θ_inc, M, R, t_intensify, Π, c_hat) c_hat(θ, Π') denotes the empirical cost of θ on
    the subset of instances Π' ⊆ Π, based on the trials in R; max_config_calls is a parameter where:
    Θ_new: Sequence of parameter settings to evaluate, challengers in this class.
    θ_inc: incumbent parameter setting, incumbent in this class.

    1  for i := 1, . . . , length(Θ_new) do
    2      θ_new ← Θ_new[i]

           STAGE -> RUN_INCUMBENT

    3      if R contains less than max_config_calls trials with configuration θ_inc then
    4          Π' ← {π'∈ Π | R contains less than or equal number of trials using θ_inc and π'
    0          than using θ_inc and any other π''∈ Π}
    5          π ← instance sampled uniformly at random from Π'
    6          s ← seed, drawn uniformly at random
    7          R ← ExecuteRun(R, θ_inc, π, s)
    8      N ← 1

           STAGE -> RUN_CHALLENGER

    9      while true do
    10         S_missing ← {instance, seed} pairs for which θ_inc was run before, but not θ_new
    11         S_torun ← random subset of S_missing of size min(N, size(S_missing))
    12         foreach (π, s) ∈ S_torun do R ← ExecuteRun(R, θ_new, π, s)
    13         S_missing ← S_missing \\ S_torun
    14         Π_common ← instances for which we previously ran both θ_inc and θ_new
    15         if c_hat(θ_new, Π_common) > c_hat(θ_inc, Π_common) then break
    16         else if S_missing = ∅ then θ_inc ← θ_new; break
    17         else N ← 2 · N
    18     if time spent in this call to this procedure exceeds t_intensify and i ≥ 2 then break
    19 return [R, θ_inc]

    Parameters
    ----------
    scenario : Scenario
    min_config_calls : int, defaults to 1
        Minimum number of trials per config (summed over all calls to intensify).
    max_config_calls : int, defaults to 2000
        Maximum number of trials per config (summed over all calls to intensify).
    min_challenger : int, defaults to 2
        Minimal number of challengers to be considered (even if time_bound is exhausted earlier).
    intensify_percentage : float, defaults to 0.5
        How much percentage of the time should configurations be intensified (evaluated on higher budgets or
        more instances). This parameter is accessed in the SMBO class.
    race_against : Configuration | None, defaults to none
        If incumbent changes, race this configuration always against new incumbent.  Prevents sometimes
        over-tuning.
    seed : int | None, defaults to none
    """

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

        # Intensify percentage must be between 0 and 1
        assert intensify_percentage >= 0.0 and intensify_percentage <= 1.0

        super().__init__(
            scenario=scenario,
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            min_challenger=min_challenger,
            seed=seed,
        )

        # General attributes
        self._race_against = race_against

        if race_against is not None and race_against.origin is None:
            assert self._race_against is not None
            if race_against == scenario.configspace.get_default_configuration():
                self._race_against.origin = "Default"
            else:
                logger.warning(
                    "The passed configuration to the intensifier was not specified with an origin. "
                    "The origin is set to `Unknown`."
                )
                self._race_against.origin = "Unknown"

        self._elapsed_time = 0.0

        # Stage variables
        # the intensification procedure is divided into 4 'stages':
        # 0. Run 1st configuration (only in the 1st run when incumbent=None)
        # 1. Add incumbent run
        # 2. Race challenger
        # 3. Race against configuration for a new incumbent
        self._stage = IntensifierStage.RUN_FIRST_CONFIG
        self._n_iters = 0

        # Challenger related variables
        self._challenger_id = 0
        self._num_challenger_run = 0
        self._current_challenger = None
        self._continue_challenger = False
        self._configs_to_run: Iterator[Optional[Configuration]] = iter([])
        self._update_configs_to_run = True
        self._intensify_percentage = intensify_percentage
        self._last_seed_idx = -1
        self._target_function_seeds = [
            int(s) for s in self._rng.randint(low=0, high=MAXINT, size=self._max_config_calls)
        ]

        # Racing related variables
        self._to_run: list[InstanceSeedBudgetKey] = []
        self._N = -1

    @property
    def intensify_percentage(self) -> float:
        """How much percentage of the time should configurations be intensified (evaluated on higher budgets or
        more instances). This parameter is accessed in the SMBO class.
        """
        return self._intensify_percentage

    @property
    def uses_seeds(self) -> bool:  # noqa: D102
        return True

    @property
    def uses_budgets(self) -> bool:  # noqa: D102
        return False

    @property
    def uses_instances(self) -> bool:  # noqa: D102
        if self._instances == [None]:
            return False

        return True

    def get_target_function_seeds(self) -> list[int]:  # noqa: D102
        if self._deterministic:
            return [0]
        else:
            return self._target_function_seeds

    def get_target_function_budgets(self) -> list[float | None]:  # noqa: D102
        return [None]

    def get_target_function_instances(self) -> list[str | None]:  # noqa: D102
        if self._instances == [None] or None in self._instances:
            return [None]

        instances: list[str | None] = []
        for instance in self._instances:
            if instance is not None:
                instances.append(instance)

        return instances

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        race_against: dict | None = None
        if self._race_against is not None:
            race_against = self._race_against.get_dictionary()

        meta = super().meta
        meta.update(
            {
                "race_against": race_against,
                "intensify_percentage": self.intensify_percentage,
            }
        )

        return meta

    def get_next_trial(
        self,
        challengers: list[Configuration] | None,
        incumbent: Configuration,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
        n_workers: int = 1,
    ) -> tuple[TrialInfoIntent, TrialInfo]:
        """This procedure is in charge of generating a TrialInfo object to comply with lines 7 (in
        case stage is stage == RUN_INCUMBENT) or line 12 (In case of stage == RUN_CHALLENGER).

        A TrialInfo object encapsulates the necessary information for a worker to execute the job, nevertheless, a
        challenger is not always available. This could happen because no more configurations are available or the new
        configuration to try was already executed.

        To circumvent this, an intent is also returned:

        * intent == RUN: Run the TrialInfo object (Normal Execution).
        * intent == SKIP: Skip this iteration. No challenger is available. In particular because challenger is the same
            as incumbent

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
        if n_workers > 1:
            raise ValueError("The selected intensifier does not support more than 1 worker.")

        # If this function is called, it means the iteration is
        # not complete (we can be starting a new iteration, or re-running a
        # challenger due to line 17). We evaluate if a iteration is complete or not
        # via _process_results
        self._iteration_done = False

        # In case a crash happens, and FirstRunCrashedException prevents a
        # failure, revert back to running the incumbent
        # Challenger case is by construction ok, as there is no special
        # stage for its processing
        if self._stage == IntensifierStage.PROCESS_FIRST_CONFIG_RUN:
            self._stage = IntensifierStage.RUN_FIRST_CONFIG
        elif self._stage == IntensifierStage.PROCESS_INCUMBENT_RUN:
            self._stage = IntensifierStage.RUN_INCUMBENT

        # if first ever run, then assume current challenger to be the incumbent
        # Because this is the first ever run, we need to sample a new challenger
        # This new challenger is also assumed to be the incumbent
        if self._stage == IntensifierStage.RUN_FIRST_CONFIG:
            if incumbent is None:
                logger.info("No incumbent provided in the first run. Sampling a new challenger...")
                challenger, self._new_challenger = self.get_next_challenger(
                    challengers=challengers,
                    get_next_configurations=get_next_configurations,
                )
                incumbent = challenger
            else:
                inc_trials = runhistory.get_trials(incumbent, only_max_observed_budget=True)
                if len(inc_trials) > 0:
                    logger.debug("Skipping RUN_FIRST_CONFIG stage since incumbent has already been ran.")
                    self._stage = IntensifierStage.RUN_INCUMBENT

        # LINES 3-7
        if self._stage in [IntensifierStage.RUN_FIRST_CONFIG, IntensifierStage.RUN_INCUMBENT]:
            # Line 3
            # A modified version, that not only checks for max_config_calls
            # but also makes sure that there are runnable instances, that is, instances has not been exhausted
            inc_trials = runhistory.get_trials(incumbent, only_max_observed_budget=True)

            # Line 4
            pending_instances = self._get_pending_instances(incumbent, runhistory)
            if pending_instances and len(inc_trials) < self._max_config_calls:
                # Lines 5-7
                instance, seed = self._get_next_instance(pending_instances)

                return TrialInfoIntent.RUN, TrialInfo(
                    config=incumbent,
                    instance=instance,
                    seed=seed,
                    budget=None,
                )
            else:
                # This point marks the transitions from lines 3-7 to 8-18
                logger.debug("No further instance-seed pairs for incumbent available.")

                self._stage = IntensifierStage.RUN_CHALLENGER

        # Understand who is the active challenger.
        if self._stage == IntensifierStage.RUN_BASIS:
            # if in RUN_BASIS stage,
            # return the basis configuration (i.e., `race_against`)
            logger.debug("Race against default configuration after incumbent change.")
            challenger = self._race_against
        elif self._current_challenger and self._continue_challenger:
            # if the current challenger could not be rejected,
            # it is run again on more instances
            challenger = self._current_challenger
        else:
            # Get a new challenger if all instance/pairs have been completed. Else return the currently running
            # challenger
            challenger, self._new_challenger = self.get_next_challenger(
                challengers=challengers,
                get_next_configurations=get_next_configurations,
            )

        # No new challengers are available for this iteration, move to the next iteration. This can only happen
        # when all configurations for this iteration are exhausted and have been run in all proposed instance/pairs.
        if challenger is None:
            return TrialInfoIntent.SKIP, TrialInfo(
                config=None,
                instance=None,
                seed=None,
                budget=None,
            )

        # Skip the iteration if the challenger was previously run
        if challenger == incumbent and self._stage == IntensifierStage.RUN_CHALLENGER:
            self._challenger_same_as_incumbent = True
            logger.debug("Challenger was the same as the current incumbent. Challenger is skipped.")
            return TrialInfoIntent.SKIP, TrialInfo(
                config=None,
                instance=None,
                seed=None,
                budget=None,
            )

        logger.debug("Intensify on %s.", challenger.get_dictionary())
        if hasattr(challenger, "origin"):
            logger.debug("Configuration origin: %s", challenger.origin)

        if self._stage in [IntensifierStage.RUN_CHALLENGER, IntensifierStage.RUN_BASIS]:
            if not self._to_run:
                self._to_run = self._get_missing_instances(
                    incumbent=incumbent, challenger=challenger, runhistory=runhistory, N=self._N
                )

            # If there is no more configs to run in this iteration, or no more
            # time to do so, change the current stage base on how the current
            # challenger performs as compared to the incumbent. This is done
            # via _process_racer_results
            if len(self._to_run) == 0:
                # Nevertheless, if there are no more instances to run,
                # we might need to comply with line 17 and keep running the
                # same challenger. In this case, if there is not enough information
                # to decide if the challenger is better/worst than the incumbent,
                # line 17 doubles the number of instances to run.
                logger.debug("No further trials for challenger possible.")
                self._process_racer_results(
                    challenger=challenger,
                    incumbent=incumbent,
                    runhistory=runhistory,
                )

                # Request SMBO to skip this run. This function will
                # be called again, after the _process_racer_results
                # has updated the intensifier stage
                return TrialInfoIntent.SKIP, TrialInfo(
                    config=None,
                    instance=None,
                    seed=None,
                    budget=None,
                )

            else:
                # Lines 8-11
                incumbent, instance, seed = self._get_next_racer(
                    challenger=challenger,
                    incumbent=incumbent,
                    runhistory=runhistory,
                )

                # Line 12
                return TrialInfoIntent.RUN, TrialInfo(
                    config=challenger,
                    instance=instance,
                    seed=seed,
                    budget=None,
                )
        else:
            raise ValueError("No valid stage found.")

    def process_results(
        self,
        trial_info: TrialInfo,
        trial_value: TrialValue,
        incumbent: Configuration | None,
        runhistory: RunHistory,
        time_bound: float,
        log_trajectory: bool = True,
    ) -> tuple[Configuration, float]:
        """The intensifier stage will be updated based on the results/status of a configuration execution.
        During intensification, the following can happen:

        * Challenger raced against incumbent.
        * Also, during a challenger run, a capped exception can be triggered, where no racer post processing is needed.
        * A run on the incumbent for more confidence needs to be processed, IntensifierStage.PROCESS_INCUMBENT_RUN.
        * The first run results need to be processed (PROCESS_FIRST_CONFIG_RUN).

        At the end of any run, checks are done to move to a new iteration.

        Parameters
        ----------
        trial_info : TrialInfo
        trial_value: TrialValue
        incumbent : Configuration | None
            Best configuration seen so far.
        runhistory : RunHistory
        time_bound : float
            Time [sec] available to perform intensify.
        log_trajectory: bool
            Whether to log changes of incumbents in the trajectory.

        Returns
        -------
        incumbent: Configuration
            Current (maybe new) incumbent configuration.
        incumbent_costs: float | list[float]
            Empirical cost(s) of the incumbent configuration.
        """
        if self._stage == IntensifierStage.PROCESS_FIRST_CONFIG_RUN:
            if incumbent is None:
                logger.info("First run and no incumbent provided. Challenger is assumed to be the incumbent.")
                incumbent = trial_info.config

        if self._stage in [
            IntensifierStage.PROCESS_INCUMBENT_RUN,
            IntensifierStage.PROCESS_FIRST_CONFIG_RUN,
        ]:
            self._num_trials += 1
            self._process_incumbent(
                incumbent=incumbent,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )
        else:
            self._num_trials += 1
            self._num_challenger_run += 1
            incumbent = self._process_racer_results(
                challenger=trial_info.config,
                incumbent=incumbent,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )

        self._elapsed_time += trial_value.time

        # check if 1 intensification run is complete - line 18
        # this is different to regular SMAC as it requires at least successful challenger run,
        # which is necessary to work on a fixed grid of configurations.
        if (
            self._stage == IntensifierStage.RUN_INCUMBENT
            and self._challenger_id >= self._min_challenger
            and self._num_challenger_run > 0
        ):
            # if self._num_trials > self._run_limit:
            #    logger.debug("Maximum number of trials for intensification reached.")
            #    self._next_iteration()
            if self._elapsed_time - time_bound >= 0:
                logger.debug(f"Wallclock time limit for intensification reached ({self._elapsed_time}/{time_bound})")
                self._next_iteration()

        inc_perf = runhistory.get_cost(incumbent)

        return incumbent, inc_perf

    def _get_next_instance(
        self,
        pending_instances: list[str | None],
    ) -> tuple[str | None, int | None]:
        """Method to extract the next instance and seed instance in which a incumbent run most be evaluated."""
        # Line 5 - and avoid https://github.com/numpy/numpy/issues/10791
        _idx = self._rng.choice(len(pending_instances))
        next_instance = pending_instances[_idx]

        # Line 6
        if self._deterministic:
            next_seed = self.get_target_function_seeds()[0]
        else:
            self._last_seed_idx += 1
            next_seed = self.get_target_function_seeds()[self._last_seed_idx]

        # Line 7
        logger.debug(f"Add run of incumbent for instance = {next_instance}")
        if self._stage == IntensifierStage.RUN_FIRST_CONFIG:
            self._stage = IntensifierStage.PROCESS_FIRST_CONFIG_RUN
        else:
            self._stage = IntensifierStage.PROCESS_INCUMBENT_RUN

        return next_instance, next_seed

    def _get_pending_instances(
        self,
        incumbent: Configuration,
        runhistory: RunHistory,
    ) -> list[str | None]:
        """Implementation of line 4 of Intensification. This method queries the incumbent trials in the runhistory and
        return the pending instances if any is available.
        """
        # Line 4
        # Find all instances that have the most trials on the inc
        inc_trials = runhistory.get_trials(incumbent, only_max_observed_budget=True)

        # Now we select and count the instances and sort them
        inc_inst = [s.instance for s in inc_trials]
        inc_inst_counter = list(Counter(inc_inst).items())
        inc_inst_counter.sort(key=lambda x: x[1], reverse=True)

        try:
            max_trials = inc_inst_counter[0][1]
        except IndexError:
            logger.debug("No run for incumbent found.")
            max_trials = 0

        # Those are the instances with run the most
        inc_inst = [x[0] for x in inc_inst_counter if x[1] == max_trials]

        # We basically sort the instances by their name
        available_insts: list[str | None] = list(set(self._instances) - set(inc_inst))
        if None in available_insts:
            assert len(available_insts) == 1
            available_insts = [None]
        else:
            available_insts = sorted(available_insts)  # type: ignore

        # If all instances were used n times, we can pick an instances from the complete set again
        if not self._deterministic and not available_insts:
            available_insts = self._instances

        return available_insts

    def _process_incumbent(
        self,
        incumbent: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> None:
        """Method to process the results of a challenger that races an incumbent."""
        # output estimated performance of incumbent
        inc_trials = runhistory.get_trials(incumbent, only_max_observed_budget=True)
        inc_perf = runhistory.get_cost(incumbent)
        format_value = format_array(inc_perf)
        logger.info(f"Updated estimated cost of incumbent on {len(inc_trials)} trials: {format_value}")

        # if running first configuration, go to next stage after 1st run
        if self._stage in [
            IntensifierStage.RUN_FIRST_CONFIG,
            IntensifierStage.PROCESS_FIRST_CONFIG_RUN,
        ]:
            self._stage = IntensifierStage.RUN_INCUMBENT
            self._next_iteration()
        else:
            # Termination condition; after each run, this checks
            # whether further trials are necessary due to min_config_calls
            if len(inc_trials) >= self._min_config_calls or len(inc_trials) >= self._max_config_calls:
                self._stage = IntensifierStage.RUN_CHALLENGER
            else:
                self._stage = IntensifierStage.RUN_INCUMBENT

        self._compare_configs(
            incumbent=incumbent, challenger=incumbent, runhistory=runhistory, log_trajectory=log_trajectory
        )

    def _get_next_racer(
        self,
        challenger: Configuration,
        incumbent: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> tuple[Configuration, str | None, int | None]:
        """Method to return the next config setting to aggressively race challenger against
        incumbent. Returns either the challenger or incumbent as new incumbent alongside with the
        instance and seed.
        """
        # By the time this function is called, the run history might
        # have shifted. Re-populate the list if necessary
        if not self._to_run:
            # Lines 10/11
            self._to_run = self._get_missing_instances(
                incumbent=incumbent, challenger=challenger, runhistory=runhistory, N=self._N
            )

        # Run challenger on all <instance, seed> to run
        instance_seed_budget_key = self._to_run.pop()
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
        """
        chal_trials = runhistory.get_trials(challenger, only_max_observed_budget=True)
        chal_perf = runhistory.get_cost(challenger)

        # If all <instance, seed> have been run, compare challenger performance
        if not self._to_run:
            new_incumbent = self._compare_configs(
                incumbent=incumbent,
                challenger=challenger,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )

            # Update intensification stage
            if new_incumbent == incumbent:
                # move on to the next iteration
                self._stage = IntensifierStage.RUN_INCUMBENT
                self._continue_challenger = False
                logger.debug(
                    "Estimated cost of challenger on %d trials: %.4f, but worse than incumbent.",
                    len(chal_trials),
                    chal_perf,
                )
            elif new_incumbent == challenger:
                # New incumbent found
                incumbent = challenger
                self._continue_challenger = False
                # compare against basis configuration if provided, else go to next iteration
                if self._race_against and self._race_against != challenger:
                    self._stage = IntensifierStage.RUN_BASIS
                else:
                    self._stage = IntensifierStage.RUN_INCUMBENT
                logger.debug(
                    "Estimated cost of challenger on %d trials (%.4f) becomes new incumbent.",
                    len(chal_trials),
                    chal_perf,
                )

            else:  # Line 17
                # Challenger is not worse, continue
                self._N = 2 * self._N
                self._continue_challenger = True
                logger.debug(
                    "Estimated cost of challenger on %d trials (%.4f). Adding %d trials to the queue.",
                    len(chal_trials),
                    chal_perf,
                    self._N / 2,
                )
        else:
            logger.debug(
                "Estimated cost of challenger on %d trials (%.4f). Still %d trials to go (continue racing).",
                len(chal_trials),
                chal_perf,
                len(self._to_run),
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
        """
        # Get next instances left for the challenger
        # Line 8
        inc_inst_seeds = set(runhistory.get_trials(incumbent, only_max_observed_budget=True))
        chall_inst_seeds = set(runhistory.get_trials(challenger, only_max_observed_budget=True))

        # Line 10
        missing_trials = list(sorted(inc_inst_seeds - chall_inst_seeds))

        # Line 11
        self._rng.shuffle(missing_trials)  # type: ignore

        if N < 0:
            raise ValueError("Argument N must not be smaller than zero, but is %s." % str(N))

        to_run = missing_trials[: min(N, len(missing_trials))]
        missing_trials = missing_trials[min(N, len(missing_trials)) :]

        return to_run

    def get_next_challenger(
        self,
        challengers: list[Configuration] | None,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
    ) -> tuple[Configuration | None, bool]:
        """This function returns the next challenger, that should be exercised though lines 8-17.

        It does so by populating `_configs_to_run`, which is a pool of configuration from which the racer will sample.
        Each configuration within `_configs_to_run`, will be intensified on different instances/seed registered in
        `self._to_run` as stated in line 11.

        A brand new configuration should only be sampled, after all `self._to_run` instance seed pairs are exhausted.

        This method triggers a call to `_next_iteration` if there are no more configurations to run, for the current
        intensification loop. This marks the transition to Line 2, where a new configuration to intensify will be drawn
        from epm/initial challengers.

        Parameters
        ----------
        challengers : list[Configuration] | None, defaults to None
            Promising configurations.
        get_next_configurations : Callable[[], Iterator[Configuration]] | None
            Function that generates next configurations to use for racing.

        Returns
        -------
        configuration : Configuration | None
            The configuration of the selected challenger.
        new_challenger : bool
            If the configuration is a new challenger.
        """
        # Select new configuration when entering 'race challenger' stage or for the first run
        if not self._current_challenger or (self._stage == IntensifierStage.RUN_CHALLENGER and not self._to_run):
            # This is a new intensification run, get the next list of configurations to run
            if self._update_configs_to_run:
                configs_to_run = self._generate_challengers(
                    challengers=challengers, get_next_configurations=get_next_configurations
                )
                self._configs_to_run = cast(Iterator[Optional[Configuration]], configs_to_run)
                self._update_configs_to_run = False

            # Pick next configuration from the generator
            try:
                challenger = next(self._configs_to_run)
            except StopIteration:
                # Out of challengers for the current iteration, start next incumbent iteration
                self._next_iteration()
                return None, False

            if challenger:
                # Reset instance index for the new challenger
                self._challenger_id += 1
                self._current_challenger = challenger
                self._N = max(1, self._min_config_calls)
                self._to_run = []

            return challenger, True

        # Return currently running challenger
        return self._current_challenger, False

    def _generate_challengers(
        self,
        challengers: list[Configuration] | None,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
    ) -> Iterator[Optional[Configuration]]:
        """Retuns a sequence of challengers to use in intensification. If challengers are not
        provided, then optimizer will be used to generate the challenger list.
        """
        chall_gen: Iterator[Optional[Configuration]]
        if challengers:
            # Iterate over challengers provided
            logger.debug("Using provided challengers...")
            chall_gen = iter(challengers)
        elif get_next_configurations:
            # Generating challengers on-the-fly if optimizer is given
            logger.debug("Generating new challenger from optimizer...")
            chall_gen = get_next_configurations()
        else:
            raise ValueError("No configuration function provided. Can not generate challenger!")

        return chall_gen

    def _next_iteration(self) -> None:
        """Updates tracking variables at the end of an intensification run."""
        assert self._stats

        # Track iterations
        self._n_iters += 1
        self._iteration_done = True
        self._configs_to_run = iter([])
        self._update_configs_to_run = True

        # Reset for a new iteration
        self._num_trials = 0
        self._num_challenger_run = 0
        self._challenger_id = 0
        self._elapsed_time = 0.0

        self._stats.update_average_configs_per_intensify(n_configs=self._challenger_id)
