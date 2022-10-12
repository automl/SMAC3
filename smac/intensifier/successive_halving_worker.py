from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
from ConfigSpace import Configuration

from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.intensifier.successive_halving import SuccessiveHalving
from smac.runhistory import StatusType, TrialInfo, TrialInfoIntent, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class SuccessiveHalvingWorker(AbstractIntensifier):
    """This is the worker class for Successive Halving.

    Warning
    -------
    Do not use this class as stand-alone.

    Parameters
    ----------
    successive_halving : SuccessiveHalving
        The controller of the instance.
    identifier : int, defaults to 0
        Adds a numerical identifier on the instance. Used for debug and tagging logger messages properly.
    _all_budgets : np.ndarray | None, defaults to None
        Used internally when HB uses SH as a sub-routing.
    _n_configs_in_stage : np.ndarray | None, defaults to None
        Used internally when HB uses SH as a sub-routing.
    _min_budget : float | None, defaults to None
        Overwrites the budget from the controller instance if it is not none.
    _max_budget : float | None, defaults to None
        Overwrites the budget from the controller instance if it is not none.
    """

    def __init__(
        self,
        successive_halving: SuccessiveHalving,
        identifier: int = 0,
        _all_budgets: list[float] | None = None,
        _n_configs_in_stage: np.ndarray | None = None,
        _min_budget: float | None = None,
        _max_budget: float | None = None,
    ) -> None:
        super().__init__(
            scenario=successive_halving._scenario,
            min_challenger=successive_halving._min_challenger,
            seed=successive_halving._seed,
        )

        self._successive_halving = successive_halving
        self._stats = successive_halving._stats
        self._first_run = True
        self._identifier = identifier
        self._logger = get_logger(f"{__name__}.{identifier}")

        self._min_budget = _min_budget if _min_budget is not None else successive_halving._min_budget
        self._max_budget = _max_budget if _max_budget is not None else successive_halving._max_budget

        if _all_budgets is not None and _n_configs_in_stage is not None:
            # Assert we use the given numbers to avoid rounding issues, see #701
            self._all_budgets = _all_budgets
            self._n_configs_in_stage = _n_configs_in_stage
        else:
            eta = successive_halving._eta

            # Max sh iterations and n initial challengers are depending on min and max budget
            # Since min and max budget can be changed by the user, we need to update
            # Pre-computing stuff for SH
            (
                self._max_sh_iterations,
                self._n_initial_challengers,
                self._all_budgets,
            ) = self._successive_halving.calculate_budgets(self._min_budget, self._max_budget)

            # Number of challengers to consider in each stage
            linspace = -np.linspace(0, self._max_sh_iterations, self._max_sh_iterations + 1)
            n_configs_in_stage = self._n_initial_challengers * np.power(eta, linspace)
            self._n_configs_in_stage = np.array(np.round(n_configs_in_stage), dtype=int).tolist()

        self._sh_iters = 0
        self._stage = 0
        self._configs_to_run: list[Configuration] = []

        # Current instance index tracks two things. Configurations that are to be launched,
        # That is config A needs to run in 3 instances/seed pairs, then current_instance_indices should
        # track this. But then, if a new configuration is added in the case of parallelism
        # a new separate current_instance_indices needs to be started.
        # The indices normally are of type int, but np.inf is used to indicate to not further
        # launch instances for this configuration, hence the type is Union[int, float].
        self._current_instance_indices: dict[Configuration, int | float] = {}
        self._running_challenger = None
        self._success_challengers: set[Configuration] = set()
        self._do_not_advance_challengers: set[Configuration] = set()
        self._failed_challengers: set[Configuration] = set()
        self._failed_challenger_offset = 0
        self._new_challenger = False

        # Track which configs were launched. This will allow to have an extra check to make sure
        # that a successive halver deals only with the configs it launched,
        # but also allows querying the status of the configs via the run history.
        # In other works, the run history is agnostic of the origin of the configurations,
        # that is, which successive halving instance created it. The RunInfo object
        # is aware of this information, and for parallel execution, the routing of
        # finish results is expected to use this information.
        # Nevertheless, the common object among SMBO/intensifier, which is the
        # run history, does not have this information and so we track locally. That way,
        # when we access the complete list of configs from the run history, we filter
        # the ones launched by the current succesive halver using self._run_tracker
        self._run_tracker: dict[tuple[Configuration, str | None, int | None, float | None], bool] = {}

        # Those general arguments are initialized in the parent class
        # self._num_trials = 0
        # self._challenger_id = 0
        # self._iteration_done = False
        # self._target_function_time = 0

    @property
    def stage(self) -> int:
        """The current stage of the worker."""
        return self._stage

    @property
    def uses_seeds(self) -> bool:  # noqa: D102
        return self._successive_halving.uses_seeds

    @property
    def uses_budgets(self) -> bool:  # noqa: D102
        return self._successive_halving.uses_budgets

    @property
    def uses_instances(self) -> bool:  # noqa: D102
        return self._successive_halving.uses_instances

    def get_target_function_seeds(self) -> list[int]:  # noqa: D102
        return self._successive_halving.get_target_function_seeds()

    def get_target_function_budgets(self) -> list[float | None]:  # noqa: D102
        return self._successive_halving.get_target_function_budgets()

    def get_target_function_instances(self) -> list[str | None]:  # noqa: D102
        return self._successive_halving.get_target_function_instances()

    def process_results(
        self,
        trial_info: TrialInfo,
        trial_value: TrialValue,
        incumbent: Configuration | None,
        runhistory: RunHistory,
        time_bound: float,
        log_trajectory: bool = True,
    ) -> tuple[Configuration, float]:  # noqa: D102
        # Mark the fact that we processed this configuration
        self._run_tracker[(trial_info.config, trial_info.instance, trial_info.seed, trial_info.budget)] = True

        # If The incumbent is None and it is the first run, we use the challenger
        if not incumbent and self._first_run:
            # We already displayed this message before
            # self._logger.info("First run and no incumbent provided. Challenger is assumed to be the incumbent.")
            incumbent = trial_info.config
            self._first_run = False

        # In a serial run, if we have to CAP a run, then we stop launching
        # more configurations for this run.
        # In the context of parallelism, we launch the instances proactively
        # The fact that self._current_instance_indices[trial_info.config] is np.inf means
        # that no more instances will be launched for the current config, so we
        # can add a check to make sure that if we are capping, this makes sense
        # for the active challenger
        # if result.status == StatusType.CAPPED and trial_info.config == self._running_challenger:
        #    self._current_instance_indices[trial_info.config] = np.inf
        # else:
        self._target_function_time += trial_value.time
        self._num_trials += 1

        # 0: Before moving to a new stage, we have to complete M x N tasks, where M is the
        # total number of configurations evaluated in N instance/seed pairs.
        # The last active configuration is stored in self.running challengers is the M
        # configuration, and to get to this point, we have already submitted tasks
        # for (M - 1) configurations on N instances seed-pairs. The status of such
        # (M - 1) * N tasks is tracked in self._run_tracker, that has a value of False
        # if not processed and true if such task has been processed.
        # This stage is complete only if all tasks have been launched and all of the
        # already launched tasks are processed.

        # 1: We first query if we have launched everything already (All M * N tasks)
        all_config_inst_seed_launched = self._all_config_instance_seed_pairs_launched(
            runhistory=runhistory,
            activate_configuration_being_intensified=self._running_challenger,
        )

        # 2: Then we get the already submitted tasks (that is, proposed by get_next_trials),
        # that have not yet been processed process_results
        all_config_inst_seeds_processed = len([v for v in self._run_tracker.values() if not v]) <= 0

        # 3: Then the total number of remaining task before we can conclude this stage
        # is calculated by taking into account point 2 and 3 above
        is_stage_done = all_config_inst_seed_launched and all_config_inst_seeds_processed

        # ADDED IN SMAC 2.0: Makes sure we can not call `tell` method without `ask` first
        if trial_info.config not in self._current_instance_indices:
            raise RuntimeError("Successive Halving does not support calling `tell` method without calling `ask` first.")

        # adding challengers to the list of evaluated challengers
        #  - Stop: CAPPED/CRASHED/TIMEOUT/MEMOUT/DONOTADVANCE (!= SUCCESS)
        #  - Advance to next stage: SUCCESS
        # curr_challengers is a set, so "at least 1" success can be counted by set addition (no duplicates)
        # If a configuration is successful, it is added to curr_challengers.
        # if it fails it is added to fail_challengers.
        if np.isfinite(self._current_instance_indices[trial_info.config]) and trial_value.status == StatusType.SUCCESS:
            self._success_challengers.add(trial_info.config)  # successful configs
        elif (
            np.isfinite(self._current_instance_indices[trial_info.config])
            and trial_value.status == StatusType.DONOTADVANCE
        ):
            self._do_not_advance_challengers.add(trial_info.config)
        else:
            self._failed_challengers.add(trial_info.config)  # capped/crashed/do not advance configs

        # We need to update the incumbent if this config we are processing
        # completes all scheduled instance-seed pairs.
        # Here, a config/seed/instance is going to be processed for the first time
        # (it has been previously scheduled by get_next_trial and marked False, indicating
        # that it has not been processed yet. Entering process_results() this config/seed/instance
        # is marked as TRUE as an indication that it has finished and should be processed)
        # so if all configurations runs are marked as TRUE it means that this new config
        # was the missing piece to have everything needed to compare against the incumbent
        update_incumbent = all([v for k, v in self._run_tracker.items() if k[0] == trial_info.config])

        # get incumbent if all instances have been evaluated
        if is_stage_done or update_incumbent:
            incumbent = self._compare_configs(
                challenger=trial_info.config,
                incumbent=incumbent,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )

        if is_stage_done:
            self._logger.info(
                "Finished Successive Halving iteration-step %d-%d with "
                "budget [%.2f / %d] and %d evaluated challenger(s)."
                % (
                    self._sh_iters + 1,
                    self._stage + 1,
                    self._all_budgets[self._stage],
                    self._max_budget,
                    self._n_configs_in_stage[self._stage],
                )
            )

            self._update_stage(runhistory=runhistory)

        # Get incumbent cost
        inc_perf = runhistory.get_cost(incumbent)

        return incumbent, inc_perf

    def get_next_trial(
        self,
        challengers: list[Configuration] | None,
        incumbent: Configuration,
        get_next_configurations: Callable[[], Iterator[Configuration]] | None,
        runhistory: RunHistory,
        repeat_configs: bool = True,
        n_workers: int = 1,
    ) -> tuple[TrialInfoIntent, TrialInfo]:
        """Selects which challenger to use based on the iteration stage and set the iteration
        parameters. First iteration will choose configurations from the function ``get_next_configurations`` or input
        challengers, while the later iterations pick top configurations from the previously selected
        challengers in that iteration.

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
            self._logger.warning(
                "Consider using ParallelSuccesiveHalving instead of "
                "SuccesiveHalving. The later will halt on each stage "
                "transition until all configs for the current stage are completed."
            )

        sh = self._successive_halving

        # In the case of multiprocessing, we have runs in Running stage, which have not
        # been processed via process_results(). get_next_trial() is called agnostically by
        # smbo. To prevent launching more configs, than the ones needed, we query if
        # there is room for more configurations, else we wait for process_results()
        # to trigger a new stage
        if self._all_config_instance_seed_pairs_launched(runhistory, self._running_challenger):
            return TrialInfoIntent.WAIT, TrialInfo(
                config=None,
                instance=None,
                seed=None,
                budget=None,
                source=self._identifier,
            )

        # Sampling from next challenger marks the beginning of a new iteration
        self._iteration_done = False

        current_budget = self._all_budgets[self._stage]

        # If all instances have been executed, then reset and move on to next config
        if sh._instance_as_budget:
            previous_budget = int(self._all_budgets[self._stage - 1]) if self._stage > 0 else 0
            n_insts = int(current_budget) - previous_budget
        else:
            n_insts = len(sh._instance_seed_pairs)

        # The instances remaining tell us, per configuration, how many instances we
        # have suggested to SMBO
        n_insts_remaining = n_insts
        if self._running_challenger is not None:
            n_insts_remaining = n_insts - self._current_instance_indices[self._running_challenger]

        # If there are instances pending, finish running configuration
        if self._running_challenger and n_insts_remaining > 0:
            challenger = self._running_challenger
            new_challenger = False
        else:
            # Select next configuration
            if self._stage == 0:
                # first stage, so sample from configurations/chooser provided
                challenger = self._next_challenger(
                    challengers=challengers,
                    get_next_configurations=get_next_configurations,
                    runhistory=runhistory,
                    repeat_configs=repeat_configs,
                )
                if challenger is None:
                    # If no challenger was sampled from the EPM or
                    # initial challengers, it might mean that the EPM
                    # is proposing a configuration that is currently running.
                    # There is a filtering on the above _next_challenger to return
                    # None if the proposed config us already in the run history
                    # To get a new config, we wait for more data
                    return TrialInfoIntent.WAIT, TrialInfo(
                        config=None,
                        instance=None,
                        seed=None,
                        budget=None,
                        source=self._identifier,
                    )

                new_challenger = True
            else:
                # Sample top configs from previously sampled configurations
                try:
                    challenger = self._configs_to_run.pop(0)
                    new_challenger = False
                except IndexError:
                    # self._configs_to_run is populated via update_stage,
                    # which is triggered after the completion of a run
                    # If by there are no more configs to run (which is the case
                    # if we run into a IndexError),
                    return TrialInfoIntent.SKIP, TrialInfo(
                        config=None,
                        instance=None,
                        seed=None,
                        budget=None,
                        source=self._identifier,
                    )

            if challenger:
                # We see a challenger for the first time, so no
                # instance has been launched
                self._current_instance_indices[challenger] = 0
                self._challenger_id += 1
                self._running_challenger = challenger

        if (incumbent is None or incumbent == challenger) and self._first_run:
            self._logger.info("First run and no incumbent provided. Challenger is assumed to be the incumbent.")
            incumbent = challenger

        # Selecting instance-seed subset for this budget, depending on the kind of budget
        if sh._instance_as_budget:
            previous_budget = int(self._all_budgets[self._stage - 1]) if self._stage > 0 else 0
            current_instances = sh._instance_seed_pairs[int(previous_budget) : int(current_budget)]
        else:
            current_instances = sh._instance_seed_pairs

        self._logger.debug(f"Running challenger: {challenger}")

        # run the next instance-seed pair for the given configuration
        instance, seed = current_instances[self._current_instance_indices[challenger]]  # type: ignore[index]
        # At this point self._current_instance_indices[challenger] will still be an integer and might
        # be marked LATER with np.inf, so ignore mypy error.

        # For testing purposes, this attribute highlights whether a
        # new challenger is proposed or not. Not required from a functional
        # perspective
        self._new_challenger = new_challenger

        budget = None if sh._instance_as_budget else current_budget
        self._run_tracker[(challenger, instance, seed, budget)] = False

        # self._current_instance_indices Tell us our current instance to be run. The upcoming return
        # will launch a challenger on a given instance/seed/pair. The next time this function
        # is called, we will like to run self._current_instance_indices + 1 for this configuration
        self._current_instance_indices[challenger] += 1

        return TrialInfoIntent.RUN, TrialInfo(
            config=challenger,
            instance=instance,
            seed=seed,
            budget=budget,
            source=self._identifier,
        )

    def _update_stage(self, runhistory: RunHistory) -> None:
        """Updates tracking information for a new stage/iteration and update statistics. This method
        is called to initialize stage variables and after all configurations of a Successive Halving
        stage are completed.
        """
        self._stage += 1

        # Only uncapped challengers are considered valid for the next iteration
        valid_challengers = list(
            (self._success_challengers | self._do_not_advance_challengers) - self._failed_challengers
        )

        if self._stage < len(self._all_budgets) and len(valid_challengers) > 0:
            # If this is the next stage in same iteration,
            # use top 'k' from the evaluated configurations for next iteration

            # Determine 'k' for the next iteration - at least 1
            next_n_chal = int(max(1, self._n_configs_in_stage[self._stage]))

            # Selecting the top 'k' challengers for the next iteration
            configs_to_run = self._top_k(configs=valid_challengers, runhistory=runhistory, k=next_n_chal)
            self._configs_to_run = [
                config for config in configs_to_run if config not in self._do_not_advance_challengers
            ]

            # If some runs were capped, top_k returns less than the required configurations
            # to handle that, we keep track of how many configurations are missing
            # (since they are technically failed here too)
            missing_challengers = int(self._n_configs_in_stage[self._stage]) - len(self._configs_to_run)
            if missing_challengers > 0:
                self._failed_challenger_offset = missing_challengers
            else:
                self._failed_challenger_offset = 0

            if next_n_chal == missing_challengers:
                next_stage = True
                self._logger.info(
                    "Successive Halving iteration-step: %d-%d with "
                    "budget [%.2f / %d] - expected %d new challenger(s), but "
                    "no configurations propagated to the next budget.",
                    self._sh_iters + 1,
                    self._stage + 1,
                    self._all_budgets[self._stage],
                    self._max_budget,
                    self._n_configs_in_stage[self._stage],
                )
            else:
                next_stage = False
        else:
            next_stage = True

        if next_stage:
            # Update stats for the prev iteration
            assert self._stats
            self._stats.update_average_configs_per_intensify(n_configs=self._challenger_id)

            # Reset stats for the new iteration
            self._sh_iters += 1
            self._stage = 0
            self._configs_to_run = []
            self._target_function_time = 0
            self._challenger_id = 0
            self._num_trials = 0
            self._iteration_done = True

            self._failed_challenger_offset = 0
            self._run_tracker = {}

            # Randomize instance-seed pairs per successive halving run, if user specifies
            if self._successive_halving._instance_order == "shuffle":
                self._rng.shuffle(self._instance_seed_pairs)  # type: ignore

        # To track configurations for the next stage
        self._running_challenger = None
        self._success_challengers = set()
        self._do_not_advance_challengers = set()
        self._failed_challengers = set()
        self._current_instance_indices = {}

    def _compare_configs(
        self,
        incumbent: Configuration,
        challenger: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> Configuration | None:
        """Compares the challenger with the current incumbent and returns the best configuration based
        on the given incumbent selection design.

        Returns
        -------
        configuration : Configuration | None
            The better configuration.
        """
        assert self._stats

        if self._successive_halving._instance_as_budget:
            new_incumbent = super()._compare_configs(incumbent, challenger, runhistory, log_trajectory)

            # If compare config returned none, then it is undecided. So return old incumbent.
            return incumbent if new_incumbent is None else new_incumbent

        # For real-valued budgets, compare configs based on the incumbent selection design
        current_budget = self._all_budgets[self._stage]

        # incumbent selection: best on any budget
        if self._successive_halving._incumbent_selection == "any_budget":
            new_incumbent = self._compare_configs_across_budgets(
                challenger=challenger,
                incumbent=incumbent,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )
            return new_incumbent

        # get runs for both configurations
        inc_runs = runhistory.get_trials(incumbent, only_max_observed_budget=True)
        chall_runs = runhistory.get_trials(challenger, only_max_observed_budget=True)

        if len(inc_runs) > 1:
            raise ValueError(
                "Number of incumbent runs on budget %f must not exceed 1, but is %d",
                inc_runs[0].budget,
                len(inc_runs),
            )

        if len(chall_runs) > 1:
            raise ValueError(
                "Number of challenger runs on budget %f must not exceed 1, but is %d",
                chall_runs[0].budget,
                len(chall_runs),
            )

        inc_run = inc_runs[0]
        chall_run = chall_runs[0]

        if inc_run.budget is None or chall_run.budget is None:
            raise RuntimeError("Since budgets are not used for instance optimization, this should not happen.")

        # Incumbent selection: highest budget only
        if self._successive_halving._incumbent_selection == "highest_budget":
            assert chall_run.budget is not None
            if chall_run.budget < self._max_budget:
                self._logger.debug(
                    "Challenger (budget=%.4f) has not been evaluated on the highest budget %.4f yet.",
                    chall_run.budget,
                    self._max_budget,
                )
                return incumbent

        # Incumbent selection: highest budget run so far
        if inc_run.budget > chall_run.budget:
            self._logger.debug(
                "Incumbent evaluated on higher budget than challenger (%.4f > %.4f), not changing the incumbent",
                inc_run.budget,
                chall_run.budget,
            )
            return incumbent

        if inc_run.budget < chall_run.budget:
            self._logger.debug(
                "Challenger evaluated on higher budget than incumbent (%.4f > %.4f), changing the incumbent",
                chall_run.budget,
                inc_run.budget,
            )
            if log_trajectory:
                assert self._stats
                self._stats.add_incumbent(
                    cost=runhistory.get_cost(challenger),
                    incumbent=challenger,
                    budget=current_budget,
                )

            return challenger

        # incumbent and challenger were both evaluated on the same budget, compare them based on their cost
        chall_cost = runhistory.get_cost(challenger)
        inc_cost = runhistory.get_cost(incumbent)
        if chall_cost < inc_cost:
            self._logger.info(
                "Challenger (%.4f) is better than incumbent (%.4f) on budget %.4f.",
                chall_cost,
                inc_cost,
                chall_run.budget,
            )
            self._log_incumbent_changes(incumbent, challenger)
            new_incumbent = challenger
            if log_trajectory:
                assert self._stats
                self._stats.add_incumbent(cost=chall_cost, incumbent=new_incumbent, budget=current_budget)
        else:
            self._logger.debug(
                "Incumbent (%.4f) is at least as good as the challenger (%.4f) on budget %.4f.",
                inc_cost,
                chall_cost,
                inc_run.budget,
            )

            if log_trajectory and self._stats.incumbent_changed == 0:
                assert self._stats
                self._stats.add_incumbent(
                    cost=inc_cost,
                    incumbent=incumbent,
                    budget=current_budget,
                )
            new_incumbent = incumbent

        return new_incumbent

    def _compare_configs_across_budgets(
        self,
        challenger: Configuration,
        incumbent: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> Configuration | None:
        """Compares challenger with current incumbent on any budget.

        Returns
        -------
        configuration : Configuration | None
            The better configuration.
        """
        assert self._stats
        current_budget = self._all_budgets[self._stage]

        # compare challenger and incumbent based on cost
        chall_cost = runhistory.get_min_cost(challenger)
        inc_cost = runhistory.get_min_cost(incumbent)
        if np.isfinite(chall_cost) and np.isfinite(inc_cost):
            if chall_cost < inc_cost:
                self._logger.info(
                    "Challenger (%.4f) is better than incumbent (%.4f) for any budget.",
                    chall_cost,
                    inc_cost,
                )
                self._log_incumbent_changes(incumbent, challenger)
                new_incumbent = challenger
                if log_trajectory:
                    assert self._stats
                    self._stats.add_incumbent(
                        cost=chall_cost,
                        incumbent=new_incumbent,
                        budget=current_budget,
                    )
            else:
                self._logger.debug(
                    "Incumbent (%.4f) is at least as good as the challenger (%.4f) for any budget.",
                    inc_cost,
                    chall_cost,
                )
                if log_trajectory and self._stats.incumbent_changed == 0:
                    self._stats.add_incumbent(cost=inc_cost, incumbent=incumbent, budget=current_budget)
                new_incumbent = incumbent
        else:
            self._logger.debug("Non-finite costs from run history!")
            new_incumbent = incumbent

        return new_incumbent

    def _top_k(self, configs: list[Configuration], runhistory: RunHistory, k: int) -> list[Configuration]:
        """Selects the top 'k' configurations from the given list based on their performance.

        This retrieves the performance for each configuration from the runhistory and checks
        that the highest budget they've been evaluated on is the same for each of the configurations.

        Parameters
        ----------
        configs : list[Configuration]
            List of configurations to filter from.
        runhistory : RunHistory
        k : int
            Number of configurations to select

        Returns
        -------
        list[Configuration]
            Top challenger configurations, sorted in increasing costs.
        """
        # Extracting costs for each given configuration
        config_costs = {}

        # Sample list instance-seed-budget key to act as base
        run_key = runhistory.get_trials(configs[0], only_max_observed_budget=True)
        for c in configs:
            # Ensuring that all configurations being compared are run on the same set of instance, seed & budget
            cur_run_key = runhistory.get_trials(c, only_max_observed_budget=True)

            # Move to compare set -- get_trials queries form a dictionary
            # which is not an ordered structure. Some queries to that dictionary returned unordered
            # list which wrongly trigger the below if
            if set(cur_run_key) != set(run_key):
                raise ValueError(
                    "Can not compare configs that were run on different instances-seeds-budgets: \n"
                    f"{run_key} and\n{cur_run_key}"
                )
            config_costs[c] = runhistory.get_cost(c)

        configs_sorted = [k for k, v in sorted(config_costs.items(), key=lambda item: item[1])]
        # Select top configurations only
        top_configs = configs_sorted[:k]
        return top_configs

    def _all_config_instance_seed_pairs_launched(
        self,
        runhistory: RunHistory,
        activate_configuration_being_intensified: Configuration | None,
    ) -> bool:
        """When running SH, M configs might require N instances. Before moving to the next stage, we
        need to make sure that tasks (each of the MxN jobs) are launched.

        This function returns a true if any M configs are pending or if N instance/seed are
        still remaining.

        Parameters
        ----------
        runhistory : RunHistory
        activate_configuration_being_intensified: Configuration | None
            The last configuration being actively processes by this intensifier.

        Returns
        -------
        pending : bool
            Whether a instance/pair of any of the M configurations for the current stage are pending.
        """
        # 1: First we count the number of configurations that have been launched
        # We only submit a new configuration M if all instance-seed pairs of (M - 1)
        # have been proposed
        configurations_by_this_intensifier = [
            c for c, i, s, b in self._run_tracker  # if b == self._all_budgets[self.stage]  # Bugfix closes #880
        ]
        running_configs = set()
        for k, v in runhistory.items():
            if runhistory.ids_config[k.config_id] in configurations_by_this_intensifier:
                # We get all configurations launched by the current intensifier
                # regardless if status is RUNNING or not, to make it more robust
                running_configs.add(runhistory.ids_config[k.config_id])

        # The total number of runs for this stage account for finished configurations
        # (success + failed + do not advance) + the offset + running but not finished
        # configurations. Also we account for the instances not launched for the
        # currently running configuration
        total_pending_configurations = max(
            0,
            self._n_configs_in_stage[self._stage]
            - (
                len(
                    set().union(
                        self._success_challengers,
                        self._failed_challengers,
                        self._do_not_advance_challengers,
                        running_configs,
                    )
                )
                + self._failed_challenger_offset
            ),
        )

        # 2: Second we have to account for the number of pending instances for the active
        # configuration. We assume for all (M - 1) configurations, all N instances-seeds
        # have been already been launched
        current_budget = self._all_budgets[self._stage]
        if self._successive_halving._instance_as_budget:
            previous_budget = int(self._all_budgets[self._stage - 1]) if self._stage > 0 else 0
            current_instances = self._successive_halving._instance_seed_pairs[
                int(previous_budget) : int(current_budget)
            ]
        else:
            current_instances = self._successive_halving._instance_seed_pairs

        if activate_configuration_being_intensified is None:
            # When a new stage begins, there is no active configuration.
            # Therefore activate_configuration_being_intensified is empty and all instances are
            # remaining
            pending_instances_to_launch = len(current_instances)
        else:
            # self._current_instance_indices - 1 is the last proposed instance/seed pair from get_next_trial
            # But it is zero indexed, so (self._current_instance_indices - 1) + 1 is the number of
            # configurations that we have proposed to run in total for the running
            # configuration via get_next_trial
            pending_instances_to_launch = max(
                len(current_instances) - self._current_instance_indices[activate_configuration_being_intensified], 0
            )  # type: ignore

        # If the there are any pending configuration, or instances/seed pending for the
        # active runner, we return a boolean
        return (total_pending_configurations + pending_instances_to_launch) <= 0
