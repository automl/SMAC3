from __future__ import annotations

from typing import Callable, Iterator, List

import numpy as np

from smac.configspace import Configuration
from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.intensification.successive_halving import SuccessiveHalving
from smac.runhistory import TrialInfo, TrialInfoIntent, TrialValue, StatusType
from smac.runhistory.runhistory import RunHistory
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class SuccessiveHalvingWorker(AbstractIntensifier):
    """Races multiple challengers against an incumbent using Successive Halving method.

    This class contains the logic to implement:
    "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)
    Supplementary reference: http://proceedings.mlr.press/v80/falkner18a/falkner18a-supp.pdf

    The `SuccessiveHalving` class can create multiple `_SuccessiveHalving` objects, to
    allow parallelism in the method (up to the number of workers available). The user  interface
    is expected to be `SuccessiveHalving`, yet this class (`_SuccessiveHalving`) contains the
    actual single worker implementation of the SMAC4MF method.

    Successive Halving intensifier (and Hyperband) can operate on two kinds of budgets:

    1. **'Instances' as budget**:

       When multiple instances are provided or when run objective is "runtime",
       this is the criterion used as budget
       for successive halving iterations i.e., the budget determines how many
       instances the challengers are evaluated
       on at a time. Top challengers for the next iteration are selected based
       on the combined performance across all instances used.

       If ``min_budget`` and ``max_budget`` are not provided, then they are
       set to 1 and total number of available instances respectively by default.

    2. **'Real-valued' budget**:

       This is used when there is only one instance provided and when run
       objective is "quality", i.e. budget is a positive, real-valued number
       that can be passed to the target algorithm as an argument.
       It can be used to control anything by the target algorithm,
       Eg: number of epochs for training a neural network.

       ``min_budget`` and ``max_budget`` are required parameters for
       this type of budget.

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
    _all_budgets: Optional[np.ndarray] = None
        Used internally when HB uses SH as a subrouting
    _n_configs_in_stage: Optional[np.ndarray] = None
        Used internally when HB uses SH as a subrouting
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
    identifier: int
        Adds a numerical identifier on this SH instance. Used for debug and tagging
        logger messages properly
    """

    def __init__(
        self,
        successive_halving: SuccessiveHalving,
        identifier: int = 0,
        _all_budgets: np.ndarray | None = None,
        _n_configs_in_stage: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            scenario=successive_halving.scenario,
            min_challenger=successive_halving.min_challenger,
            intensify_percentage=successive_halving.intensify_percentage,
            seed=successive_halving.seed,
        )

        self.successive_halving = successive_halving
        self.first_run = True
        self.identifier = identifier
        self.logger = get_logger(f"{__name__}.{identifier}")

        if _all_budgets is not None and _n_configs_in_stage is not None:
            # Assert we use the given numbers to avoid rounding issues, see #701
            self.all_budgets = _all_budgets
            self.n_configs_in_stage = _n_configs_in_stage
        else:
            eta = successive_halving.eta
            max_sh_iterations = successive_halving.max_sh_iterations
            max_budget = successive_halving.max_budget
            n_initial_challengers = successive_halving.n_initial_challengers

            # budgets to consider in each stage
            linspace = -np.linspace(max_sh_iterations, 0, max_sh_iterations + 1)
            self.all_budgets = max_budget * np.power(eta, linspace)
            # number of challengers to consider in each stage
            n_configs_in_stage = n_initial_challengers * np.power(
                eta, -np.linspace(0, max_sh_iterations, max_sh_iterations + 1)
            )
            self.n_configs_in_stage = np.array(np.round(n_configs_in_stage), dtype=int).tolist()

        # Current instance index tracks two things. Configurations that are to be launched,
        # That is config A needs to run in 3 instances/seed pairs, then current_instance_indices should
        # track this. But then, if a new configuration is added in the case of parallelism
        # a new separate current_instance_indices needs to be started.
        # The indices normally are of type int, but np.inf is used to indicate to not further
        # launch instances for this configuration, hence the type is Union[int, float].
        self.current_instance_indices: dict[Configuration, int | float] = {}
        self.running_challenger = None
        self.success_challengers: set[Configuration] = set()
        self.do_not_advance_challengers: set[Configuration] = set()
        self.fail_challengers: set[Configuration] = set()
        self.fail_chal_offset = 0

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
        # the ones launched by the current succesive halver using self.run_tracker
        self.run_tracker: dict[tuple[Configuration, str | None, int | None, float], bool] = {}

    def process_results(
        self,
        run_info: TrialInfo,
        run_value: TrialValue,
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
        incumbent: Configuration
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """

        # Mark the fact that we processed this configuration
        self.run_tracker[(run_info.config, run_info.instance, run_info.seed, run_info.budget)] = True

        # If The incumbent is None and it is the first run, we use the challenger
        if not incumbent and self.first_run:
            # We already displayed this message before
            # self.logger.info("First run and no incumbent provided. Challenger is assumed to be the incumbent.")
            incumbent = run_info.config
            self.first_run = False

        # In a serial run, if we have to CAP a run, then we stop launching
        # more configurations for this run.
        # In the context of parallelism, we launch the instances proactively
        # The fact that self.current_instance_indices[run_info.config] is np.inf means
        # that no more instances will be launched for the current config, so we
        # can add a check to make sure that if we are capping, this makes sense
        # for the active challenger
        # if result.status == StatusType.CAPPED and run_info.config == self.running_challenger:
        #    self.current_instance_indices[run_info.config] = np.inf
        # else:
        self._target_algorithm_time = self._target_algorithm_time
        self._target_algorithm_time += run_value.time
        self.num_run = self.num_run
        self.num_run += 1

        # 0: Before moving to a new stage, we have to complete M x N tasks, where M is the
        # total number of configurations evaluated in N instance/seed pairs.
        # The last active configuration is stored in self.running challengers is the M
        # configuration, and to get to this point, we have already submitted tasks
        # for (M - 1) configurations on N instances seed-pairs. The status of such
        # (M - 1) * N tasks is tracked in self.run_tracker, that has a value of False
        # if not processed and true if such task has been processed.
        # This stage is complete only if all tasks have been launched and all of the
        # already launched tasks are processed.

        # 1: We first query if we have launched everything already (All M * N tasks)
        all_config_inst_seed_launched = self._all_config_instance_seed_pairs_launched(
            runhistory=runhistory,
            activate_configuration_being_intensified=self.running_challenger,
        )

        # 2: Then we get the already submitted tasks (that is, proposed by get_next_runs),
        # that have not yet been processed process_results
        all_config_inst_seeds_processed = len([v for v in self.run_tracker.values() if not v]) <= 0

        # 3: Then the total number of remaining task before we can conclude this stage
        # is calculated by taking into account point 2 and 3 above
        is_stage_done = all_config_inst_seed_launched and all_config_inst_seeds_processed

        # adding challengers to the list of evaluated challengers
        #  - Stop: CAPPED/CRASHED/TIMEOUT/MEMOUT/DONOTADVANCE (!= SUCCESS)
        #  - Advance to next stage: SUCCESS
        # curr_challengers is a set, so "at least 1" success can be counted by set addition (no duplicates)
        # If a configuration is successful, it is added to curr_challengers.
        # if it fails it is added to fail_challengers.
        if np.isfinite(self.current_instance_indices[run_info.config]) and run_value.status == StatusType.SUCCESS:
            self.success_challengers.add(run_info.config)  # successful configs
        elif (
            np.isfinite(self.current_instance_indices[run_info.config]) and run_value.status == StatusType.DONOTADVANCE
        ):
            self.do_not_advance_challengers.add(run_info.config)
        else:
            self.fail_challengers.add(run_info.config)  # capped/crashed/do not advance configs

        # We need to update the incumbent if this config we are processing
        # completes all scheduled instance-seed pairs.
        # Here, a config/seed/instance is going to be processed for the first time
        # (it has been previously scheduled by get_next_run and marked False, indicating
        # that it has not been processed yet. Entering process_results() this config/seed/instance
        # is marked as TRUE as an indication that it has finished and should be processed)
        # so if all configurations runs are marked as TRUE it means that this new config
        # was the missing piece to have everything needed to compare against the incumbent
        update_incumbent = all([v for k, v in self.run_tracker.items() if k[0] == run_info.config])

        # get incumbent if all instances have been evaluated
        if is_stage_done or update_incumbent:
            incumbent = self._compare_configs(
                challenger=run_info.config,
                incumbent=incumbent,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )
        if is_stage_done:
            self.logger.info(
                "Finished Successive Halving iteration-step %d-%d with "
                "budget [%.2f / %d] and %d evaluated challenger(s)."
                % (
                    self.sh_iters + 1,
                    self.stage + 1,
                    self.all_budgets[self.stage],
                    self.successive_halving.max_budget,
                    self.n_configs_in_stage[self.stage],
                )
            )

            self._update_stage(runhistory=runhistory)

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
    ) -> tuple[TrialInfoIntent, TrialInfo]:
        """Selects which challenger to use based on the iteration stage and set the iteration
        parameters. First iteration will choose configurations from the ``chooser`` or input
        challengers, while the later iterations pick top configurations from the previously selected
        challengers in that iteration.

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
        if n_workers > 1:
            self.logger.warning(
                "Consider using ParallelSuccesiveHalving instead of "
                "SuccesiveHalving. The later will halt on each stage "
                "transition until all configs for the current stage are completed."
            )
        # if this is the first run, then initialize tracking variables
        if not hasattr(self, "stage"):
            self._update_stage(runhistory=runhistory)

        # In the case of multiprocessing, we have runs in Running stage, which have not
        # been processed via process_results(). get_next_run() is called agnostically by
        # smbo. To prevent launching more configs, than the ones needed, we query if
        # there is room for more configurations, else we wait for process_results()
        # to trigger a new stage
        if self._all_config_instance_seed_pairs_launched(runhistory, self.running_challenger):
            return TrialInfoIntent.WAIT, TrialInfo(
                config=None,
                instance=None,
                seed=0,
                budget=0.0,
                source=self.identifier,
            )

        # sampling from next challenger marks the beginning of a new iteration
        self.iteration_done = False

        curr_budget = self.all_budgets[self.stage]

        # if all instances have been executed, then reset and move on to next config
        if self.successive_halving.instance_as_budget:
            prev_budget = int(self.all_budgets[self.stage - 1]) if self.stage > 0 else 0
            n_insts = int(curr_budget) - prev_budget
        else:
            n_insts = len(self.successive_halving.instance_seed_pairs)

        # The instances remaining tell us, per configuration, how many instances we
        # have suggested to SMBO
        n_insts_remaining = n_insts
        if self.running_challenger is not None:
            n_insts_remaining = n_insts - self.current_instance_indices[self.running_challenger]

        # if there are instances pending, finish running configuration
        if self.running_challenger and n_insts_remaining > 0:
            challenger = self.running_challenger
            new_challenger = False
        else:
            # select next configuration
            if self.stage == 0:
                # first stage, so sample from configurations/chooser provided
                challenger = self._next_challenger(
                    challengers=challengers,
                    ask=ask,
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
                        seed=0,
                        budget=0.0,
                        source=self.identifier,
                    )

                new_challenger = True
            else:
                # sample top configs from previously sampled configurations
                try:
                    challenger = self.configs_to_run.pop(0)
                    new_challenger = False
                except IndexError:
                    # self.configs_to_run is populated via update_stage,
                    # which is triggered after the completion of a run
                    # If by there are no more configs to run (which is the case
                    # if we run into a IndexError),
                    return TrialInfoIntent.SKIP, TrialInfo(
                        config=None,
                        instance=None,
                        seed=0,
                        budget=0.0,
                        source=self.identifier,
                    )

            if challenger:
                # We see a challenger for the first time, so no
                # instance has been launched
                self.current_instance_indices[challenger] = 0
                self._challenger_id = self._challenger_id  # type: int # make mypy happy
                self._challenger_id += 1
                self.running_challenger = challenger

        if (incumbent is None or incumbent == challenger) and self.first_run:
            self.logger.info("First run and no incumbent provided. Challenger is assumed to be the incumbent.")
            incumbent = challenger

        # Selecting instance-seed subset for this budget, depending on the kind of budget
        if self.successive_halving.instance_as_budget:
            prev_budget = int(self.all_budgets[self.stage - 1]) if self.stage > 0 else 0
            curr_insts = self.successive_halving.instance_seed_pairs[int(prev_budget) : int(curr_budget)]
        else:
            curr_insts = self.successive_halving.instance_seed_pairs

        self.logger.debug("Running challenger - %s" % str(challenger))

        # run the next instance-seed pair for the given configuration
        instance, seed = curr_insts[self.current_instance_indices[challenger]]  # type: ignore[index]
        # At this point self.current_instance_indices[challenger] will still be an integer and might
        # be marked LATER with np.inf, so ignore mypy error.

        # For testing purposes, this attribute highlights whether a
        # new challenger is proposed or not. Not required from a functional
        # perspective
        self.new_challenger = new_challenger

        # TODO: Budget = None?
        budget = 0.0 if self.successive_halving.instance_as_budget else curr_budget
        self.run_tracker[(challenger, instance, seed, budget)] = False

        # self.current_instance_indices Tell us our current instance to be run. The upcoming return
        # will launch a challenger on a given instance/seed/pair. The next time this function
        # is called, we will like to run self.current_instance_indices + 1 for this configuration
        self.current_instance_indices[challenger] += 1

        return TrialInfoIntent.RUN, TrialInfo(
            config=challenger,
            instance=instance,
            seed=seed,
            budget=budget,
            source=self.identifier,
        )

    def _update_stage(self, runhistory: RunHistory) -> None:
        """Update tracking information for a new stage/iteration and update statistics. This method
        is called to initialize stage variables and after all configurations of a successive halving
        stage are completed.

        Parameters
        ----------
         runhistory : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        """
        if not hasattr(self, "stage"):
            # initialize all relevant variables for first run
            # (this initialization is not a part of init because hyperband uses the same init method and has a )
            # to track iteration and stage
            self.sh_iters = 0
            self.stage = 0
            # to track challengers across stages
            self.configs_to_run: list[Configuration] = []
            self.current_instance_indices = {}
            self.running_challenger = None
            self.success_challengers = set()  # Successful configs
            self.do_not_advance_challengers = set()  # Configs which are successful, but should not be advanced
            self.fail_challengers = set()  # Failures configs
            self.fail_chal_offset = 0
        else:
            self.stage += 1
            # only uncapped challengers are considered valid for the next iteration
            valid_challengers = list(
                (self.success_challengers | self.do_not_advance_challengers) - self.fail_challengers
            )

            if self.stage < len(self.all_budgets) and len(valid_challengers) > 0:
                # If this is the next stage in same iteration,
                # use top 'k' from the evaluated configurations for next iteration

                # Determine 'k' for the next iteration - at least 1
                next_n_chal = int(max(1, self.n_configs_in_stage[self.stage]))

                # Selecting the top 'k' challengers for the next iteration
                configs_to_run = self._top_k(configs=valid_challengers, runhistory=runhistory, k=next_n_chal)
                self.configs_to_run = [
                    config for config in configs_to_run if config not in self.do_not_advance_challengers
                ]

                # If some runs were capped, top_k returns less than the required configurations
                # to handle that, we keep track of how many configurations are missing
                # (since they are technically failed here too)
                missing_challengers = int(self.n_configs_in_stage[self.stage]) - len(self.configs_to_run)
                if missing_challengers > 0:
                    self.fail_chal_offset = missing_challengers
                else:
                    self.fail_chal_offset = 0

                if next_n_chal == missing_challengers:
                    next_stage = True
                    self.logger.info(
                        "Successive Halving iteration-step: %d-%d with "
                        "budget [%.2f / %d] - expected %d new challenger(s), but "
                        "no configurations propagated to the next budget.",
                        self.sh_iters + 1,
                        self.stage + 1,
                        self.all_budgets[self.stage],
                        self.successive_halving.max_budget,
                        self.n_configs_in_stage[self.stage],
                    )
                else:
                    next_stage = False
            else:
                next_stage = True

            if next_stage:
                # Update stats for the prev iteration
                assert self.stats
                self.stats.update_average_configs_per_intensify(n_configs=self._challenger_id)

                # reset stats for the new iteration
                self._target_algorithm_time = 0
                self._challenger_id = 0
                self.num_run = 0

                self.iteration_done = True
                self.sh_iters += 1
                self.stage = 0
                self.run_tracker = {}
                self.configs_to_run = []
                self.fail_chal_offset = 0

                # Randomize instance-seed pairs per successive halving run, if user specifies
                if self.successive_halving.instance_order == "shuffle":
                    self.rng.shuffle(self.instance_seed_pairs)  # type: ignore

        # to track configurations for the next stage
        self.success_challengers = set()  # successful configs
        self.do_not_advance_challengers = set()  # successful, but should not be advanced to the next budget/stage
        self.fail_challengers = set()  # capped/failed configs
        self.current_instance_indices = {}
        self.running_challenger = None

    def _compare_configs(
        self,
        incumbent: Configuration,
        challenger: Configuration,
        runhistory: RunHistory,
        log_trajectory: bool = True,
    ) -> Configuration | None:
        """Compares the challenger with current incumbent and returns the best configuration, based
        on the given incumbent selection design.

        Parameters
        ----------
        challenger : Configuration
            promising configuration
        incumbent : Configuration
            best configuration so far
        runhistory : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        log_trajectory : bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        Optional[Configuration]
            incumbent configuration
        """
        assert self.stats

        if self.successive_halving.instance_as_budget:
            new_incumbent = super()._compare_configs(incumbent, challenger, runhistory, log_trajectory)
            # if compare config returned none, then it is undecided. So return old incumbent
            new_incumbent = incumbent if new_incumbent is None else new_incumbent
            return new_incumbent

        # For real-valued budgets, compare configs based on the incumbent selection design
        curr_budget = self.all_budgets[self.stage]

        # incumbent selection: best on any budget
        if self.successive_halving.incumbent_selection == "any_budget":
            new_incumbent = self._compare_configs_across_budgets(
                challenger=challenger,
                incumbent=incumbent,
                runhistory=runhistory,
                log_trajectory=log_trajectory,
            )
            return new_incumbent

        # get runs for both configurations
        inc_runs = runhistory.get_runs_for_config(incumbent, only_max_observed_budget=True)
        chall_runs = runhistory.get_runs_for_config(challenger, only_max_observed_budget=True)

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

        # incumbent selection: highest budget only
        if self.successive_halving.incumbent_selection == "highest_budget":
            if chall_run.budget < self.successive_halving.max_budget:
                self.logger.debug(
                    "Challenger (budget=%.4f) has not been evaluated on the highest budget %.4f yet.",
                    chall_run.budget,
                    self.successive_halving.max_budget,
                )
                return incumbent

        # incumbent selection: highest budget run so far
        if inc_run.budget > chall_run.budget:
            self.logger.debug(
                "Incumbent evaluated on higher budget than challenger (%.4f > %.4f), " "not changing the incumbent",
                inc_run.budget,
                chall_run.budget,
            )
            return incumbent
        if inc_run.budget < chall_run.budget:
            self.logger.debug(
                "Challenger evaluated on higher budget than incumbent (%.4f > %.4f), " "changing the incumbent",
                chall_run.budget,
                inc_run.budget,
            )
            if log_trajectory:
                assert self.stats
                self.stats.add_incumbent(
                    cost=runhistory.get_cost(challenger),
                    incumbent=challenger,
                    budget=curr_budget,
                )

            return challenger

        # incumbent and challenger were both evaluated on the same budget, compare them based on their cost
        chall_cost = runhistory.get_cost(challenger)
        inc_cost = runhistory.get_cost(incumbent)
        if chall_cost < inc_cost:
            self.logger.info(
                "Challenger (%.4f) is better than incumbent (%.4f) on budget %.4f.",
                chall_cost,
                inc_cost,
                chall_run.budget,
            )
            self._log_incumbent_changes(incumbent, challenger)
            new_incumbent = challenger
            if log_trajectory:
                assert self.stats
                self.stats.add_incumbent(cost=chall_cost, incumbent=new_incumbent, budget=curr_budget)
        else:
            self.logger.debug(
                "Incumbent (%.4f) is at least as good as the challenger (%.4f) on budget %.4f.",
                inc_cost,
                chall_cost,
                inc_run.budget,
            )

            if log_trajectory and self.stats.incumbent_changed == 0:
                assert self.stats
                self.stats.add_incumbent(
                    cost=inc_cost,
                    incumbent=incumbent,
                    budget=curr_budget,
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

        Parameters
        ----------
        challenger : Configuration
            promising configuration
        incumbent : Configuration
            best configuration so far
        runhistory : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        log_trajectory : bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        Optional[Configuration]
            incumbent configuration
        """
        assert self.stats
        curr_budget = self.all_budgets[self.stage]

        # compare challenger and incumbent based on cost
        chall_cost = runhistory.get_min_cost(challenger)
        inc_cost = runhistory.get_min_cost(incumbent)
        if np.isfinite(chall_cost) and np.isfinite(inc_cost):
            if chall_cost < inc_cost:
                self.logger.info(
                    "Challenger (%.4f) is better than incumbent (%.4f) for any budget.",
                    chall_cost,
                    inc_cost,
                )
                self._log_incumbent_changes(incumbent, challenger)
                new_incumbent = challenger
                if log_trajectory:
                    assert self.stats
                    self.stats.add_incumbent(
                        cost=chall_cost,
                        incumbent=new_incumbent,
                        budget=curr_budget,
                    )
            else:
                self.logger.debug(
                    "Incumbent (%.4f) is at least as good as the challenger (%.4f) for any budget.",
                    inc_cost,
                    chall_cost,
                )
                if log_trajectory and self.stats.incumbent_changed == 0:
                    self.stats.add_incumbent(cost=inc_cost, incumbent=incumbent, budget=curr_budget)
                new_incumbent = incumbent
        else:
            self.logger.debug("Non-finite costs from run history!")
            new_incumbent = incumbent

        return new_incumbent

    def _top_k(self, configs: List[Configuration], runhistory: RunHistory, k: int) -> List[Configuration]:
        """Selects the top 'k' configurations from the given list based on their performance.

        This retrieves the performance for each configuration from the runhistory and checks
        that the highest budget they've been evaluated on is the same for each of the configurations.

        Parameters
        ----------
        configs: List[Configuration]
            list of configurations to filter from
        runhistory: smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        k: int
            number of configurations to select

        Returns
        -------
        List[Configuration]
            top challenger configurations, sorted in increasing costs
        """
        # extracting costs for each given configuration
        config_costs = {}
        # sample list instance-seed-budget key to act as base
        run_key = runhistory.get_runs_for_config(configs[0], only_max_observed_budget=True)
        for c in configs:
            # ensuring that all configurations being compared are run on the same set of instance, seed & budget
            cur_run_key = runhistory.get_runs_for_config(c, only_max_observed_budget=True)

            # Move to compare set -- get_runs_for_config queries form a dictionary
            # which is not an ordered structure. Some queries to that dictionary returned unordered
            # list which wrongly trigger the below if
            if set(cur_run_key) != set(run_key):
                raise ValueError(
                    "Can not compare configs that were run on different instances-seeds-budgets: %s vs %s"
                    % (run_key, cur_run_key)
                )
            config_costs[c] = runhistory.get_cost(c)

        configs_sorted = [k for k, v in sorted(config_costs.items(), key=lambda item: item[1])]
        # select top configurations only
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
            stores all runs we ran so far
        activate_configuration_being_intensified: Optional[Configuration]
            The last configuration being actively processes by this intensifier

        Returns
        -------
            bool: whether a instance/pair of any of the M configurations for the current
                stage are pending
        """
        # 1: First we count the number of configurations that have been launched
        # We only submit a new configuration M if all instance-seed pairs of (M - 1)
        # have been proposed
        configurations_by_this_intensifier = [c for c, i, s, b in self.run_tracker]
        running_configs = set()
        for k, v in runhistory.data.items():
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
            self.n_configs_in_stage[self.stage]
            - (
                len(
                    set().union(
                        self.success_challengers,
                        self.fail_challengers,
                        self.do_not_advance_challengers,
                        running_configs,
                    )
                )
                + self.fail_chal_offset
            ),
        )

        # 2: Second we have to account for the number of pending instances for the active
        # configuration. We assume for all (M - 1) configurations, all N instances-seeds
        # have been already been launched
        curr_budget = self.all_budgets[self.stage]
        if self.successive_halving.instance_as_budget:
            prev_budget = int(self.all_budgets[self.stage - 1]) if self.stage > 0 else 0
            curr_insts = self.successive_halving.instance_seed_pairs[int(prev_budget) : int(curr_budget)]
        else:
            curr_insts = self.successive_halving.instance_seed_pairs

        if activate_configuration_being_intensified is None:
            # When a new stage begins, there is no active configuration.
            # Therefore activate_configuration_being_intensified is empty and all instances are
            # remaining
            pending_instances_to_launch = len(curr_insts)
        else:
            # self.current_instance_indices - 1 is the last proposed instance/seed pair from get_next_run
            # But it is zero indexed, so (self.current_instance_indices - 1) + 1 is the number of
            # configurations that we have proposed to run in total for the running
            # configuration via get_next_run
            pending_instances_to_launch = max(
                len(curr_insts) - self.current_instance_indices[activate_configuration_being_intensified], 0
            )  # type: ignore

        # If the there are any pending configuration, or instances/seed pending for the
        # active runner, we return a boolean
        return (total_pending_configurations + pending_instances_to_launch) <= 0
