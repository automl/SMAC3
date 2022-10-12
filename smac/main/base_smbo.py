from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator

import time
from itertools import product

import numpy as np
from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.callback import Callback
from smac.constants import MAXINT
from smac.initial_design import AbstractInitialDesign
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.model.abstract_model import AbstractModel
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.runhistory import StatusType, TrialInfo, TrialInfoIntent, TrialValue
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.runhistory.runhistory import RunHistory
from smac.runner.abstract_runner import AbstractRunner
from smac.scenario import Scenario
from smac.stats import Stats
from smac.utils.data_structures import recursively_compare_dicts
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class BaseSMBO:
    """Implementation that contains the main Bayesian optimization loop.

    Parameters
    ----------
    scenario : Scenario
        The scenario object, holding all environmental information.
    stats : Stats
        Stats object to collect statistics about SMAC.
    runner : AbstractRunner
        The runner (containing the target function) is called internally to judge a trial's performance.
    initial_design : InitialDesign
        The sampled configurations from the initial design are evaluated before the Bayesian optimization loop starts.
    runhistory : Runhistory
        The runhistory stores all trials.
    runhistory_encoder : RunHistoryEncoder
        Based on the runhistory, the surrogate model is trained. However, the data first needs to be encoded, which
        is done by the runhistory encoder. For example, inactive hyperparameters need to be encoded or cost values
        can be log transformed.
    intensifier : AbstractIntensifier
        The intensifier decides which trial (combination of configuration, seed, budget and instance) should be run
        next.
    model : AbstractModel
        The surrogate model.
    acquisition_maximizer : AbstractAcquisitionMaximizer
        The acquisition maximizer, deciding which configuration is most promising based on the surrogate model and
        acquisition function.
    acquisition_function : AbstractAcquisitionFunction
        The acquisition function.
    random_design : RandomDesign
        The random design is used in the acquisition maximier, deciding whether the next configuration should be drawn
        from the acquisition function or randomly.
    overwrite: bool, defaults to False
        When True, overwrites the run results if a previous run is found that is
        inconsistent in the meta data with the current setup. If ``overwrite`` is set to False, the user is asked
        for the exact behaviour (overwrite completely, save old run, or use old results).

    Warning
    -------
    This model should only be initialized by a facade.
    """

    def __init__(
        self,
        scenario: Scenario,
        stats: Stats,
        runner: AbstractRunner,
        initial_design: AbstractInitialDesign,
        runhistory: RunHistory,
        runhistory_encoder: RunHistoryEncoder,
        intensifier: AbstractIntensifier,
        model: AbstractModel,
        acquisition_maximizer: AbstractAcquisitionMaximizer,
        acquisition_function: AbstractAcquisitionFunction,
        random_design: AbstractRandomDesign,
        overwrite: bool = False,
    ):
        self._scenario = scenario
        self._configspace = scenario.configspace
        self._stats = stats
        self._initial_design = initial_design
        self._runhistory = runhistory
        self._runhistory_encoder = runhistory_encoder
        self._intensifier = intensifier
        self._model = model
        self._acquisition_maximizer = acquisition_maximizer
        self._acquisition_function = acquisition_function
        self._random_design = random_design
        self._runner = runner
        self._overwrite = overwrite

        # Those are the configs sampled from the passed initial design
        # Selecting configurations from initial design
        self._initial_design_configs = self._initial_design.select_configurations()
        if len(self._initial_design_configs) == 0:
            raise RuntimeError("SMAC needs initial configurations to work.")

        # Internal variables
        self._finished = False
        self._allow_optimization = True
        self._stop = False  # Gracefully stop SMAC
        self._min_time = 10**-5
        self._callbacks: list[Callback] = []

        # We don't restore the incumbent anymore but derive it directly from
        # the stats object when the run is started.
        self._incumbent = None

        # We initialize the state based on previous data.
        # If no previous data is found then we take care of the initial design.
        self._initialize_state()

    @property
    def runhistory(self) -> RunHistory:
        """The run history, which is filled with all information during the optimization process."""
        return self._runhistory

    @property
    def stats(self) -> Stats:
        """The stats object, which is updated during the optimization and shows relevant information, e.g., how many
        trials have been finished and how the trajectory looks like.
        """
        return self._stats

    @property
    def incumbent(self) -> Configuration | None:
        """The best configuration so far."""
        return self._incumbent

    def update_model(self, model: AbstractModel) -> None:
        """Updates the model and updates the acquisition function."""
        self._model = model
        self._acquisition_function.model = model

    def update_acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        """Updates acquisition function and assosiates the current model. Also, the acquisition
        optimizer is updated.
        """
        self._acquisition_function = acquisition_function
        self._acquisition_function.model = self._model
        self._acquisition_maximizer.acquisition_function = acquisition_function

    def optimize(self) -> Configuration:
        """Runs the Bayesian optimization loop.

        Returns
        -------
        incumbent : Configuration
            The best found configuration.
        """
        if not self._allow_optimization:
            raise NotImplementedError("Unfortunately, previous runs can not be continued yet. 😞")

        # We return the incumbent if we already finished the optimization process (we don't want to allow to call
        # optimize more than once).
        if self._finished:
            return self._incumbent

        # Start the timer before we do anything
        self._stats.start_timing()
        time_left = None

        for callback in self._callbacks:
            callback.on_start(self)

        # Main BO loop
        while True:
            start_time = time.time()

            for callback in self._callbacks:
                callback.on_iteration_start(self)

            # Sample next configuration for intensification.
            # Initial design runs are also included in the BO loop now.
            intent, trial_info = self.ask()

            # Update timebound only if a 'new' configuration is sampled as the challenger
            if self._intensifier.num_trials == 0 or time_left is None:
                time_spent = time.time() - start_time
                time_left = self._get_timebound_for_intensification(time_spent)
                logger.debug(f"New intensification time bound: {time_left}")
            else:
                old_time_left = time_left
                time_spent = time_spent + (time.time() - start_time)
                time_left = self._get_timebound_for_intensification(time_spent)
                logger.debug(f"Updated intensification time bound from {old_time_left} to {time_left}.")

            # Skip starting new runs if the budget is now exhausted
            if self._stats.is_budget_exhausted():
                intent = TrialInfoIntent.SKIP

            # Skip the run if there was a request to do so.
            # For example, during intensifier intensification, we
            # don't want to rerun a config that was previously ran
            if intent == TrialInfoIntent.RUN:
                n_objectives = self._scenario.count_objectives()

                # Track the fact that a run was launched in the run
                # history. It's status is tagged as RUNNING, and once
                # completed and processed, it will be updated accordingly
                self._runhistory.add(
                    config=trial_info.config,
                    cost=float(MAXINT) if n_objectives == 1 else [float(MAXINT) for _ in range(n_objectives)],
                    time=0.0,
                    status=StatusType.RUNNING,
                    instance=trial_info.instance,
                    seed=trial_info.seed,
                    budget=trial_info.budget,
                )

                trial_info.config.config_id = self._runhistory._config_ids[trial_info.config]
                self._runner.submit_trial(trial_info=trial_info)
            elif intent == TrialInfoIntent.SKIP:
                # No launch is required
                # This marks a transition request from the intensifier
                # To a new iteration
                pass
            elif intent == TrialInfoIntent.WAIT:
                # In any other case, we wait for resources
                # This likely indicates that no further decision
                # can be taken by the intensifier until more data is
                # available
                self._runner.wait()
            else:
                raise NotImplementedError("No other RunInfoIntent has been coded.")

            # Check if there is any result, or else continue
            for trial_info, trial_value in self._runner.iter_results():
                # Add the results of the run to the run history
                # Additionally check for new incumbent
                self.tell(trial_info, trial_value, time_left)

            logger.debug(
                f"Remaining wallclock time: {self._stats.get_remaing_walltime()}; "
                f"Remaining cpu time: {self._stats.get_remaining_cputime()}; "
                f"Remaining trials: {self._stats.get_remaining_trials()}"
            )

            if self._stats.is_budget_exhausted() or self._stop:
                if self._stats.is_budget_exhausted():
                    logger.info("Configuration budget is exhausted.")
                else:
                    logger.info("Shutting down because the stop flag was set.")

                # The budget can be exhausted  for 2 reasons: number of ta runs or
                # time. If the number of ta runs is reached, but there is still budget,
                # wait for the runs to finish.
                while self._runner.is_running():
                    self._runner.wait()

                    for trial_info, trial_value in self._runner.iter_results():
                        # Add the results of the run to the run history
                        # Additionally check for new incumbent
                        self.tell(trial_info, trial_value, time_left)

                # Break from the intensification loop, as there are no more resources.
                break

            for callback in self._callbacks:
                callback.on_iteration_end(self)

            # Print stats at the end of each intensification iteration.
            if self._intensifier.iteration_done:
                self._stats.print()

        for callback in self._callbacks:
            callback.on_end(self)

        self._finished = True
        return self._incumbent

    @abstractmethod
    def get_next_configurations(self, n: int | None = None) -> Iterator[Configuration]:
        """Chooses next candidate solution with Bayesian optimization. The suggested configurations
        depend on the surrogate model acquisition optimizer/function. This method is used by
        the intensifier.

        Parameters
        ----------
        n : int | None, defaults to None
            Number of configurations to return. If None, uses the number of challengers defined in the acquisition
            optimizer.

        Returns
        -------
        configurations : Iterator[Configuration]
            Iterator over configurations from the acquisition optimizer.
        """
        raise NotImplementedError

    @abstractmethod
    def ask(self) -> tuple[TrialInfoIntent, TrialInfo]:
        """Asks the intensifier for the next trial.

        Returns
        -------
        intent : TrialInfoIntent
            Intent of the trials (wait/skip/run).
        info : TrialInfo
            Information about the trial (config, instance, seed, budget).
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self, info: TrialInfo, value: TrialValue, time_left: float | None = None, save: bool = True) -> None:
        """Adds the result of a trial to the runhistory and updates the intensifier. Also,
        the stats object is updated.

        Parameters
        ----------
        info : TrialInfo
            Describes the trial from which to process the results.
        value : TrialValue
            Contains relevant information regarding the execution of a trial.
        time_left : float | None, defaults to None
            How much time in seconds is left to perform intensification.
        save : bool, optional to True
            Whether the runhistory should be saved.
        """
        raise NotImplementedError

    def save(self) -> None:
        """Saves the current stats and runhistory."""
        self._stats.save()

        path = self._scenario.output_directory
        if path is not None:
            self._runhistory.save_json(str(path / "runhistory.json"))

    def _register_callback(self, callback: Callback) -> None:
        """Registers a callback to be called before, in between, and after the Bayesian optimization loop."""
        self._callbacks += [callback]

    def _initialize_state(self) -> None:
        """Detects whether the optimization is restored from a previous state."""
        # Here we actually check whether the run should be continued or not.
        # More precisely, we update our stats and runhistory object if all component arguments
        # and scenario/stats object are the same. For doing so, we create a specific hash.
        # The SMBO object recognizes that stats is not empty and hence does not the run initial design anymore.
        # Since the runhistory is already updated, the model uses previous data directly.

        if not self._overwrite:
            # First we get the paths from potentially previous data
            old_output_directory = self._scenario.output_directory
            old_runhistory_filename = self._scenario.output_directory / "runhistory.json"
            old_stats_filename = self._scenario.output_directory / "stats.json"

            if old_output_directory.exists() and old_runhistory_filename.exists() and old_stats_filename.exists():
                old_scenario = Scenario.load(old_output_directory)

                if self._scenario == old_scenario:
                    # TODO: We have to do something different here:
                    # The intensifier needs to know about what happened.
                    # Therefore, we read in the runhistory but use the tell method to add everything.
                    # Update: Not working yet as it's much more complicated. Therefore, we just throw an error.

                    logger.info("Continuing from previous run.")

                    # We update the runhistory and stats in-place.
                    # Stats use the output directory from the config directly.
                    self._runhistory.reset()
                    self._runhistory.load_json(str(old_runhistory_filename), configspace=self._scenario.configspace)
                    self._stats.load()

                    if self._stats.submitted == 1 and self._stats.finished == 0:
                        # Reset runhistory and stats if first run was not successful
                        logger.info("Since the previous run was not successful, SMAC will start from scratch again.")
                        self._runhistory.reset()
                        self._stats.reset()
                    elif self._stats.submitted == 0 and self._stats.finished == 0:
                        # If the other run did not start, we can just continue
                        self._runhistory.reset()
                        self._stats.reset()
                    else:
                        self._allow_optimization = False
                else:
                    diff = recursively_compare_dicts(
                        Scenario.make_serializable(self._scenario),
                        Scenario.make_serializable(old_scenario),
                        level="scenario",
                    )
                    logger.info(
                        f"Found old run in `{self._scenario.output_directory}` but it is not the same as the current "
                        f"one:\n{diff}"
                    )

                    feedback = input(
                        "\nPress one of the following numbers to continue or any other key to abort:\n"
                        "(1) Overwrite old run completely and start a new run.\n"
                        "(2) Rename the old run (append an '-old') and start a new run.\n"
                        "(3) Overwrite old run and re-use previous runhistory data. The configuration space "
                        "has to be the same for this option. This option is not tested yet.\n"
                    )

                    if feedback == "1":
                        # We don't have to do anything here, since we work with a clean runhistory and stats object
                        pass
                    elif feedback == "2":
                        # Rename old run
                        new_dir = str(old_scenario.output_directory.parent)
                        while True:
                            new_dir += "-old"
                            try:
                                old_scenario.output_directory.parent.rename(new_dir)
                                break
                            except OSError:
                                pass
                    elif feedback == "3":
                        # We overwrite runhistory and stats.
                        # However, we should ensure that we use the same configspace.
                        assert self._scenario.configspace == old_scenario.configspace

                        self._runhistory.load_json(str(old_runhistory_filename), configspace=self._scenario.configspace)
                        self._stats.load()
                    else:
                        raise RuntimeError("SMAC run was stopped by the user.")

        # And now we save everything
        self._scenario.save()
        self.save()

        # Make sure we use the current incumbent
        self._incumbent = self.stats.get_incumbent()

        # Sanity-checking: We expect an empty runhistory if finished in stats is 0
        # Note: stats.submitted might not be 0 because the user could have provide information via the tell method only
        if self.stats.finished == 0 or self._incumbent is None:
            assert self.runhistory.empty()
        else:
            # That's the case when the runhistory is not empty
            assert not self.runhistory.empty()

            logger.info(f"Starting optimization with incumbent {self._incumbent.get_dictionary()}.")
            self.stats.print()

    def _get_timebound_for_intensification(self, time_spent: float) -> float:
        """Calculate time left for intensify from the time spent on choosing challengers using the
        fraction of time intended for intensification (which is specified in
        ``intensifier.intensification_percentage``).

        Parameters
        ----------
        time_spent : float

        Returns
        -------
        time_left : float
        """
        if not hasattr(self._intensifier, "intensify_percentage"):
            return np.inf

        intensify_percentage = self._intensifier.intensify_percentage  # type: ignore
        total_time = time_spent / (1 - intensify_percentage)
        time_left = intensify_percentage * total_time

        logger.debug(
            f"\n--- Total time: {round(total_time, 4)}"
            f"\n--- Time spent on choosing next configurations: {round(time_spent, 4)} ({(1 - intensify_percentage)})"
            f"\n--- Time left for intensification: {round(time_left, 4)} ({intensify_percentage})"
        )

        return time_left

    def validate(
        self,
        config: Configuration,
        *,
        instances: list[str] | None = None,
        seed: int | None = None,
    ) -> float | list[float]:
        """Validates a configuration with different seeds than in the optimization process and on the highest
        budget (if budget type is real-valued).

        Parameters
        ----------
        config : Configuration
            Configuration to validate
        instances : list[str] | None, defaults to None
            Which instances to validate. If None, all instances specified in the scenario are used.
            In case that the budget type is real-valued budget, this argument is ignored.
        seed : int | None, defaults to None
            If None, the seed from the scenario is used.

        Returns
        -------
        cost : float | list[float]
            The averaged cost of the configuration. In case of multi-fidelity, the cost of each objective is
            averaged.
        """
        if seed is None:
            seed = self._scenario.seed

        rng = np.random.default_rng(seed)

        seeds = []
        for _ in range(len(self._intensifier.get_target_function_seeds())):
            seed = int(rng.integers(low=0, high=MAXINT, size=1)[0])
            seeds += [seed]

        used_budgets: list[float | None] = [None]
        if self._intensifier.uses_budgets:
            # Select last budget
            used_budgets = [self._intensifier.get_target_function_budgets()[-1]]

        used_instances: list[str | None] = [None]
        if self._intensifier.uses_instances:
            if instances is None:
                assert self._scenario.instances is not None
                used_instances = self._scenario.instances  # type: ignore

        costs = []
        for s, b, i in product(seeds, used_budgets, used_instances):
            kwargs: dict[str, Any] = {}
            if s is not None:
                kwargs["seed"] = s
            if b is not None:
                kwargs["budget"] = b
            if i is not None:
                kwargs["instance"] = i

            _, cost, _, _ = self._runner.run(config, **kwargs)
            costs += [cost]

        np_costs = np.array(costs)
        return np.mean(np_costs, axis=0)
