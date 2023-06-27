from __future__ import annotations

from typing import Any

import json
import time
from pathlib import Path

import numpy as np
from ConfigSpace import Configuration
from numpy import ndarray

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.callback.callback import Callback
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.model.abstract_model import AbstractModel
from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.runner import FirstRunCrashedException
from smac.runner.abstract_runner import AbstractRunner
from smac.runner.dask_runner import DaskParallelRunner
from smac.scenario import Scenario
from smac.utils.data_structures import recursively_compare_dicts
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class SMBO:
    """Implementation that contains the main Bayesian optimization loop.

    Parameters
    ----------
    scenario : Scenario
        The scenario object, holding all environmental information.
    runner : AbstractRunner
        The runner (containing the target function) is called internally to judge a trial's performance.
    runhistory : Runhistory
        The runhistory stores all trials.
    intensifier : AbstractIntensifier
        The intensifier decides which trial (combination of configuration, seed, budget and instance) should be run
        next.
    overwrite: bool, defaults to False
        When True, overwrites the run results if a previous run is found that is
        inconsistent in the meta data with the current setup. If ``overwrite`` is set to False, the user is asked
        for the exact behaviour (overwrite completely, save old run, or use old results).

    Warning
    -------
    This model should be initialized by a facade only.
    """

    def __init__(
        self,
        scenario: Scenario,
        runner: AbstractRunner,
        runhistory: RunHistory,
        intensifier: AbstractIntensifier,
        overwrite: bool = False,
    ):
        self._scenario = scenario
        self._configspace = scenario.configspace
        self._runhistory = runhistory
        self._intensifier = intensifier
        self._trial_generator = iter(intensifier)
        self._runner = runner
        self._overwrite = overwrite

        # Internal variables
        self._finished = False
        self._stop = False  # Gracefully stop SMAC
        self._callbacks: list[Callback] = []

        # Stats variables
        self._start_time: float | None = None
        self._used_target_function_walltime = 0.0

        # Set walltime used method for intensifier
        self._intensifier.used_walltime = lambda: self.used_walltime  # type: ignore

        # We initialize the state based on previous data.
        # If no previous data is found then we take care of the initial design.
        self._initialize_state()

    @property
    def runhistory(self) -> RunHistory:
        """The run history, which is filled with all information during the optimization process."""
        return self._runhistory

    @property
    def intensifier(self) -> AbstractIntensifier:
        """The run history, which is filled with all information during the optimization process."""
        return self._intensifier

    @property
    def remaining_walltime(self) -> float:
        """Subtracts the runtime configuration budget with the used wallclock time."""
        assert self._start_time is not None
        return self._scenario.walltime_limit - (time.time() - self._start_time)

    @property
    def remaining_cputime(self) -> float:
        """Subtracts the target function running budget with the used time."""
        return self._scenario.cputime_limit - self._used_target_function_walltime

    @property
    def remaining_trials(self) -> int:
        """Subtract the target function runs in the scenario with the used ta runs."""
        return self._scenario.n_trials - self.runhistory.submitted

    @property
    def budget_exhausted(self) -> bool:
        """Checks whether the the remaining walltime, cputime or trials was exceeded."""
        A = self.remaining_walltime <= 0
        B = self.remaining_cputime <= 0
        C = self.remaining_trials <= 0

        return A or B or C

    @property
    def used_walltime(self) -> float:
        """Returns used wallclock time."""
        if self._start_time is None:
            return 0.0

        return time.time() - self._start_time

    @property
    def used_target_function_walltime(self) -> float:
        """Returns how much walltime the target function spend so far."""
        return self._used_target_function_walltime

    def ask(self) -> TrialInfo:
        """Asks the intensifier for the next trial.

        Returns
        -------
        info : TrialInfo
            Information about the trial (config, instance, seed, budget).
        """
        logger.debug("Calling ask...")

        for callback in self._callbacks:
            callback.on_ask_start(self)

        # Now we use our generator to get the next trial info
        trial_info = next(self._trial_generator)

        # Track the fact that the trial was returned
        # This is really important because otherwise the intensifier would most likly sample the same trial again
        self._runhistory.add_running_trial(trial_info)

        for callback in self._callbacks:
            callback.on_ask_end(self, trial_info)

        logger.debug("...and received a new trial.")

        return trial_info

    def tell(
        self,
        info: TrialInfo,
        value: TrialValue,
        save: bool = True,
    ) -> None:
        """Adds the result of a trial to the runhistory and updates the stats object.

        Parameters
        ----------
        info : TrialInfo
            Describes the trial from which to process the results.
        value : TrialValue
            Contains relevant information regarding the execution of a trial.
        save : bool, optional to True
            Whether the runhistory should be saved.
        """
        if info.config.origin is None:
            info.config.origin = "Custom"

        for callback in self._callbacks:
            response = callback.on_tell_start(self, info, value)

            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called.
            if response is False:
                logger.info("A callback returned False. Abort is requested.")
                self._stop = True

        # Some sanity checks here
        if self._intensifier.uses_instances and info.instance is None:
            raise ValueError("Passed instance is None but intensifier requires instances.")

        if self._intensifier.uses_budgets and info.budget is None:
            raise ValueError("Passed budget is None but intensifier requires budgets.")

        self._runhistory.add(
            config=info.config,
            cost=value.cost,
            time=value.time,
            status=value.status,
            instance=info.instance,
            seed=info.seed,
            budget=info.budget,
            starttime=value.starttime,
            endtime=value.endtime,
            additional_info=value.additional_info,
            force_update=True,  # Important to overwrite the status RUNNING
        )

        logger.debug(f"Tell method was called with cost {value.cost} ({StatusType(value.status).name}).")

        for callback in self._callbacks:
            response = callback.on_tell_end(self, info, value)

            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called.
            if response is False:
                logger.info("A callback returned False. Abort is requested.")
                self._stop = True

        if save:
            self.save()

    def update_model(self, model: AbstractModel) -> None:
        """Updates the model and updates the acquisition function."""
        if (config_selector := self._intensifier._config_selector) is not None:
            config_selector._model = model

            assert config_selector._acquisition_function is not None
            config_selector._acquisition_function.model = model

    def update_acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        """Updates the acquisition function including the associated model and the acquisition
        optimizer.
        """
        if (config_selector := self._intensifier._config_selector) is not None:
            config_selector._acquisition_function = acquisition_function
            config_selector._acquisition_function.model = config_selector._model

            assert config_selector._acquisition_maximizer is not None
            config_selector._acquisition_maximizer.acquisition_function = acquisition_function

    def optimize(self, *, data_to_scatter: dict[str, Any] | None = None) -> Configuration | list[Configuration]:
        """Runs the Bayesian optimization loop.

        Parameters
        ----------
        data_to_scatter: dict[str, Any] | None
            When a user scatters data from their local process to the distributed network,
            this data is distributed in a round-robin fashion grouping by number of cores.
            Roughly speaking, we can keep this data in memory and then we do not have to (de-)serialize the data
            every time we would like to execute a target function with a big dataset.
            For example, when your target function has a big dataset shared across all the target function,
            this argument is very useful.

        Returns
        -------
        incumbent : Configuration
            The best found configuration.
        """
        # We return the incumbent if we already finished the a process (we don't want to allow to call
        # optimize more than once).
        if self._finished:
            logger.info("Optimization process was already finished. Returning incumbent...")
            if self._scenario.count_objectives() == 1:
                return self.intensifier.get_incumbent()
            else:
                return self.intensifier.get_incumbents()

        # Start the timer before we do anything
        # If we continue the optimization, the starting time is set by the load method
        if self._start_time is None:
            self._start_time = time.time()

        for callback in self._callbacks:
            callback.on_start(self)

        dask_data_to_scatter = {}
        if isinstance(self._runner, DaskParallelRunner) and data_to_scatter is not None:
            dask_data_to_scatter = dict(data_to_scatter=self._runner._client.scatter(data_to_scatter, broadcast=True))
        elif data_to_scatter is not None:
            raise ValueError(
                "data_to_scatter is valid only for DaskParallelRunner, "
                f"but {dask_data_to_scatter} was provided for {self._runner.__class__.__name__}"
            )

        # Main BO loop
        while True:
            for callback in self._callbacks:
                callback.on_iteration_start(self)

            try:
                # Sample next trial from the intensification
                trial_info = self.ask()

                # We submit the trial to the runner
                # In multi-worker mode, SMAC waits till a new worker is available here
                self._runner.submit_trial(trial_info=trial_info, **dask_data_to_scatter)
            except StopIteration:
                self._stop = True

            # We add results from the runner if results are available
            self._add_results()

            # Some statistics
            logger.debug(
                f"Remaining wallclock time: {self.remaining_walltime}; "
                f"Remaining cpu time: {self.remaining_cputime}; "
                f"Remaining trials: {self.remaining_trials}"
            )

            if self.runhistory.finished % 50 == 0:
                logger.info(f"Finished {self.runhistory.finished} trials.")

            for callback in self._callbacks:
                callback.on_iteration_end(self)

            # Now we check whether we have to stop the optimization
            if self.budget_exhausted or self._stop:
                if self.budget_exhausted:
                    logger.info("Configuration budget is exhausted:")
                    logger.info(f"--- Remaining wallclock time: {self.remaining_walltime}")
                    logger.info(f"--- Remaining cpu time: {self.remaining_cputime}")
                    logger.info(f"--- Remaining trials: {self.remaining_trials}")
                else:
                    logger.info("Shutting down because the stop flag was set.")

                # Wait for the trials to finish
                while self._runner.is_running():
                    self._runner.wait()
                    self._add_results()

                # Break from the intensification loop, as there are no more resources
                break

        for callback in self._callbacks:
            callback.on_end(self)

        # We only set the finished flag if the budget really was exhausted
        if self.budget_exhausted:
            self._finished = True

        if self._scenario.count_objectives() == 1:
            return self.intensifier.get_incumbent()
        else:
            return self.intensifier.get_incumbents()

    def reset(self) -> None:
        """Resets the internal variables of the optimizer, intensifier, and runhistory."""
        self._used_target_function_walltime = 0
        self._finished = False

        # We also reset runhistory and intensifier here
        self._runhistory.reset()
        self._intensifier.reset()

    def exists(self, filename: str | Path) -> bool:
        """Checks if the files associated with the run already exist.
        Checks all files that are created by the optimizer.

        Parameters
        ----------
        filename : str | Path
            The name of the folder of the SMAC run.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        optimization_fn = filename / "optimization.json"
        runhistory_fn = filename / "runhistory.json"
        intensifier_fn = filename / "intensifier.json"

        if optimization_fn.exists() and runhistory_fn.exists() and intensifier_fn.exists():
            return True

        return False

    def load(self) -> None:
        """Loads the optimizer, intensifier, and runhistory from the output directory specified in the scenario."""
        filename = self._scenario.output_directory

        optimization_fn = filename / "optimization.json"
        runhistory_fn = filename / "runhistory.json"
        intensifier_fn = filename / "intensifier.json"

        if filename is not None:
            with open(optimization_fn) as fp:
                data = json.load(fp)

            self._runhistory.load(runhistory_fn, configspace=self._scenario.configspace)
            self._intensifier.load(intensifier_fn)

            self._used_target_function_walltime = data["used_target_function_walltime"]
            self._finished = data["finished"]
            self._start_time = time.time() - data["used_walltime"]

    def save(self) -> None:
        """Saves the current stats, runhistory, and intensifier."""
        path = self._scenario.output_directory

        if path is not None:
            data = {
                "used_walltime": self.used_walltime,
                "used_target_function_walltime": self.used_target_function_walltime,
                "last_update": time.time(),
                "finished": self._finished,
            }

            # Save optimization data
            with open(str(path / "optimization.json"), "w") as file:
                json.dump(data, file, indent=2)

            # And save runhistory and intensifier
            self._runhistory.save(path / "runhistory.json")
            self._intensifier.save(path / "intensifier.json")

    def _add_results(self) -> None:
        """Adds results from the runner to the runhistory. Although most of the functionality could be written
        in the tell method, we separate it here to make it accessible for the automatic optimization procedure only.
        """
        # Check if there is any result
        for trial_info, trial_value in self._runner.iter_results():
            # Add the results of the run to the run history
            self.tell(trial_info, trial_value)

            # We expect the first run to always succeed.
            if self.runhistory.finished == 0 and trial_value.status == StatusType.CRASHED:
                additional_info = ""
                if "traceback" in trial_value.additional_info:
                    additional_info = "\n\n" + trial_value.additional_info["traceback"]

                raise FirstRunCrashedException(
                    "The first run crashed. Please check your setup again." + additional_info
                )

            # Update SMAC stats
            self._used_target_function_walltime += float(trial_value.time)

            # Gracefully end optimization if termination cost is reached
            if self._scenario.termination_cost_threshold != np.inf:
                cost = self.runhistory.average_cost(trial_info.config)

                if not isinstance(cost, list):
                    cost = [cost]

                if not isinstance(self._scenario.termination_cost_threshold, list):
                    cost_threshold = [self._scenario.termination_cost_threshold]
                else:
                    cost_threshold = self._scenario.termination_cost_threshold

                if len(cost) != len(cost_threshold):
                    raise RuntimeError("You must specify a termination cost threshold for each objective.")

                if all(cost[i] < cost_threshold[i] for i in range(len(cost))):
                    logger.info("Cost threshold was reached. Abort is requested.")
                    self._stop = True

    def register_callback(self, callback: Callback, index: int | None = None) -> None:
        """
        Registers a callback to be called before, in between, and after the Bayesian optimization loop.

        Callback is appended to the list by default.

        Parameters
        ----------
        callback : Callback
            The callback to be registered.
        index : int, optional
            The index at which the callback should be registered. The default is None.
            If it is None, append the callback to the list.
        """
        if index is None:
            index = len(self._callbacks)
        self._callbacks.insert(index, callback)

    def _initialize_state(self) -> None:
        """Detects whether the optimization is restored from a previous state."""
        # Here we actually check whether the run should be continued or not.
        # More precisely, we update our smbo/runhistory/intensifier object if all component arguments
        # and scenario object are the same. For doing so, we create a specific hash.
        # The SMBO object recognizes that stats (based on runhistory) is not empty and hence does not the run initial
        # design anymore.
        # Since the runhistory is already updated, the model uses previous data directly.

        if not self._overwrite:
            old_output_directory = self._scenario.output_directory
            if self.exists(old_output_directory):
                old_scenario = Scenario.load(old_output_directory)

                if self._scenario == old_scenario:
                    logger.info("Continuing from previous run.")

                    # First we reset everything and then we load the old states
                    self.reset()
                    self.load()

                    # If the last run was not successful, we reset everything again
                    if self._runhistory.submitted <= 1 and self._runhistory.finished == 0:
                        logger.info("Since the previous run was not successful, SMAC will start from scratch again.")
                        self.reset()
                else:
                    # Here, we run into different scenarios
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
                    else:
                        raise RuntimeError("SMAC run was stopped by the user.")

        # And now we save everything
        self._scenario.save()
        self.save()

    def validate(
        self,
        config: Configuration,
        *,
        seed: int | None = None,
    ) -> float | ndarray[float]:
        """Validates a configuration on other seeds than the ones used in the optimization process and on the highest
        budget (if budget type is real-valued). Does not exceed the maximum number of config calls or seeds as defined
        in the scenario.

        Parameters
        ----------
        config : Configuration
            Configuration to validate
            In case that the budget type is real-valued budget, this argument is ignored.
        seed : int | None, defaults to None
            If None, the seed from the scenario is used.

        Returns
        -------
        cost : float | ndarray[float]
            The averaged cost of the configuration. In case of multi-fidelity, the cost of each objective is
            averaged.
        """
        if seed is None:
            seed = self._scenario.seed

        costs = []
        for trial in self._intensifier.get_trials_of_interest(config, validate=True, seed=seed):
            kwargs: dict[str, Any] = {}
            if trial.seed is not None:
                kwargs["seed"] = trial.seed
            if trial.budget is not None:
                kwargs["budget"] = trial.budget
            if trial.instance is not None:
                kwargs["instance"] = trial.instance

            # TODO: Use submit run for faster evaluation
            # self._runner.submit_trial(trial_info=trial)
            _, cost, _, _ = self._runner.run(config, **kwargs)
            costs += [cost]

        np_costs = np.array(costs)
        return np.mean(np_costs, axis=0)

    def print_stats(self) -> None:
        """Prints all statistics."""
        logger.info(
            "\n"
            f"--- STATISTICS -------------------------------------\n"
            f"--- Incumbent changed: {self.intensifier.incumbents_changed}\n"
            f"--- Submitted trials: {self.runhistory.submitted} / {self._scenario.n_trials}\n"
            f"--- Finished trials: {self.runhistory.finished} / {self._scenario.n_trials}\n"
            f"--- Configurations: {self.runhistory._n_id}\n"
            f"--- Used wallclock time: {round(self.used_walltime)} / {self._scenario.walltime_limit} sec\n"
            "--- Used target function runtime: "
            f"{round(self.used_target_function_walltime, 2)} / {self._scenario.cputime_limit} sec\n"
            f"----------------------------------------------------"
        )
