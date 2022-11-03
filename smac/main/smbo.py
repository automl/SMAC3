from __future__ import annotations

from typing import Any

import numpy as np
import time
from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)

from smac.runner import FirstRunCrashedException, TargetAlgorithmAbortException
from smac.callback import Callback
from smac.intensifier.abstract_intensifier import AbstractIntensifier
from smac.model.abstract_model import AbstractModel
from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.runhistory.runhistory import RunHistory
from smac.runner.abstract_runner import AbstractRunner
from smac.scenario import Scenario
from smac.stats import Stats
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
        self._allow_optimization = True
        self._stop = False  # Gracefully stop SMAC
        self._callbacks: list[Callback] = []

        # Stats variables
        self._start_time = 0.0
        self._submitted = 0
        self._finished = 0
        self._n_configs = 0
        self._used_target_function_walltime = 0.0

        # We initialize the state based on previous data.
        # If no previous data is found then we take care of the initial design.
        self._initialize_state()

    @property
    def runhistory(self) -> RunHistory:
        """The run history, which is filled with all information during the optimization process."""
        return self._runhistory

    @property
    def submitted_trials(self) -> int:
        """How many trials have been submitted."""
        return self._submitted

    @property
    def finished_trials(self) -> int:
        """How many trials have been evaluated."""
        return self._finished

    @property
    def n_configs(self) -> int:
        """How many different configurations have been evaluated."""
        return self._n_configs

    @property
    def remaing_walltime(self) -> float:
        """Subtracts the runtime configuration budget with the used wallclock time."""
        return self._scenario.walltime_limit - (time.time() - self._start_time)

    @property
    def remaining_cputime(self) -> float:
        """Subtracts the target function running budget with the used time."""
        return self._scenario.cputime_limit - self._used_target_function_walltime

    @property
    def remaining_trials(self) -> int:
        """Subtract the target function runs in the scenario with the used ta runs."""
        return self._scenario.n_trials - self.submitted_trials

    @property
    def budget_exhausted(self) -> bool:
        """Check whether the the remaining walltime, cputime or trials was exceeded."""
        A = self.remaing_walltime < 0
        B = self.remaining_cputime < 0
        C = self.remaining_trials < 0

        return A or B or C

    @property
    def used_walltime(self) -> float:
        """Returns used wallclock time."""
        return time.time() - self._start_time

    @property
    def used_target_function_walltime(self) -> float:
        """Returns how much walltime the target function spend so far."""
        return self._used_target_function_walltime

    @property
    def trajectory(self):
        pass

    @property
    def incumbent(self) -> Configuration | None:
        """The best configuration so far."""
        return self._incumbent

    def ask(self) -> TrialInfo:
        """Asks the intensifier for the next trial.

        Returns
        -------
        info : TrialInfo
            Information about the trial (config, instance, seed, budget).
        """
        for callback in self._callbacks:
            callback.on_ask_start(self)
            
        # TODO: Does this work with the new intensifier?
        if (mo := self._runhistory_encoder.multi_objective_algorithm) is not None:
            mo.update_on_iteration_start()

        # Now we use our generator to get the next trial info
        trial_info = next(self._trial_generator)

        # Track the fact that the trial was returned
        self._runhistory.add_running_trial(trial_info)

        for callback in self._callbacks:
            callback.on_ask_end(self, trial_info)

        return trial_info

    def tell(
        self,
        info: TrialInfo,
        value: TrialValue,
        save: bool = True,
    ) -> None:
        """Adds the result of a trial to the runhistory. Also, the stats object is updated.

        Parameters
        ----------
        info : TrialInfo
            Describes the trial from which to process the results.
        value : TrialValue
            Contains relevant information regarding the execution of a trial.
        save : bool, optional to True
            Whether the runhistory should be saved.
        """
        # We first check if budget/instance/seed is supported by the intensifier
        # if info.seed not in (seeds := self._intensifier.get_target_function_seeds()):
        #    raise ValueError(f"Seed {info.seed} is not supported by the intensifier. Consider using one of {seeds}.")
        # elif info.budget not in (budgets := self._intensifier.get_target_function_budgets()):
        #    raise ValueError(
        #        f"Budget {info.budget} is not supported by the intensifier. Consider using one of {budgets}."
        #    )
        # elif info.instance not in (instances := self._intensifier.get_target_function_instances()):
        #    raise ValueError(
        #        f"Instance {info.instance} is not supported by the intensifier. Consider using one of {instances}."
        #    )

        if info.config.origin is None:
            info.config.origin = "Custom"

        for callback in self._callbacks:
            response = callback.on_tell_start(self, info, value)

            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called.
            if response is False:
                logger.info("An callback returned False. Abort is requested.")
                self._stop = True

        logger.debug(
            f"Status: {value.status}, "
            f"Cost: {value.cost}, "
            f"Time: {value.time}, "
            f"Additional: {value.additional_info}"
        )

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

        for callback in self._callbacks:
            response = callback.on_tell_end(self, info, value)

            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called.
            if response is False:
                logger.info("An callback returned False. Abort is requested.")
                self._stop = True

        if save:
            self.save()

    def update_model(self, model: AbstractModel) -> None:
        """Updates the model and updates the acquisition function."""
        if (config_selector := self._intensifier._config_selector) is not None:
            config_selector._model = model
            config_selector._acquisition_function.model = model

    def update_acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        """Updates acquisition function and assosiates the current model. Also, the acquisition
        optimizer is updated.
        """
        if (config_selector := self._intensifier._config_selector) is not None:
            config_selector._acquisition_function = acquisition_function
            config_selector._acquisition_function.model = config_selector._model
            config_selector._acquisition_maximizer.acquisition_function = acquisition_function

    def optimize(self) -> Configuration:
        """Runs the Bayesian optimization loop.

        Returns
        -------
        incumbent : Configuration
            The best found configuration.
        """

        # We return the incumbent if we already finished the optimization process (we don't want to allow to call
        # optimize more than once).
        if self._finished:
            return self._incumbent

        # Start the timer before we do anything
        self._start_time = time.time() - self.used_walltime

        for callback in self._callbacks:
            callback.on_start(self)

        # Main BO loop
        while True:
            for callback in self._callbacks:
                callback.on_iteration_start(self)

            # Sample next trial from the intensification
            trial_info = self.ask()
            # trial_info.config.config_id = self._runhistory._config_ids[trial_info.config]

            # self._runhistory.add(
            #    config=trial_info.config,
            #    cost=float(MAXINT) if n_objectives == 1 else [float(MAXINT) for _ in range(n_objectives)],
            #    time=0.0,
            #    status=StatusType.RUNNING,
            #    instance=trial_info.instance,
            #    seed=trial_info.seed,
            #    budget=trial_info.budget,
            # )

            # We submit the trial to the runner
            self._runner.submit_trial(trial_info=trial_info)
            self._submitted += 1
            self._n_configs = len(self._runhistory._config_ids)

            # We add results from the runner if results are available
            self._add_results()

            # Some statistics
            logger.debug(
                f"Remaining wallclock time: {self.remaing_walltime}; "
                f"Remaining cpu time: {self.remaining_cputime}; "
                f"Remaining trials: {self.remaining_trials}"
            )

            # Now we check whether we have to stop the optimization
            if self.budget_exhausted or self._stop:
                if self.budget_exhausted:
                    logger.info("Configuration budget is exhausted.")
                else:
                    logger.info("Shutting down because the stop flag was set.")

                # Wait for the trials to finish
                while self._runner.is_running():
                    self._runner.wait()
                    self._add_results()

                # Break from the intensification loop, as there are no more resources
                break

            for callback in self._callbacks:
                callback.on_iteration_end(self)

        for callback in self._callbacks:
            callback.on_end(self)

        self._finished = True
        return self.runhistory.get_incumbent()[0]

    def save(self) -> None:
        """Saves the current stats and runhistory."""
        self._stats.save()

        path = self._scenario.output_directory
        if path is not None:
            self._runhistory.save_json(str(path / "runhistory.json"))

    def _add_results(self) -> None:
        """Adds results from the runner to the runhistory. Although most of the functionality could be written
        in the tell method, we separate it here to make it accessible for the automatic optimization procedure only.
        """
        # Check if there is any result
        for trial_info, trial_value in self._runner.iter_results():
            # Add the results of the run to the run history
            self.tell(trial_info, trial_value)

            # We expect the first run to always succeed.
            if self.finished == 0 and trial_value.status == StatusType.CRASHED:
                additional_info = ""
                if "traceback" in trial_value.additional_info:
                    additional_info = "\n\n" + trial_value.additional_info["traceback"]

                raise FirstRunCrashedException(
                    "The first run crashed. Please check your setup again." + additional_info
                )

            # Update SMAC stats
            self._used_target_function_walltime += float(trial_value.time)
            self._finished += 1

            if trial_value.status == StatusType.ABORT:
                raise TargetAlgorithmAbortException(
                    "The target function was aborted. The last incumbent can be found in the trajectory file."
                )
            elif trial_value.status == StatusType.STOP:
                logger.debug("Value holds the status stop. Abort is requested.")
                self._stop = True

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

                    if self.submitted == 1 and self.finished == 0:
                        # Reset runhistory and stats if first run was not successful
                        logger.info("Since the previous run was not successful, SMAC will start from scratch again.")
                        self._runhistory.reset()
                        self._stats.reset()
                    elif self.submitted == 0 and self.finished == 0:
                        # If the other run did not start, we can just continue
                        self._runhistory.reset()
                        self._stats.reset()
                    # else:
                    #    self._allow_optimization = False
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
        if self.finished == 0 or self._incumbent is None:
            assert self.runhistory.empty()
        else:
            # That's the case when the runhistory is not empty
            assert not self.runhistory.empty()

            logger.info(f"Starting optimization with incumbent {self._incumbent.get_dictionary()}.")
            self.stats.print()

    def validate(
        self,
        config: Configuration,
        *,
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

        costs = []
        for trial in self._intensifier.get_trials_of_interest(config, seed=seed, validate=True):
            kwargs: dict[str, Any] = {}
            if trial.seed is not None:
                kwargs["seed"] = trial.seed
            if trial.budget is not None:
                kwargs["budget"] = trial.budget
            if trial.instance is not None:
                kwargs["instance"] = trial.instance

            # TODO: Use submit run
            _, cost, _, _ = self._runner.run(config, **kwargs)
            costs += [cost]

        np_costs = np.array(costs)
        return np.mean(np_costs, axis=0)

    def print_stats(self) -> None:
        """Prints all statistics."""
        logger.info(
            "\n"
            f"--- STATISTICS -------------------------------------\n"
            f"--- Incumbent changed: {self._incumbent_changed - 1}\n"
            f"--- Submitted trials: {self.submitted_trials} / {self._scenario.n_trials}\n"
            f"--- Finished trials: {self.finished_trials} / {self._scenario.n_trials}\n"
            f"--- Configurations: {self._n_configs}\n"
            f"--- Used wallclock time: {round(self.used_walltime)} / {self._scenario.walltime_limit} sec\n"
            "--- Used target function runtime: "
            f"{round(self.used_target_function_walltime, 2)} / {self._scenario.cputime_limit} sec\n"
            f"----------------------------------------------------"
        )
