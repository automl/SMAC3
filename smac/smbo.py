from __future__ import annotations

from typing import Callable, Type

import time

import numpy as np

from smac.acquisition_function import AbstractAcquisitionFunction
from smac.acquisition_optimizer import AbstractAcquisitionOptimizer
from smac.callbacks.callbacks import IncorporateRunResultCallback
from smac.chooser import Chooser
from smac.chooser.random_chooser import RandomChooser
from smac.configspace import Configuration
from smac.constants import MAXINT
from smac.initial_design import InitialDesign
from smac.intensification.abstract_racer import AbstractRacer, RunInfoIntent
from smac.model.base_model import BaseModel
from smac.runhistory import RunInfo, RunValue
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory_transformer import AbstractRunhistoryTransformer
from smac.runner import (
    FirstRunCrashedException,
    Runner,
    StatusType,
    TargetAlgorithmAbortException,
)
from smac.scenario import Scenario
from smac.utils.logging import get_logger
from smac.utils.stats import Stats
from smac.utils.validate import Validator

__author__ = "Aaron Klein, Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class SMBO:
    """Interface that contains the main Bayesian optimization loop.

    Parameters
    ----------
    scenario: smac.config.config.config
        scenario object
    stats: Stats
        statistics object with configuration budgets
    initial_design: InitialDesign
        initial sampling design
    runhistory: RunHistory
        runhistory with all runs so far
    runhistory_transformer : Abstractrunhistory_transformer
        Object that implements the Abstractrunhistory_transformer to convert runhistory
        data into EPM data
    intensifier: Intensifier
        intensification of new challengers against incumbent configuration
        (probably with some kind of racing on the instances)
    model: BaseEPM
        empirical performance model
    acquisition_optimizer: AcquisitionFunctionMaximizer
        Optimizer of acquisition function.
    acquisition_function : AcquisitionFunction
        Object that implements the AbstractAcquisitionFunction (i.e., infill criterion for acquisition_optimizer)
    restore_incumbent: Configuration
        incumbent to be used from the start. ONLY used to restore states.
    rng: np.random.RandomState
        Random number generator
    runner : smac.tae.base.BaseRunner Object
        target algorithm run executor
    random_configuration_chooser
        Chooser for random configuration -- one of
        * ChooserNoCoolDown(modulus)
        * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)
    predict_x_best: bool
        Choose x_best for computing the acquisition function via the model instead of via the observations.
    min_samples_model: int
        Minimum number of samples to build a model.
    configuration_chooser_kwargs: typing.Optional[typing.Dict]
        Additional arguments passed to epmchooser

    Attributes
    ----------
    incumbent
    config
    config_space
    stats
    initial_design
    runhistory
    intensifier
    rng
    initial_design_configs
    configuration_chooser
    runner
    """

    def __init__(
        self,
        scenario: Scenario,
        stats: Stats,
        runner: Runner,
        initial_design: InitialDesign,
        runhistory: RunHistory,
        runhistory_transformer: AbstractRunhistoryTransformer,
        intensifier: AbstractRacer,
        model: BaseModel,
        acquisition_optimizer: AbstractAcquisitionOptimizer,
        acquisition_function: AbstractAcquisitionFunction,
        configuration_chooser: Chooser,
        random_configuration_chooser: RandomChooser,
        seed: int = 0,
    ):
        # Changed in 2.0: We don't restore the incumbent anymore but derive it directly from
        # the stats object when the run is started.
        self.incumbent = None

        self.scenario = scenario
        self.configspace = scenario.configspace
        self.stats = stats
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.runhistory_transformer = runhistory_transformer
        self.intensifier = intensifier
        self.model = model
        self.acquisition_optimizer = acquisition_optimizer
        self.acquisition_function = acquisition_function
        self.configuration_chooser = configuration_chooser
        self.random_configuration_chooser = random_configuration_chooser
        self.runner = runner

        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.initial_design_configs: list[Configuration] = []

        # Callbacks. All known callbacks have a key. If something does not have a key here, there is
        # no callback available.
        self._callbacks: dict[str, list[Callable]] = {"_incorporate_run_results": list()}
        self._callback_to_key: dict[Type, str] = {
            IncorporateRunResultCallback: "_incorporate_run_results",
        }

        # Internal variable - if this is set to True it will gracefully stop SMAC
        self._stop = False
        self._min_time = 10**-5

    def start(self) -> None:
        """Starts the Bayesian Optimization loop.

        Detects whether the optimization is restored from a previous state.
        """
        self.stats.start_timing()

        # Initialization, depends on input
        if self.stats.submitted == 0 and self.incumbent is None:
            logger.info("Running initial design...")
            # Intensifier initialization
            self.initial_design_configs = self.initial_design.select_configurations()

            # to be on the safe side, never return an empty list of initial configs
            if not self.initial_design_configs:
                self.initial_design_configs = [self.configspace.get_default_configuration()]
        else:
            # Restoring state!
            assert self.incumbent is not None
            logger.info(f"State restored! Starting optimization with incumbent {self.incumbent.get_dictionary()}.")
            self.stats.print()

    def run(self) -> Configuration:
        """Runs the Bayesian optimization loop.

        Returns
        -------
        incumbent: np.array(1, H)
            The best found configuration.
        """
        # Make sure we use the right incumbent
        self.incumbent = self.stats.get_incumbent()

        self.start()
        n_objectives = self.scenario.count_objectives()

        # Main BO loop
        while True:
            start_time = time.time()

            # sample next configuration for intensification
            # Initial design runs are also included in the BO loop now.
            intent, run_info = self.intensifier.get_next_run(
                challengers=self.initial_design_configs,
                incumbent=self.incumbent,
                chooser=self.configuration_chooser,
                runhistory=self.runhistory,
                repeat_configs=self.intensifier.repeat_configs,
                num_workers=self.runner.num_workers(),
            )

            # remove config from initial design challengers to not repeat it again
            self.initial_design_configs = [c for c in self.initial_design_configs if c != run_info.config]

            # update timebound only if a 'new' configuration is sampled as the challenger
            if self.intensifier.run_id == 0:
                time_spent = time.time() - start_time
                time_left = self._get_timebound_for_intensification(time_spent, update=False)
                logger.debug("New intensification time bound: %f", time_left)
            else:
                old_time_left = time_left
                time_spent = time_spent + (time.time() - start_time)
                time_left = self._get_timebound_for_intensification(time_spent, update=True)
                logger.debug(f"Updated intensification time bound from {old_time_left} to {time_left}")

            # Skip starting new runs if the budget is now exhausted
            if self.stats.is_budget_exhausted():
                intent = RunInfoIntent.SKIP

            # Skip the run if there was a request to do so.
            # For example, during intensifier intensification, we
            # don't want to rerun a config that was previously ran
            if intent == RunInfoIntent.RUN:
                # Track the fact that a run was launched in the run
                # history. It's status is tagged as RUNNING, and once
                # completed and processed, it will be updated accordingly
                self.runhistory.add(
                    config=run_info.config,
                    cost=float(MAXINT) if n_objectives == 1 else np.full(n_objectives, float(MAXINT)),
                    time=0.0,
                    status=StatusType.RUNNING,
                    instance_id=run_info.instance,
                    seed=run_info.seed,
                    budget=run_info.budget,
                )

                run_info.config.config_id = self.runhistory.config_ids[run_info.config]
                self.runner.submit_run(run_info=run_info)

                # There are 2 criteria that the stats object uses to know
                # if the budged was exhausted.
                # The budget time, which can only be known when the run finishes,
                # And the number of ta executions. Because we submit the job at this point,
                # we count this submission as a run. This prevent for using more
                # runner runs than what the config allows
                self.stats.submitted += 1

            elif intent == RunInfoIntent.SKIP:
                # No launch is required
                # This marks a transition request from the intensifier
                # To a new iteration
                pass
            elif intent == RunInfoIntent.WAIT:
                # In any other case, we wait for resources
                # This likely indicates that no further decision
                # can be taken by the intensifier until more data is
                # available
                self.runner.wait()
            else:
                raise NotImplementedError("No other RunInfoIntent has been coded!")

            # Check if there is any result, or else continue
            for run_info, result in self.runner.get_finished_runs():
                # Add the results of the run to the run history
                # Additionally check for new incumbent
                self._incorporate_run_results(run_info, result, time_left)

            logger.debug(
                "Remaining budget: %f (wallclock time), %f (target algorithm time), %f (target algorithm runs)"
                % (
                    self.stats.get_remaing_time_budget(),
                    self.stats.get_remaining_target_algorithm_budget(),
                    self.stats.get_remaining_target_algorithm_runs(),
                )
            )

            if self.stats.is_budget_exhausted() or self._stop:
                if self.stats.is_budget_exhausted():
                    logger.debug("Configuration budget is exhausted.")
                else:
                    logger.debug("Shutting down because a configuration or callback returned status STOP.")

                # The budget can be exhausted  for 2 reasons: number of ta runs or
                # time. If the number of ta runs is reached, but there is still budget,
                # wait for the runs to finish.
                while self.runner.pending_runs():

                    self.runner.wait()

                    for run_info, result in self.runner.get_finished_runs():
                        # Add the results of the run to the run history
                        # Additionally check for new incumbent
                        self._incorporate_run_results(run_info, result, time_left)

                # Break from the intensification loop, as there are no more resources.
                break

            # Print stats at the end of each intensification iteration.
            if self.intensifier.iteration_done:
                self.stats.print()

        return self.incumbent

    '''
    # TODO: Is this still needed?
    # I think it is important when using instances
    def validate(
        self,
        config_mode: Union[str, List[Configuration]] = "inc",
        instance_mode: Union[str, List[str]] = "train+test",
        repetitions: int = 1,
        use_epm: bool = False,
        n_jobs: int = -1,
        backend: str = "threading",
    ) -> RunHistory:
        """Create validator-object and run validation, using config- information, runhistory from
        smbo and runner from intensify.

        Parameters
        ----------
        config_mode: str or list<Configuration>
            string or directly a list of Configuration
            str from [def, inc, def+inc, wallclock_time, cpu_time, all]
            time evaluates at cpu- or wallclock-timesteps of:
            [max_time/2^0, max_time/2^1, max_time/2^3, ..., default]
            with max_time being the highest recorded time
        instance_mode: string
            what instances to use for validation, from [train, test, train+test]
        repetitions: int
            number of repetitions in nondeterministic algorithms (in
            deterministic will be fixed to 1)
        use_epm: bool
            whether to use an EPM instead of evaluating all runs with the TAE
        n_jobs: int
            number of parallel processes used by joblib

        Returns
        -------
        runhistory: RunHistory
            runhistory containing all specified runs
        """
        if isinstance(config_mode, str):
            assert self.config.output_directory is not None  # Please mypy
            traj_fn = os.path.join(self.config.output_directory, "traj_aclib2.json")
            trajectory = TrajLogger.read_traj_aclib_format(
                fn=traj_fn, cs=self.configspace
            )  # type: Optional[List[Dict[str, Union[float, int, Configuration]]]]
        else:
            trajectory = None
        if self.config.output_directory:
            new_rh_path = os.path.join(
                self.config.output_directory, "validated_runhistory.json"
            )  # type: Optional[str] # noqa E501
        else:
            new_rh_path = None

        validator = Validator(self.config, trajectory, self.rng)
        if use_epm:
            new_rh = validator.validate_epm(
                config_mode=config_mode,
                instance_mode=instance_mode,
                repetitions=repetitions,
                runhistory=self.runhistory,
                output_fn=new_rh_path,
            )
        else:
            new_rh = validator.validate(
                config_mode,
                instance_mode,
                repetitions,
                n_jobs,
                backend,
                self.runhistory,
                self.runner,
                output_fn=new_rh_path,
            )
        return new_rh
    '''

    def _get_timebound_for_intensification(self, time_spent: float, update: bool) -> float:
        """Calculate time left for intensify from the time spent on choosing challengers using the
        fraction of time intended for intensification (which is specified in
        scenario.intensification_percentage).

        Parameters
        ----------
        time_spent : float

        update : bool
            Only used to check in the unit tests how this function was called

        Returns
        -------
        time_left : float
        """
        frac_intensify = self.scenario.intensify_percentage
        total_time = time_spent / (1 - frac_intensify)
        time_left = frac_intensify * total_time

        logger.debug(
            f"\n--- Total time: {round(total_time, 4)}"
            f"\n--- Time spent on choosing next configurations: {round(time_spent, 4)} ({(1 - frac_intensify)})"
            f"\n--- Time left for intensification: {round(time_left, 4)} ({frac_intensify})"
        )
        return time_left

    def _incorporate_run_results(self, run_info: RunInfo, result: RunValue, time_left: float) -> None:
        """The SMBO submits a config-run-request via a RunInfo object. When that config run is
        completed, a RunValue, which contains all the relevant information obtained after running a
        job, is returned. This method incorporates the status of that run into the stats/runhistory
        objects so that other consumers can advance with their task.

        Additionally, it checks for a new incumbent via the intensifier process results,
        which also has the side effect of moving the intensifier to a new state

        Parameters
        ----------
        run_info: RunInfo
            Describes the run (config) from which to process the results
        result: RunValue
            Contains relevant information regarding the execution of a config
        time_left: float
            time in [sec] available to perform intensify
        """
        # We removed `abort_on_first_run_crash` and therefore we expect the first
        # run to always succeed.
        if self.stats.finished == 0 and result.status == StatusType.CRASHED:
            additional_info = ""
            if "traceback" in result.additional_info:
                additional_info = "\n\n" + result.additional_info["traceback"]

            raise FirstRunCrashedException("The first run crashed. Please check your setup again." + additional_info)

        # Update SMAC stats
        self.stats.target_algorithm_walltime_used += float(result.time)
        self.stats.finished += 1

        logger.debug(
            f"Return: Status: {result.status}, cost: {result.cost}, time: {result.time}, "
            f"additional: {result.additional_info}"
        )

        self.runhistory.add(
            config=run_info.config,
            cost=result.cost,
            time=result.time,
            status=result.status,
            instance_id=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
            starttime=result.starttime,
            endtime=result.endtime,
            force_update=True,
            additional_info=result.additional_info,
        )
        self.stats.n_configs = len(self.runhistory.config_ids)

        if result.status == StatusType.ABORT:
            raise TargetAlgorithmAbortException(
                "The target algorithm was aborted. The last incumbent can be found in the trajectory file."
            )
        elif result.status == StatusType.STOP:
            self._stop = True
            return

        # Update the intensifier with the result of the runs
        self.incumbent, _ = self.intensifier.process_results(
            run_info=run_info,
            incumbent=self.incumbent,
            runhistory=self.runhistory,
            time_bound=max(self._min_time, time_left),
            result=result,
        )

        for callback in self._callbacks["_incorporate_run_results"]:
            response = callback(smbo=self, run_info=run_info, result=result, time_left=time_left)

            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called.
            if response is False:
                logger.debug("An IncorporateRunResultCallback returned False. Abort is requested.")
                self._stop = True

        # We always save immediately
        # TODO: Performance issues if we always save?
        self.save()

        return

    def save(self) -> None:
        """Saves the current stats and runhistory."""
        self.stats.save()

        path = self.scenario.output_directory
        if path is not None:
            self.runhistory.save_json(fn=str(path / "runhistory.json"))
