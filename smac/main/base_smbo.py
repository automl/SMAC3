from __future__ import annotations
from abc import abstractmethod

from typing import Iterator

import time

import numpy as np

from smac.acquisition.abstract_acqusition_optimizer import AbstractAcquisitionOptimizer
from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.callback import Callback
from smac.configspace import Configuration
from smac.constants import MAXINT
from smac.initial_design import InitialDesign
from smac.intensification.abstract_intensifier import AbstractIntensifier
from smac.model.base_model import BaseModel
from smac.runhistory import RunInfo, RunInfoIntent, RunValue, StatusType
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.runhistory.runhistory import RunHistory
from smac.runner.runner import AbstractRunner
from smac.scenario import Scenario
from smac.utils.logging import get_logger
from smac.utils.stats import Stats
from smac.random_design.random_design import RandomDesign

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class BaseSMBO:
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
    runhistory_encoder : Abstractrunhistory_encoder
        Object that implements the Abstractrunhistory_encoder to convert runhistory
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
    runner : AbstractRunner
        target algorithm run executor
    random_design
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
    runner
    """

    def __init__(
        self,
        scenario: Scenario,
        stats: Stats,
        runner: AbstractRunner,
        initial_design: InitialDesign,
        runhistory: RunHistory,
        runhistory_encoder: RunHistoryEncoder,
        intensifier: AbstractIntensifier,
        model: BaseModel,
        acquisition_optimizer: AbstractAcquisitionOptimizer,
        acquisition_function: AbstractAcquisitionFunction,
        random_design: RandomDesign,
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
        self.runhistory_encoder = runhistory_encoder
        self.intensifier = intensifier
        self.model = model
        self.acquisition_optimizer = acquisition_optimizer
        self.acquisition_function = acquisition_function
        self.random_design = random_design
        self.runner = runner

        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.initial_design_configs: list[Configuration] = []

        # Internal variable - if this is set to True it will gracefully stop SMAC
        self._finished = False
        self._stop = False
        self._min_time = 10**-5
        self._callbacks: list[Callback] = []

    def _start(self) -> None:
        """Starts the Bayesian Optimization loop and detects whether the optimization is restored
        from a previous state."""
        self.stats.start_timing()

        # Initialization, depends on input
        if self.stats.submitted == 0 and self.incumbent is None:
            if len(self.runhistory) == 0:
                logger.info("Running initial design...")
                # Intensifier initialization
                self.initial_design_configs = self.initial_design.select_configurations()

                # to be on the safe side, never return an empty list of initial configs
                if not self.initial_design_configs:
                    self.initial_design_configs = [self.configspace.get_default_configuration()]
            else:
                logger.info(
                    f"Initial design is skipped since {len(self.runhistory)} entries are found in the runhistory."
                )

                self.incumbent = self.runhistory.get_incumbent()
                self.initial_design_configs = self.runhistory.get_configs()

                logger.info(
                    f"Added {len(self.initial_design_configs)} configs from the runhistory as initial design and "
                    "determined the incumbent."
                )

        else:
            # Restoring state!
            if self.incumbent is None:
                raise RuntimeError(
                    "It seems like SMAC restored from a previous state which failed.\n"
                    "Please remove the previous files and try again. "
                    "Alternatively, you can set `overwrite` to true in the facade."
                )

            logger.info(f"State restored! Starting optimization with incumbent {self.incumbent.get_dictionary()}.")
            self.stats.print()

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
        intensifier.intensification_percentage).

        Parameters
        ----------
        time_spent : float

        update : bool
            Only used to check in the unit tests how this function was called

        Returns
        -------
        time_left : float
        """
        frac_intensify = self.intensifier.intensify_percentage
        total_time = time_spent / (1 - frac_intensify)
        time_left = frac_intensify * total_time

        logger.debug(
            f"\n--- Total time: {round(total_time, 4)}"
            f"\n--- Time spent on choosing next configurations: {round(time_spent, 4)} ({(1 - frac_intensify)})"
            f"\n--- Time left for intensification: {round(time_left, 4)} ({frac_intensify})"
        )
        return time_left

    def register_callback(self, callback: Callback) -> None:
        self._callbacks += [callback]

    def run(self) -> Configuration:
        """Runs the Bayesian optimization loop.

        Returns
        -------
        incumbent: np.array(1, H)
            The best found configuration.
        """
        # We return the incumbent if we already finished the optimization process
        if self._finished:
            return self.incumbent

        # Make sure we use the right incumbent
        self.incumbent = self.stats.get_incumbent()

        self._start()
        n_objectives = self.scenario.count_objectives()

        for callback in self._callbacks:
            callback.on_start(self)

        # Main BO loop
        while True:
            for callback in self._callbacks:
                callback.on_iteration_start(self)

            start_time = time.time()

            # Sample next configuration for intensification.
            # Initial design runs are also included in the BO loop now.
            intent, run_info = self.intensifier.get_next_run(
                challengers=self.initial_design_configs,
                incumbent=self.incumbent,
                ask=self.ask,
                runhistory=self.runhistory,
                repeat_configs=self.intensifier.repeat_configs,
                n_workers=self.runner.available_worker_count(),
            )

            # Remove config from initial design challengers to not repeat it again
            self.initial_design_configs = [c for c in self.initial_design_configs if c != run_info.config]

            # Update timebound only if a 'new' configuration is sampled as the challenger
            if self.intensifier.num_run == 0:
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
                    cost=float(MAXINT) if n_objectives == 1 else [float(MAXINT) for _ in range(n_objectives)],
                    time=0.0,
                    status=StatusType.RUNNING,
                    instance=run_info.instance,
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
            for run_info, run_value in self.runner.iter_results():
                # Add the results of the run to the run history
                # Additionally check for new incumbent
                self.tell(run_info, run_value, time_left)

            logger.debug(
                "Remaining budget: %f (wallclock time), %f (target algorithm time), %f (target algorithm runs)"
                % (
                    self.stats.get_remaing_walltime(),
                    self.stats.get_remaining_cputime(),
                    self.stats.get_remaining_trials(),
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
                while self.runner.is_running():
                    self.runner.wait()

                    for run_info, run_value in self.runner.iter_results():
                        # Add the results of the run to the run history
                        # Additionally check for new incumbent
                        self.tell(run_info, run_value, time_left)

                # Break from the intensification loop, as there are no more resources.
                break

            # Gracefully end optimization if termination cost is reached
            if self.scenario.termination_cost_threshold != np.inf:
                if not isinstance(run_value.cost, list):
                    cost = [run_value.cost]
                else:
                    cost = run_value.cost

                if not isinstance(self.scenario.termination_cost_threshold, list):
                    cost_threshold = [self.scenario.termination_cost_threshold]
                else:
                    cost_threshold = self.scenario.termination_cost_threshold

                if len(cost) != len(cost_threshold):
                    raise RuntimeError("You must specify a termination cost threshold for each objective.")

                if all(cost[i] < cost_threshold[i] for i in range(len(cost))):
                    self._stop = True

            for callback in self._callbacks:
                response = callback.on_iteration_end(smbo=self, info=run_info, value=run_value)

                # If a callback returns False, the optimization loop should be interrupted
                # the other callbacks are still being called.
                if response is False:
                    logger.debug("An callback returned False. Abort is requested.")
                    self._stop = True

            # Print stats at the end of each intensification iteration.
            if self.intensifier.iteration_done:
                self.stats.print()

        for callback in self._callbacks:
            callback.on_end(self)

        self._finished = True
        return self.incumbent

    @abstractmethod
    def ask(self) -> Iterator[Configuration]:
        """Choose next candidate solution with Bayesian optimization. The suggested configurations
        depend on the surrogate model acquisition optimizer/function.
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self, run_info: RunInfo, run_value: RunValue, time_left: float, save: bool = True) -> None:
        """The SMBO submits a config-run-request via a RunInfo object. When that config run is
        completed, a RunValue, which contains all the relevant information obtained after running a
        job, is returned. This method incorporates the status of that run into the stats/runhistory
        objects so that other consumers can advance with their task.

        Additionally, it checks for a new incumbent via the intensifier process results,
        which also has the side effect of moving the intensifier to a new state

        Parameters
        ----------
        run_info: RunInfo
            Describes the run (config) from which to process the results.
        result: RunValue
            Contains relevant information regarding the execution of a config.
        time_left: float
            How much time in seconds is left to perform intensification.
        save : bool, optional to True
            Whether the runhistory should be saved.
        """
        raise NotImplementedError

    def save(self) -> None:
        """Saves the current stats and runhistory."""
        self.stats.save()

        path = self.scenario.output_directory
        if path is not None:
            self.runhistory.save_json(str(path / "runhistory.json"))
