import typing

import logging
import os
import time

import numpy as np

from smac.callbacks import IncorporateRunResultCallback
from smac.configspace import Configuration
from smac.epm.base_epm import AbstractEPM
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.abstract_racer import AbstractRacer, RunInfoIntent
from smac.optimizer import pSMAC
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.optimizer.random_configuration_chooser import (
    ChooserNoCoolDown,
    RandomConfigurationChooser,
)
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae import FirstRunCrashedException, StatusType, TAEAbortException
from smac.tae.base import BaseRunner
from smac.utils.constants import MAXINT
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator

__author__ = "Aaron Klein, Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"


class SMBO(object):
    """Interface that contains the main Bayesian optimization loop.

    Parameters
    ----------
    scenario: smac.scenario.scenario.Scenario
        Scenario object
    stats: Stats
        statistics object with configuration budgets
    initial_design: InitialDesign
        initial sampling design
    runhistory: RunHistory
        runhistory with all runs so far
    runhistory2epm : AbstractRunHistory2EPM
        Object that implements the AbstractRunHistory2EPM to convert runhistory
        data into EPM data
    intensifier: Intensifier
        intensification of new challengers against incumbent configuration
        (probably with some kind of racing on the instances)
    num_run: int
        id of this run (used for pSMAC)
    model: AbstractEPM
        empirical performance model
    acq_optimizer: AcquisitionFunctionMaximizer
        Optimizer of acquisition function.
    acquisition_func : AcquisitionFunction
        Object that implements the AbstractAcquisitionFunction (i.e., infill criterion for acq_optimizer)
    restore_incumbent: Configuration
        incumbent to be used from the start. ONLY used to restore states.
    rng: np.random.RandomState
        Random number generator
    tae_runner : smac.tae.base.BaseRunner Object
        target algorithm run executor
    random_configuration_chooser
        Chooser for random configuration -- one of
        * ChooserNoCoolDown(modulus)
        * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)
    predict_x_best: bool
        Choose x_best for computing the acquisition function via the model instead of via the observations.
    min_samples_model: int
        Minimum number of samples to build a model.

    Attributes
    ----------
    logger
    incumbent
    scenario
    config_space
    stats
    initial_design
    runhistory
    intensifier
    num_run
    rng
    initial_design_configs
    epm_chooser
    tae_runner
    """

    def __init__(
        self,
        scenario: Scenario,
        stats: Stats,
        initial_design: InitialDesign,
        runhistory: RunHistory,
        runhistory2epm: AbstractRunHistory2EPM,
        intensifier: AbstractRacer,
        num_run: int,
        model: AbstractEPM,
        acq_optimizer: AcquisitionFunctionMaximizer,
        acquisition_func: AbstractAcquisitionFunction,
        rng: np.random.RandomState,
        tae_runner: BaseRunner,
        restore_incumbent: Configuration = None,
        random_configuration_chooser: RandomConfigurationChooser = ChooserNoCoolDown(modulus=2.0),
        predict_x_best: bool = True,
        min_samples_model: int = 1,
    ):

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.incumbent = restore_incumbent

        self.scenario = scenario
        self.config_space = scenario.cs  # type: ignore[attr-defined] # noqa F821
        self.stats = stats
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.intensifier = intensifier
        self.num_run = num_run
        self.rng = rng
        self._min_time = 10**-5
        self.tae_runner = tae_runner

        self.initial_design_configs = []  # type: typing.List[Configuration]

        # TODO: consider if we need an additional EPMChooser for multi-objective optimization
        self.epm_chooser = EPMChooser(
            scenario=scenario,
            stats=stats,
            runhistory=runhistory,
            runhistory2epm=runhistory2epm,
            model=model,  # type: ignore
            acq_optimizer=acq_optimizer,
            acquisition_func=acquisition_func,
            rng=rng,
            restore_incumbent=restore_incumbent,
            random_configuration_chooser=random_configuration_chooser,
            predict_x_best=predict_x_best,
            min_samples_model=min_samples_model,
        )

        # Internal variable - if this is set to True it will gracefully stop SMAC
        self._stop = False

        # Callbacks. All known callbacks have a key. If something does not have a key here, there is
        # no callback available.
        self._callbacks = {"_incorporate_run_results": list()}  # type: typing.Dict[str, typing.List[typing.Callable]]
        self._callback_to_key = {
            IncorporateRunResultCallback: "_incorporate_run_results",
        }  # type: typing.Dict[typing.Type, str]

    def start(self) -> None:
        """Starts the Bayesian Optimization loop.

        Detects whether the optimization is restored from a previous state.
        """
        self.stats.start_timing()

        # Initialization, depends on input
        if self.stats.submitted_ta_runs == 0 and self.incumbent is None:
            self.logger.info("Running initial design")
            # Intensifier initialization
            self.initial_design_configs = self.initial_design.select_configurations()

            # to be on the safe side, never return an empty list of initial configs
            if not self.initial_design_configs:
                self.initial_design_configs = [self.config_space.get_default_configuration()]

        elif self.stats.submitted_ta_runs > 0 and self.incumbent is None:
            raise ValueError(
                "According to stats there have been runs started, "
                "but the optimizer cannot detect an incumbent. Did "
                "you set the incumbent (e.g. after restoring state)?"
            )
        elif self.stats.submitted_ta_runs == 0 and self.incumbent is not None:
            raise ValueError(
                "An incumbent is specified, but there are no runs "
                "recorded as started in the Stats-object. If you're "
                "restoring a state, please provide the Stats-object."
            )
        else:
            # Restoring state!
            self.logger.info(
                "State Restored! Starting optimization with " "incumbent %s",
                self.incumbent,
            )
            self.logger.info("State restored with following budget:")
            self.stats.print_stats()

    def run(self) -> Configuration:
        """Runs the Bayesian optimization loop.

        Returns
        -------
        incumbent: np.array(1, H)
            The best found configuration.
        """
        self.start()

        num_obj = len(self.scenario.multi_objectives)  # type: ignore[attr-defined] # noqa F821

        # Main BO loop
        while True:
            if self.scenario.shared_model:  # type: ignore[attr-defined] # noqa F821
                pSMAC.read(
                    run_history=self.runhistory,
                    output_dirs=self.scenario.input_psmac_dirs,  # type: ignore[attr-defined] # noqa F821
                    configuration_space=self.config_space,
                    logger=self.logger,
                )

            start_time = time.time()

            # sample next configuration for intensification
            # Initial design runs are also included in the BO loop now.
            intent, run_info = self.intensifier.get_next_run(
                challengers=self.initial_design_configs,
                incumbent=self.incumbent,
                chooser=self.epm_chooser,
                run_history=self.runhistory,
                repeat_configs=self.intensifier.repeat_configs,
                num_workers=self.tae_runner.num_workers(),
            )

            # remove config from initial design challengers to not repeat it again
            self.initial_design_configs = [c for c in self.initial_design_configs if c != run_info.config]

            # update timebound only if a 'new' configuration is sampled as the challenger
            if self.intensifier.num_run == 0:
                time_spent = time.time() - start_time
                time_left = self._get_timebound_for_intensification(time_spent, update=False)
                self.logger.debug("New intensification time bound: %f", time_left)
            else:
                old_time_left = time_left
                time_spent = time_spent + (time.time() - start_time)
                time_left = self._get_timebound_for_intensification(time_spent, update=True)
                self.logger.debug(
                    "Updated intensification time bound from %f to %f",
                    old_time_left,
                    time_left,
                )

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
                    cost=float(MAXINT) if num_obj == 1 else np.full(num_obj, float(MAXINT)),
                    time=0.0,
                    status=StatusType.RUNNING,
                    instance_id=run_info.instance,
                    seed=run_info.seed,
                    budget=run_info.budget,
                )

                run_info.config.config_id = self.runhistory.config_ids[run_info.config]

                self.tae_runner.submit_run(run_info=run_info)

                # There are 2 criteria that the stats object uses to know
                # if the budged was exhausted.
                # The budget time, which can only be known when the run finishes,
                # And the number of ta executions. Because we submit the job at this point,
                # we count this submission as a run. This prevent for using more
                # runner runs than what the scenario allows
                self.stats.submitted_ta_runs += 1

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
                self.tae_runner.wait()
            else:
                raise NotImplementedError("No other RunInfoIntent has been coded!")

            # Check if there is any result, or else continue
            for run_info, result in self.tae_runner.get_finished_runs():

                # Add the results of the run to the run history
                # Additionally check for new incumbent
                self._incorporate_run_results(run_info, result, time_left)

            if self.scenario.shared_model:  # type: ignore[attr-defined] # noqa F821
                assert self.scenario.output_dir_for_this_run is not None  # please mypy
                pSMAC.write(
                    run_history=self.runhistory,
                    output_directory=self.scenario.output_dir_for_this_run,  # type: ignore[attr-defined] # noqa F821
                    logger=self.logger,
                )

            self.logger.debug(
                "Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)"
                % (
                    self.stats.get_remaing_time_budget(),
                    self.stats.get_remaining_ta_budget(),
                    self.stats.get_remaining_ta_runs(),
                )
            )

            if self.stats.is_budget_exhausted() or self._stop:
                if self.stats.is_budget_exhausted():
                    self.logger.debug("Exhausted configuration budget")
                else:
                    self.logger.debug("Shutting down because a configuration or callback returned status STOP")

                # The budget can be exhausted  for 2 reasons: number of ta runs or
                # time. If the number of ta runs is reached, but there is still budget,
                # wait for the runs to finish
                while self.tae_runner.pending_runs():

                    self.tae_runner.wait()

                    for run_info, result in self.tae_runner.get_finished_runs():
                        # Add the results of the run to the run history
                        # Additionally check for new incumbent
                        self._incorporate_run_results(run_info, result, time_left)

                # Break from the intensification loop,
                # as there are no more resources
                break

            # print stats at the end of each intensification iteration
            if self.intensifier.iteration_done:
                self.stats.print_stats(debug_out=True)

        return self.incumbent

    def validate(
        self,
        config_mode: typing.Union[str, typing.List[Configuration]] = "inc",
        instance_mode: typing.Union[str, typing.List[str]] = "train+test",
        repetitions: int = 1,
        use_epm: bool = False,
        n_jobs: int = -1,
        backend: str = "threading",
    ) -> RunHistory:
        """Create validator-object and run validation, using scenario- information, runhistory from
        smbo and tae_runner from intensify.

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
            assert self.scenario.output_dir_for_this_run is not None  # Please mypy
            traj_fn = os.path.join(self.scenario.output_dir_for_this_run, "traj_aclib2.json")
            trajectory = TrajLogger.read_traj_aclib_format(
                fn=traj_fn, cs=self.config_space
            )  # type: typing.Optional[typing.List[typing.Dict[str, typing.Union[float, int, Configuration]]]]
        else:
            trajectory = None
        if self.scenario.output_dir_for_this_run:
            new_rh_path = os.path.join(
                self.scenario.output_dir_for_this_run, "validated_runhistory.json"
            )  # type: typing.Optional[str] # noqa E501
        else:
            new_rh_path = None

        validator = Validator(self.scenario, trajectory, self.rng)
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
                self.tae_runner,
                output_fn=new_rh_path,
            )
        return new_rh

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
        frac_intensify = self.scenario.intensification_percentage  # type: ignore[attr-defined] # noqa F821
        if frac_intensify <= 0 or frac_intensify >= 1:
            raise ValueError(
                "The value for intensification_percentage-" "option must lie in (0,1), instead: %.2f" % frac_intensify
            )
        total_time = time_spent / (1 - frac_intensify)
        time_left = frac_intensify * total_time
        self.logger.debug(
            "Total time: %.4f, time spent on choosing next "
            "configurations: %.4f (%.2f), time left for "
            "intensification: %.4f (%.2f)" % (total_time, time_spent, (1 - frac_intensify), time_left, frac_intensify)
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
        # update SMAC stats
        self.stats.ta_time_used += float(result.time)
        self.stats.finished_ta_runs += 1

        self.logger.debug(
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
            raise TAEAbortException(
                "Target algorithm status ABORT - SMAC will "
                "exit. The last incumbent can be found "
                "in the trajectory-file."
            )
        elif result.status == StatusType.STOP:
            self._stop = True
            return

        if self.scenario.abort_on_first_run_crash:  # type: ignore[attr-defined] # noqa F821
            if self.stats.finished_ta_runs == 1 and result.status == StatusType.CRASHED:
                raise FirstRunCrashedException(
                    "First run crashed, abort. Please check your setup -- we assume that your default "
                    "configuration does not crashes. (To deactivate this exception, use the SMAC scenario option "
                    "'abort_on_first_run_crash'). Additional run info: %s" % result.additional_info
                )

        # Update the intensifier with the result of the runs
        self.incumbent, inc_perf = self.intensifier.process_results(
            run_info=run_info,
            incumbent=self.incumbent,
            run_history=self.runhistory,
            time_bound=max(self._min_time, time_left),
            result=result,
        )

        for callback in self._callbacks["_incorporate_run_results"]:
            response = callback(smbo=self, run_info=run_info, result=result, time_left=time_left)
            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called
            if response is False:
                self.logger.debug("An IncorporateRunResultCallback returned False, requesting abort.")
                self._stop = True

        if self.scenario.save_instantly:  # type: ignore[attr-defined] # noqa F821
            self.save()

        return

    def save(self) -> None:
        """Saves the current stats and runhistory."""
        self.stats.save()

        output_dir = self.scenario.output_dir_for_this_run
        if output_dir is not None:
            self.runhistory.save_json(fn=os.path.join(output_dir, "runhistory.json"))
