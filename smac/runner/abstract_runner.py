from __future__ import annotations

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator

import time
import traceback
import inspect

import numpy as np

import smac
from smac.configspace import Configuration
from smac.runhistory import TrialInfo, TrialValue, StatusType
from smac.utils.logging import get_logger
from smac.stats import Stats

logger = get_logger(__name__)


class AbstractRunner(ABC):
    """Interface class to handle the execution of SMAC configurations.

    This interface defines how to interact with the SMBO loop.
    The complexity of running a configuration as well as handling the
    results is abstracted to the SMBO via a AbstractRunner.

    From SMBO perspective, launching a configuration follows a
    submit/collect scheme as follows:

    1. A run is launched via submit_run()

       1. Submit_run internally calls run_wrapper(), a method that
          contains common processing functions among different runners,
          for example, handling capping and stats checking.

       2. A class that implements AbstractRunner defines run() which is
          really the algorithm to translate a RunInfo to a RunValue, i.e.
          a configuration to an actual result.

    2. A completed run is collected via iter_results(), which iterates and
       consumes any finished runs, if any.

    3. This interface also offers the method wait() as a mechanism to make
       sure we have enough data in the next iteration to make a decision. For
       example, the intensifier might not be able to select the next challenger
       until more results are available.

    Parameters
    ----------
    target_algorithm : Callable
        The target algorithm to be run
    scenario: Scenario
        The scenario describes the runtime of SMAC
    stats: Stats
         stats object to collect statistics about runtime/additional info

    Attributes
    ----------
    scenario: Scenario
        The scenario the runner will use
    results: list[tuple[RunInfo, RunValue]]
        The RunInfo is an object containing the configuration and the necessary data
        to run it while the RunValue contains information about the status/performance
        of config.
    target_algorithm: Callable
        The algorithm to be called
    stats: Stats
        Where stats are collected during the run
    crash_cost: float | list[float]
        The cost to give if the target_algorithm crashes or a list of costs if doing
        multi-objective
    objectives: str | list[str]
        The name of the objective or objectives if doing multi-objective
    n_objectives: int
        The number of objectives the target_algorithm uses
    """

    def __init__(
        self,
        target_algorithm: Callable,
        scenario: smac.scenario.Scenario,
        stats: Stats,
        required_arguments: list[str],
    ):
        self.scenario = scenario

        # The results is a FIFO structure, implemented via a list
        # (because the Queue lock is not pickable). Finished runs are
        # put in this list and collected via _process_pending_runs
        self._results_queue: list[tuple[TrialInfo, TrialValue]] = []

        # Below state the support for a Runner algorithm that implements a ta
        self.target_algorithm = target_algorithm
        self.stats = stats
        self.crash_cost = scenario.crash_cost
        self._supports_memory_limit = False

        if isinstance(scenario.objectives, str):
            objectives = [scenario.objectives]
        else:
            objectives = scenario.objectives

        self.objectives = objectives
        self.n_objectives = scenario.count_objectives()

        # Check if target algorithm is callable
        if not callable(self.target_algorithm):
            raise TypeError(
                "Argument `target_algorithm` must be a callable but is type" f"`{type(self.target_algorithm)}`."
            )

        # Signatures here
        signature = inspect.signature(self.target_algorithm).parameters
        for argument in required_arguments:
            if argument not in signature.keys():
                raise RuntimeError(
                    f"Target function needs to have the arguments {required_arguments} "
                    f"but could not found {argument}."
                )

        # Now we check for additional arguments which are not used by SMAC
        # However, we only want to warn the user and not
        for key in list(signature.keys())[1:]:
            if key not in required_arguments:
                logger.warning(f"The argument {key} is not set by SMAC. Consider removing it.")

        self._required_arguments = required_arguments

    def run_wrapper(self, run_info: TrialInfo) -> tuple[TrialInfo, TrialValue]:
        """Wrapper around run() to exec and check the execution of a given config file.

        This function encapsulates common handling/processing, so that run() implementation
        is simplified.

        Parameters
        ----------
        run_info : RunInfo
            Object that contains enough information to execute a configuration run in
            isolation.

        Returns
        -------
        RunInfo:
            An object containing the configuration launched.
        RunValue:
            Contains information about the status/performance of config.
        """
        start = time.time()

        try:
            status, cost, runtime, additional_info = self.run(
                config=run_info.config,
                instance=run_info.instance,
                budget=run_info.budget,
                seed=run_info.seed,
            )
        except Exception as e:
            status = StatusType.CRASHED
            cost = self.crash_cost
            runtime = time.time() - start

            # Add context information to the error message
            exception_traceback = traceback.format_exc()
            error_message = repr(e)
            additional_info = {"traceback": exception_traceback, "error": error_message}

        end = time.time()

        if run_info.budget == 0 and status == StatusType.DONOTADVANCE:
            raise ValueError("Cannot handle DONOTADVANCE state when using intensify or SH/HB on instances.")

        # Catch NaN or inf.
        if not np.all(np.isfinite(cost)):
            logger.warning(
                "Target algorithm returned infinity or nothing at all. Result is treated as CRASHED"
                f" and cost is set to {self.crash_cost}."
            )
            status = StatusType.CRASHED

        if status == StatusType.CRASHED:
            cost = self.crash_cost

        run_value = TrialValue(
            status=status,
            cost=cost,
            time=runtime,
            additional_info=additional_info,
            starttime=start,
            endtime=end,
        )
        return run_info, run_value

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "code": str(self.target_algorithm.__code__.co_code),
        }

    @abstractmethod
    def submit_run(self, run_info: TrialInfo) -> None:
        """This function submits a configuration embedded in a RunInfo object, and uses one of the
        workers to produce a result (such result will eventually be available on the self._results_queue
        FIFO).

        This interface method will be called by SMBO, with the expectation
        that a function will be executed by a worker.

        What will be executed is dictated by run_info, and "how" will it be
        executed is decided via the child class that implements a run() method.

        Because config submission can be a serial/parallel endeavor,
        it is expected to be implemented by a child class.

        Parameters
        ----------
        run_info: RunInfo
            An object containing the configuration and the necessary data to run it
        """
        ...

    @abstractmethod
    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        budget: float | None = None,
        seed: int = 0,
    ) -> tuple[StatusType, float | list[float], float, dict]:
        """Runs the target algorithm with a configuration on a single instance with instance specifics.

        Parameters
        ----------
        config : Configuration
            Configuration to be passed to the target algorithm.
        instance : str | None, defaults to None
            The Problem instance.
        seed : int, defaults to 0
            Random seed.
        budget : float | None, defaults to None
            A positive, real-valued number representing an arbitrary limit to the
            target algorithm handled by the target algorithm internally.

        Returns
        -------
        status : StatusType
            Status of the run.
        cost : float | list[float]
            Cost, regret, quality, etc. of the run.
        runtime : float
            The time the target algorithm took to run.
        additional_info : dict
            All further additional run information.
        """
        ...

    @abstractmethod
    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:
        """This method returns any finished configuration, and returns a list with the
        results of exercising the configurations. This class keeps populating results
        to self._results_queue until a call to get_finished runs is done. In this case,
        the self._results_queue list is emptied and all RunValues produced by running
        run() are returned.

        Returns
        -------
        Iterator[tuple[RunInfo, RunValue]]:
            A list of pais RunInfo/RunValues a submitted configuration
        """
        ...

    @abstractmethod
    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.

        This method waits until 1 run completes
        """
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Whether or not there are configs still running.

        Generally if the runner is serial, launching a run instantly returns it's result. On
        parallel runners, there might be pending configurations to complete.
        """
        ...

    @abstractmethod
    def available_worker_count(self) -> int:
        """Return the number of available workers"""
        ...
