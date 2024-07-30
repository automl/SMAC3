from __future__ import annotations

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


from abc import ABC, abstractmethod
from typing import Any, Iterator

import time
import traceback

import numpy as np
from ConfigSpace import Configuration

from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.scenario import Scenario
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class AbstractRunner(ABC):
    """Interface class to handle the execution of SMAC configurations.
    This interface defines how to interact with the SMBO loop.
    The complexity of running a configuration as well as handling the results is abstracted to the
    SMBO via an AbstractRunner.

    From SMBO perspective, launching a configuration follows a
    submit/collect scheme as follows:

    1. A run is launched via ``submit_run()``

       - ``submit_run`` internally calls ``run_wrapper()``, a method that contains common processing functions among
         different runners.
       - A class that implements AbstractRunner defines ``run()`` which is really the algorithm to
         translate a ``TrialInfo`` to a ``TrialValue``, i.e. a configuration to an actual result.
    2. A completed run is collected via ``iter_results()``, which iterates and consumes any finished runs, if any.
    3. This interface also offers the method ``wait()`` as a mechanism to make sure we have enough
       data in the next iteration to make a decision. For example, the intensifier might not be
       able to select the next challenger until more results are available.

    Parameters
    ----------
    scenario : Scenario
    required_arguments : list[str]
        A list of required arguments, which are passed to the target function.
    """

    def __init__(
        self,
        scenario: Scenario,
        required_arguments: list[str] = None,
    ):
        if required_arguments is None:
            required_arguments = []
        self._scenario = scenario
        self._required_arguments = required_arguments

        # The results are a FIFO structure, implemented via a list
        # (because the Queue lock is not pickable). Finished runs are
        # put in this list and collected via _process_pending_runs
        self._results_queue: list[tuple[TrialInfo, TrialValue]] = []
        self._crash_cost = scenario.crash_cost
        self._supports_memory_limit = False

        if isinstance(scenario.objectives, str):
            objectives = [scenario.objectives]
        else:
            objectives = scenario.objectives

        self._objectives = objectives
        self._n_objectives = scenario.count_objectives()

        # We need to exapdn crash cost if the user did not do it
        if self._n_objectives > 1:
            if not isinstance(scenario.crash_cost, list):
                assert isinstance(scenario.crash_cost, float)
                self._crash_cost = [scenario.crash_cost for _ in range(self._n_objectives)]

    def run_wrapper(
        self, trial_info: TrialInfo, **dask_data_to_scatter: dict[str, Any]
    ) -> tuple[TrialInfo, TrialValue]:
        """Wrapper around run() to execute and check the execution of a given config.
        This function encapsulates common
        handling/processing, so that run() implementation is simplified.

        Parameters
        ----------
        trial_info : RunInfo
            Object that contains enough information to execute a configuration run in isolation.
        dask_data_to_scatter: dict[str, Any]
            When a user scatters data from their local process to the distributed network,
            this data is distributed in a round-robin fashion grouping by number of cores.
            Roughly speaking, we can keep this data in memory and then we do not have to (de-)serialize the data
            every time we would like to execute a target function with a big dataset.
            For example, when your target function has a big dataset shared across all the target function,
            this argument is very useful.

        Returns
        -------
        info : TrialInfo
            An object containing the configuration launched.
        value : TrialValue
            Contains information about the status/performance of config.
        """
        start = time.time()

        try:
            status, cost, runtime, additional_info = self.run(
                config=trial_info.config,
                instance=trial_info.instance,
                budget=trial_info.budget,
                seed=trial_info.seed,
                **dask_data_to_scatter,
            )
        except Exception as e:
            status = StatusType.CRASHED
            cost = self._crash_cost
            runtime = time.time() - start

            # Add context information to the error message
            exception_traceback = traceback.format_exc()
            error_message = repr(e)
            additional_info = {
                "traceback": exception_traceback,
                "error": error_message,
            }

        end = time.time()

        # Catch NaN or inf
        if not np.all(np.isfinite(cost)):
            logger.warning(
                "Target function returned infinity or nothing at all. Result is treated as CRASHED"
                f" and cost is set to {self._crash_cost}."
            )

            if "traceback" in additional_info:
                logger.warning(f"Traceback: {additional_info['traceback']}\n")

            status = StatusType.CRASHED

        if status == StatusType.CRASHED:
            cost = self._crash_cost

        trial_value = TrialValue(
            status=status,
            cost=cost,
            time=runtime,
            additional_info=additional_info,
            starttime=start,
            endtime=end,
        )

        return trial_info, trial_value

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta-data of the created object."""
        return {"name": self.__class__.__name__}

    @abstractmethod
    def submit_trial(self, trial_info: TrialInfo) -> None:
        """This function submits a configuration embedded in a TrialInfo object, and uses one of the workers to produce
        a result (such result will eventually be available on the ``self._results_queue`` FIFO).

        This interface method will be called by SMBO, with the expectation that a function will be executed by a worker.
        What will be executed is dictated by ``trial_info``, and `how` it will be executed is decided via the child
        class that implements a ``run`` method.

        Because config submission can be a serial/parallel endeavor, it is expected to be implemented by a child class.

        Parameters
        ----------
        trial_info : TrialInfo
            An object containing the configuration launched.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        budget: float | None = None,
        seed: int | None = None,
    ) -> tuple[StatusType, float | list[float], float, dict]:
        """Runs the target function with a configuration on a single instance-budget-seed
        combination (aka trial).

        Parameters
        ----------
        config : Configuration
            Configuration to be passed to the target function.
        instance : str | None, defaults to None
            The Problem instance.
        budget : float | None, defaults to None
            A positive, real-valued number representing an arbitrary limit to the target function
            handled by the target function internally.
        seed : int, defaults to None

        Returns
        -------
        status : StatusType
            Status of the trial.
        cost : float | list[float]
            Resulting cost(s) of the trial.
        runtime : float
            The time the target function took to run.
        additional_info : dict
            All further additional trial information.
        """
        raise NotImplementedError

    @abstractmethod
    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:
        """This method returns any finished configuration, and returns a list with the
        results of executing the configurations. This class keeps populating results
        to ``self._results_queue`` until a call to ``get_finished`` trials is done. In this case,
        the `self._results_queue` list is emptied and all trial values produced by running
        `run` are returned.

        Returns
        -------
        Iterator[tuple[TrialInfo, TrialValue]]:
            A list of TrialInfo/TrialValue tuples, all of which have been finished.
        """
        raise NotImplementedError

    @abstractmethod
    def wait(self) -> None:
        """The SMBO/intensifier might need to wait for trials to finish before making a decision."""
        raise NotImplementedError

    @abstractmethod
    def is_running(self) -> bool:
        """Whether there are trials still running.

        Generally, if the runner is serial, launching a trial instantly returns its result. On
        parallel runners, there might be pending configurations to complete.
        """
        raise NotImplementedError

    @abstractmethod
    def count_available_workers(self) -> int:
        """Returns the number of available workers."""
        raise NotImplementedError
