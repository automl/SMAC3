from __future__ import annotations

from typing import Any, Callable, Iterator

import copy
import inspect
import math
import time
import traceback

import numpy as np
from pynisher import MemoryLimitException, WallTimeoutException, limit

from smac.configspace import Configuration
from smac.runner.abstract_runner import StatusType
from smac.runner.abstract_runner import AbstractRunner
from smac.utils.logging import get_logger
from smac.runhistory.dataclasses import TrialInfo, TrialValue

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class TargetAlgorithmRunner(AbstractRunner):
    """Class to execute target algorithms which are python functions.

    Evaluate function for given configuration and resource limit.

    Passes the configuration as a dictionary to the target algorithm. The
    target algorithm needs to implement one of the following signatures:

    * ``target_algorithm(config: Configuration) -> Union[float, Tuple[float, Any]]``
    * ``target_algorithm(config: Configuration, seed: int) -> Union[float, Tuple[float, Any]]``
    * ``target_algorithm(config: Configuration, seed: int, instance: str) -> Union[float, Tuple[float, Any]]``

    The target algorithm can either return a float (the loss), or a tuple
    with the first element being a float and the second being additional run
    information.

    ExecuteTAFuncDict will use inspection to figure out the correct call to
    the target algorithm.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Pynisher limitations
        if (memory := self.scenario.trial_memory_limit) is not None:
            memory = int(math.ceil(memory))

        if (time := self.scenario.trial_walltime_limit) is not None:
            time = int(math.ceil(time))

        self.memory_limit = memory
        self.algorithm_walltime_limit = time

    def submit_run(self, run_info: TrialInfo) -> None:
        """This function submits a run_info object in a serial fashion.

        As there is a single worker for this task, this
        interface can be considered a wrapper over the run()
        method.

        Both result/exceptions can be completely determined in this
        step so both lists are properly filled.

        Parameters
        ----------
        run_info: RunInfo
            An object containing the configuration and the necessary data to run it
        """
        self._results_queue.append(self.run_wrapper(run_info))

    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:
        """This method returns any finished configuration, and returns a list with the
        results of exercising the configurations. This class keeps populating results to
        self._results_queue until a call to get_finished runs is done. In this case,
        the self._results_queue list is emptied and all RunValues produced by running
        self.run() are returned.

        Returns
        -------
        list[tuple[RunInfo, RunValue]]
            A list of RunInfo/RunValues pairs a submitted configuration.
        """
        while self._results_queue:
            yield self._results_queue.pop(0)  # TODO: Could switch to dequeue?

    def wait(self) -> None:
        """SMBO/intensifier might need to wait for runs to finish before making a decision.

        For serial runs, no wait is needed as the result is immediately available.
        """
        # There is no need to wait in serial runs. When launching a run via submit, as
        # the serial run uses the same process to run, the result is always available
        # immediately after. This method implements is just an implementation of the
        # abstract method via a simple return, again, because there is no need to wait
        return

    def is_running(self) -> bool:
        """Whether or not there are configs still running.

        Generally if the runner is serial, launching a run instantly returns it's result.
        On parallel runners, there might be pending configurations to complete.
        """
        return False

    def available_worker_count(self) -> int:
        """Total number of workers available. Serial workers only have 1"""
        return 1

    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        seed: int = 0,
        budget: float | None = None,
        # instance_specific: str = "0",
    ) -> tuple[StatusType, float | list[float], float, dict]:
        """Calls the target algorithm with pynisher (if algorithm walltime limit or memory limit is set) or without."""
        # The kwargs are passed to the target algorithm.
        kwargs: dict[str, Any] = {}
        if self._accepts_seed:
            kwargs["seed"] = seed
        if self._accepts_instance:
            kwargs["instance"] = instance
        if self._accepts_budget:
            kwargs["budget"] = budget

        # Presetting
        cost: float | list[float] = self.crash_cost
        runtime = 0.0
        additional_info = {}
        status = StatusType.CRASHED

        # If memory limit or walltime limit is set, we wanna use pynisher
        target_algorithm: Callable
        if self.memory_limit is not None or self.algorithm_walltime_limit is not None:
            target_algorithm = limit(
                self.target_algorithm,
                memory=self.memory_limit,
                wall_time=self.algorithm_walltime_limit,
                wrap_errors=True,  # Hard to describe; see https://github.com/automl/pynisher
            )
        else:
            target_algorithm = self.target_algorithm

        # We don't want the user to change the configuration
        config_copy = copy.deepcopy(config)

        # Call target algorithm
        try:
            start_time = time.time()
            rval = self(config_copy, target_algorithm, kwargs)
            runtime = time.time() - start_time
            status = StatusType.SUCCESS
        except WallTimeoutException:
            status = StatusType.TIMEOUT
        except MemoryLimitException:
            status = StatusType.MEMORYOUT
        except Exception as e:
            cost = np.asarray(cost).squeeze().tolist()
            additional_info = {
                "traceback": traceback.format_exc(),
                "error": repr(e),
            }
            status = StatusType.CRASHED

        if status != StatusType.SUCCESS:
            return status, cost, runtime, additional_info

        if isinstance(rval, tuple):
            result, additional_info = rval
        else:
            result, additional_info = rval, {}

        # Do some sanity checking (for multi objective)
        error = f"Returned costs {result} does not match the number of objectives {self.objectives}."

        # If dict convert to array make sure the ordering is correct.
        if isinstance(result, dict):
            if len(result) != len(self.objectives):
                raise RuntimeError(error)

            ordered_cost: list[float] = []
            for name in self.objectives:
                if name not in result:
                    raise RuntimeError(f"Objective {name} was not found in the returned costs.")

                ordered_cost.append(result[name])

            result = ordered_cost

        if isinstance(result, list):
            if len(result) != len(self.objectives):
                raise RuntimeError(error)

        if isinstance(result, float):
            if isinstance(self.objectives, list) and len(self.objectives) != 1:
                raise RuntimeError(error)

        cost = result

        if cost is None:
            status = StatusType.CRASHED
            cost = self.crash_cost

        # We want to get either a float or a list of floats.
        cost = np.asarray(cost).squeeze().tolist()

        return status, cost, runtime, additional_info

    def __call__(
        self,
        config: Configuration,
        algorithm: Callable,
        algorithm_kwargs: dict[str, int | str | float | None],
    ) -> (
        float
        | list[float]
        | dict[str, float]
        | tuple[float, dict]
        | tuple[list[float], dict]
        | tuple[dict[str, float], dict]
    ):
        """Calls the algorithm"""
        return algorithm(config, **algorithm_kwargs)
