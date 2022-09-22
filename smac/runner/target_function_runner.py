from __future__ import annotations

from typing import Any, Callable, Iterator

import copy
import math
import time
import traceback

import numpy as np
from ConfigSpace import Configuration
from pynisher import MemoryLimitException, WallTimeoutException, limit

from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runner.abstract_runner import AbstractRunner, StatusType
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class TargetFunctionRunner(AbstractRunner):
    """Class to execute target functions which are (python) functions. Evaluates functions for given configuration and
    resource limit.

    The target function can either return a float (the loss), or a tuple
    with the first element being a float and the second being additional run
    information. In a multi-objective setting, the float value is replaced by a list of floats.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Pynisher limitations
        if (memory := self._scenario.trial_memory_limit) is not None:
            memory = int(math.ceil(memory))

        if (time := self._scenario.trial_walltime_limit) is not None:
            time = int(math.ceil(time))

        self._memory_limit = memory
        self._algorithm_walltime_limit = time

    def submit_trial(self, trial_info: TrialInfo) -> None:
        """This function submits a trial_info object in a serial fashion. As there is a single worker for this task,
        this interface can be considered a wrapper over the `run` method.

        Both result/exceptions can be completely determined in this step so both lists are properly filled.

        Parameters
        ----------
        trial_info : TrialInfo
            An object containing the configuration launched.
        """
        self._results_queue.append(self.run_wrapper(trial_info))

    def iter_results(self) -> Iterator[tuple[TrialInfo, TrialValue]]:
        while self._results_queue:
            yield self._results_queue.pop(0)

    def wait(self) -> None:
        """The SMBO/intensifier might need to wait for trials to finish before making a decision.
        For serial runners, no wait is needed as the result is immediately available."""
        # There is no need to wait in serial runners. When launching a trial via submit, as
        # the serial trial uses the same process to run, the result is always available
        # immediately after. This method implements is just an implementation of the
        # abstract method via a simple return, again, because there is no need to wait
        return

    def is_running(self) -> bool:
        return False

    def count_available_workers(self) -> int:
        """Returns the number of available workers. Serial workers only have one worker."""
        return 1

    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        budget: float | None = None,
        seed: int | None = None,
    ) -> tuple[StatusType, float | list[float], float, dict]:
        """Calls the target function with pynisher if algorithm walltime limit or memory limit is set. Otherwise
        the function is called directly.

        Parameters
        ----------
        config : Configuration
            Configuration to be passed to the target function.
        instance : str | None, defaults to None
            The Problem instance.
        budget : float | None, defaults to None
            A positive, real-valued number representing an arbitrary limit to the target function handled by the
            target function internally.
        seed : int, defaults to None

        Returns
        -------
        status : StatusType
            Status of the trial.
        cost : float | list[float]
            Resulting cost(s) of the trial.
        runtime : float
            The time the target function function took to run.
        additional_info : dict
            All further additional trial information.
        """
        # The kwargs are passed to the target function.
        kwargs: dict[str, Any] = {}
        if "seed" in self._required_arguments:
            kwargs["seed"] = seed
        if "instance" in self._required_arguments:
            kwargs["instance"] = instance
        if "budget" in self._required_arguments:
            kwargs["budget"] = budget

        # Presetting
        cost: float | list[float] = self._crash_cost
        runtime = 0.0
        additional_info = {}
        status = StatusType.CRASHED

        # If memory limit or walltime limit is set, we wanna use pynisher
        target_function: Callable
        if self._memory_limit is not None or self._algorithm_walltime_limit is not None:
            target_function = limit(
                self._target_function,
                memory=self._memory_limit,
                wall_time=self._algorithm_walltime_limit,
                wrap_errors=True,  # Hard to describe; see https://github.com/automl/pynisher
            )
        else:
            target_function = self._target_function

        # We don't want the user to change the configuration
        config_copy = copy.deepcopy(config)

        # Call target function
        try:
            start_time = time.time()
            rval = self(config_copy, target_function, kwargs)
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
        error = f"Returned costs {result} does not match the number of objectives {self._objectives}."

        # If dict convert to array and make sure the order is correct
        if isinstance(result, dict):
            if len(result) != len(self._objectives):
                raise RuntimeError(error)

            ordered_cost: list[float] = []
            for name in self._objectives:
                if name not in result:
                    raise RuntimeError(f"Objective {name} was not found in the returned costs.")

                ordered_cost.append(result[name])

            result = ordered_cost

        if isinstance(result, list):
            if len(result) != len(self._objectives):
                raise RuntimeError(error)

        if isinstance(result, float):
            if isinstance(self._objectives, list) and len(self._objectives) != 1:
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
        algorithm_kwargs: dict[str, Any],
    ) -> (
        float
        | list[float]
        | dict[str, float]
        | tuple[float, dict]
        | tuple[list[float], dict]
        | tuple[dict[str, float], dict]
    ):
        """Calls the algorithm, which is processed in the `run` method."""
        return algorithm(config, **algorithm_kwargs)
