from __future__ import annotations

from typing import Any, Callable

import copy
import inspect
import math
import time
import traceback

import numpy as np
from pynisher import MemoryLimitException, WallTimeoutException, limit

from smac.configspace import Configuration
from smac.runner.runner import StatusType
from smac.runner.serial_runner import SerialRunner
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class TargetAlgorithmRunner(SerialRunner):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        signature = inspect.signature(self.target_algorithm).parameters
        self._accepts_seed = "seed" in signature.keys()
        self._accepts_instance = "instance" in signature.keys()
        self._accepts_budget = "budget" in signature.keys()

        if not callable(self.target_algorithm):
            raise TypeError(
                "Argument `target_algorithm` must be a callable but is type"
                f"`{type(self.target_algorithm)}`."
            )

        # Pynisher limitations
        if (memory := self.scenario.trial_memory_limit) is not None:
            memory = int(math.ceil(memory))

        if (time := self.scenario.trial_walltime_limit) is not None:
            time = int(math.ceil(time))

        self.memory_limit = memory
        self.algorithm_walltime_limit = time

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
        cost = self.crash_cost
        runtime = 0.0
        additional_info = {}
        status = StatusType.CRASHED

        # If memory limit or walltime limit is set, we wanna use pynisher
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
            result = rval[0]
            additional_info = rval[1]
        else:
            result = rval
            additional_info = {}

        # We update cost based on our result
        cost = result

        # Do some sanity checking (for multi objective)
        error = f"Returned costs {cost} does not match the number of objectives {self.objectives}."

        # If dict convert to array make sure the ordering is correct.
        if isinstance(cost, dict):
            if len(cost) != len(self.objectives):
                raise RuntimeError(error)

            ordered_cost = []
            for name in self.objectives:
                if name not in cost:
                    raise RuntimeError(f"Objective {name} was not found in the returned costs.")

                ordered_cost.append(cost[name])
            cost = ordered_cost

        if isinstance(cost, list):
            if len(cost) != len(self.objectives):
                raise RuntimeError(error)

        if isinstance(cost, float):
            if isinstance(self.objectives, list) and len(self.objectives) != 1:
                raise RuntimeError(error)

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
        float |
        list[float] |
        dict[str, float] |
        tuple[float, dict] |
        tuple[list[float] | dict] |
        tuple[dict[str, float] | dict]
    ):
        return algorithm(config, **algorithm_kwargs)
