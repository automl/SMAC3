from __future__ import annotations

from typing import Any, Callable

import copy
import inspect
import math
import time
import traceback
from functools import partial

import numpy as np
from ConfigSpace import Configuration
from pynisher import MemoryLimitException, WallTimeoutException, limit

from smac.runner.abstract_runner import StatusType
from smac.runner.abstract_serial_runner import AbstractSerialRunner
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class TargetFunctionRunner(AbstractSerialRunner):
    """Class to execute target functions which are python functions. Evaluates function for given
    configuration and resource limit.

    The target function can either return a float (the loss), or a tuple with the first element
    being a float and the second being additional run information. In a multi-objective
    setting, the float value is replaced by a list of floats.

    Parameters
    ----------
    target_function : Callable
        The target function.
    scenario : Scenario
    required_arguments : list[str], defaults to []
        A list of required arguments, which are passed to the target function.
    """

    def __init__(
        self,
        scenario: Scenario,
        target_function: Callable,
        required_arguments: list[str] = None,
    ):
        if required_arguments is None:
            required_arguments = []
        super().__init__(scenario=scenario, required_arguments=required_arguments)
        self._target_function = target_function

        # Check if target function is callable
        if not callable(self._target_function):
            raise TypeError(
                "Argument `target_function` must be a callable but is type" f"`{type(self._target_function)}`."
            )

        # Signatures here
        signature = inspect.signature(self._target_function).parameters
        for argument in required_arguments:
            if argument not in signature.keys():
                raise RuntimeError(
                    f"Target function needs to have the arguments {required_arguments} "
                    f"but could not find {argument}."
                )

        # Now we check for additional arguments which are not used by SMAC
        # However, we only want to warn the user and not
        for key in list(signature.keys())[1:]:
            if key not in required_arguments:
                logger.warning(f"The argument {key} is not set by SMAC: Consider removing it from the target function.")

        # Pynisher limitations
        if (memory := self._scenario.trial_memory_limit) is not None:
            unit = None
            if isinstance(memory, (tuple, list)):
                memory, unit = memory
            memory = int(math.ceil(memory))
            if unit is not None:
                memory = (memory, unit)

        if (time := self._scenario.trial_walltime_limit) is not None:
            time = int(math.ceil(time))

        self._memory_limit = memory
        self._algorithm_walltime_limit = time

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta

        # Partial's don't have a __code__ attribute but are a convenient
        # way a user might want to pass a function to SMAC, specifying
        # keyword arguments.
        f = self._target_function
        if isinstance(f, partial):
            f = f.func
            meta.update({"code": str(f.__code__.co_code)})
            meta.update({"code-partial-args": repr(f)})
        else:
            meta.update({"code": str(self._target_function.__code__.co_code)})

        return meta

    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        budget: float | None = None,
        seed: int | None = None,
        **dask_data_to_scatter: dict[str, Any],
    ) -> tuple[StatusType, float | list[float], float, dict]:
        """Calls the target function with pynisher if algorithm wall time limit or memory limit is
        set. Otherwise, the function is called directly.

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
        dask_data_to_scatter: dict[str, Any]
            This kwargs must be empty when we do not use dask! ()
            When a user scatters data from their local process to the distributed network,
            this data is distributed in a round-robin fashion grouping by number of cores.
            Roughly speaking, we can keep this data in memory and then we do not have to (de-)serialize the data
            every time we would like to execute a target function with a big dataset.
            For example, when your target function has a big dataset shared across all the target function,
            this argument is very useful.

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
        # The kwargs are passed to the target function.
        kwargs: dict[str, Any] = {}
        kwargs.update(dask_data_to_scatter)

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
        """Calls the algorithm, which is processed in the ``run`` method."""
        return algorithm(config, **algorithm_kwargs)
