from __future__ import annotations

from typing import Any

import time
from subprocess import PIPE, Popen

from ConfigSpace import Configuration

from smac.runner.abstract_runner import StatusType
from smac.runner.abstract_serial_runner import AbstractSerialRunner
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class TargetFunctionScriptRunner(AbstractSerialRunner):
    """Class to execute target functions from scripts. Uses `Popen` to execute the script in a
     subprocess.

    The following example shows how the script is called:
    ``target_function --instance=test --instance_features=test --seed=0 --hyperparameter1=5323``

    The script must return an echo in the following form (white-spaces are removed):
    ``cost=0.5; runtime=0.01; status=SUCCESS; additional_info=test`` (single-objective)
    ``cost=0.5, 0.4; runtime=0.01; status=SUCCESS; additional_info=test`` (multi-objective)

    The status must be a string and must be one of the ``StatusType`` values. However, ``runtime``,
    ``status`` and ``additional_info`` are optional.

    Note
    ----
    Everytime an instance is passed, also an instance feature in form of a comma-separated list
    (no spaces) of floats is passed. If no instance feature for the instance is given,
    an empty list is passed.

    Parameters
    ----------
    target_function : Callable
        The target function.
    scenario : Scenario
    required_arguments : list[str]
        A list of required arguments, which are passed to the target function.
    """

    def __init__(
        self,
        target_function: str,
        scenario: Scenario,
        required_arguments: list[str] = None,
    ):
        if required_arguments is None:
            required_arguments = []
        super().__init__(scenario=scenario, required_arguments=required_arguments)
        self._target_function = target_function

        # Check if target function is callable
        if not isinstance(self._target_function, str):
            raise TypeError(
                "Argument `target_function` must be a string but is type" f"`{type(self._target_function)}`."
            )

        if self._scenario.trial_memory_limit is not None:
            logger.warning("Trial memory limit is not supported for script target functions.")

        if self._scenario.trial_walltime_limit is not None:
            logger.warning("Trial walltime limit is not supported for script target functions.")

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"filename": str(self._target_function)})

        return meta

    def run(
        self,
        config: Configuration,
        instance: str | None = None,
        budget: float | None = None,
        seed: int | None = None,
    ) -> tuple[StatusType, float | list[float], float, dict]:
        """Calls the target function.

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
        # The kwargs are passed to the target function.
        kwargs: dict[str, Any] = {}
        if "seed" in self._required_arguments:
            kwargs["seed"] = seed

        if "instance" in self._required_arguments:
            kwargs["instance"] = instance

            # In contrast to the normal target function runner, we also add the instance features here.
            if self._scenario.instance_features is not None and instance in self._scenario.instance_features:
                kwargs["instance_features"] = self._scenario.instance_features[instance]
            else:
                kwargs["instance_features"] = []

        if "budget" in self._required_arguments:
            kwargs["budget"] = budget

        # Presetting
        cost: float | list[float] = self._crash_cost
        runtime = 0.0
        additional_info = {}
        status = StatusType.SUCCESS

        # Add config arguments to the kwargs
        for k, v in config.get_dictionary().items():
            if k in kwargs:
                raise RuntimeError(f"The key {k} is already in use. Please use a different one.")
            kwargs[k] = v

        # Call target function
        start_time = time.time()
        output, error = self(kwargs)
        runtime = time.time() - start_time

        # Now we have to parse the std output
        # First remove white-spaces
        output = output.replace(" ", "")

        outputs = {}
        for pair in output.split(";"):
            try:
                kv = pair.split("=")
                k, v = kv[0], kv[1]

                # Get rid of the trailing newline
                v = v.strip()

                outputs[k] = v
            except Exception:
                pass

        # Parse status
        if "status" in outputs:
            status = getattr(StatusType, outputs["status"])

        # Parse costs (depends on the number of objectives)
        if "cost" in outputs:
            if self._n_objectives == 1:
                cost = float(outputs["cost"])
            else:
                costs = outputs["cost"].split(",")
                costs = [float(c) for c in costs]
                cost = costs

                if len(costs) != self._n_objectives:
                    raise RuntimeError("The number of costs does not match the number of objectives.")
        else:
            status = StatusType.CRASHED

        # Overwrite runtime
        if "runtime" in outputs:
            runtime = float(outputs["runtime"])

        # Add additional info
        if "additional_info" in outputs:
            additional_info["additional_info"] = outputs["additional_info"]

        if status != StatusType.SUCCESS:
            additional_info["error"] = error

            if cost != self._crash_cost:
                cost = self._crash_cost
                logger.info(
                    "The target function crashed but returned a cost. The cost is ignored and replaced by crash cost."
                )

        return status, cost, runtime, additional_info

    def __call__(
        self,
        algorithm_kwargs: dict[str, Any],
    ) -> tuple[str, str]:
        """Calls the algorithm, which is processed in the ``run`` method."""
        cmd = [self._target_function]
        for k, v in algorithm_kwargs.items():
            v = str(v)
            k = str(k)

            # Let's remove some spaces
            v = v.replace(" ", "")

            cmd += [f"--{k}={v}"]

        logger.debug(f"Calling: {' '.join(cmd)}")
        p = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        output, error = p.communicate()

        logger.debug("Stdout: %s" % output)
        logger.debug("Stderr: %s" % error)

        return output, error
