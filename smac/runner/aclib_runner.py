from __future__ import annotations

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

from typing import Any

import re
from subprocess import PIPE, Popen

from smac.runner.target_function_script_runner import TargetFunctionScriptRunner
from smac.scenario import Scenario
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class ACLibRunner(TargetFunctionScriptRunner):
    def __init__(
        self,
        target_function: str,
        scenario: Scenario,
        required_arguments: list[str] | None = None,
        target_function_arguments: dict[str, str] | None = None,
    ):

        self._target_function_arguments = target_function_arguments

        super().__init__(target_function, scenario, required_arguments)

    def __call__(self, algorithm_kwargs: dict[str, Any]) -> tuple[str, str]:
        """Calls the target function with the given arguments.

        Parameters
        ----------
        algorithm_kwargs: dict[str, Any]
            The arguments to pass to the target function.

        Returns
        -------
        tuple[str, str]
            The output and error messages from the target function.
        """
        # kwargs has "instance", "seed" and "budget" --> translate those

        cmd = self._target_function.split(" ")
        if self._target_function_arguments is not None:
            for k, v in self._target_function_arguments.items():
                cmd += [f"--{k}={v}"]

        if self._scenario.trial_walltime_limit is not None:
            cmd += [f"--cutoff={self._scenario.trial_walltime_limit}"]

        config = ["--config"]

        for k, v in algorithm_kwargs.items():
            v = str(v)
            k = str(k)

            # Let's remove some spaces
            v = v.replace(" ", "")

            if k in ["instance", "seed"]:
                cmd += [f"--{k}={v}"]
            elif k == "instance_features":
                continue
            else:
                config += [k, v]

        cmd += config

        logger.debug(f"Calling: {' '.join(cmd)}")
        p = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        output, error = p.communicate()

        logger.debug("Stdout: %s" % output)
        logger.debug("Stderr: %s" % error)

        result_begin = "Result for SMAC3v2: "
        outputline = ""
        for line in output.split("\n"):
            line = line.strip()
            if re.match(result_begin, line):
                # print("match")
                outputline = line[len(result_begin) :]

        logger.debug(f"Found result in output: {outputline}")

        # Parse output to form of key=value;key2=value2;...;cost=value1,value2;...

        return outputline, error
