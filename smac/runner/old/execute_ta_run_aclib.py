from typing import Dict, List, Optional, Tuple

import json
from subprocess import PIPE, Popen

from smac.configspace import Configuration
from smac.runner import StatusType
from smac.runner.serial_runner import SerialRunner

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class ExecuteTARunAClib(SerialRunner):
    """Executes a target algorithm run with a given configuration on a given instance and some
    resource limitations.

    Uses the AClib 2.0 style
    """

    def run(
        self,
        config: Configuration,
        instance: str,
        algorithm_walltime_limit: Optional[float] = None,
        seed: int = 12345,
        budget: Optional[float] = None,
        instance_specific: str = "0",
    ) -> Tuple[StatusType, float, float, Dict]:
        """Runs target algorithm <self.ta> with configuration <config> on instance <instance> with
        instance specifics.

        <specifics> for at most.

        <algorithm_walltime_limit> seconds and random seed <seed>

        Parameters
        ----------
            config : Configuration
                Dictionary param -> value
            instance : str
                Problem instance
            algorithm_walltime_limit : float
                Runtime algorithm_walltime_limit
            seed : int
                Random seed
            budget : float (optional)
                Not implemented
            instance_specific: str
                Instance specific information -- ignored here
        Returns
        -------
            status: enum of StatusType (int)
                {SUCCESS, TIMEOUT, CRASHED, ABORT}
            cost: float
                cost/regret/quality/runtime (float) (None, if not returned by TA)
            runtime: float
                runtime (None if not returned by TA)
            additional_info: dict
                all further additional run information
        """
        if budget is not None:
            raise NotImplementedError()

        if instance is None:
            instance = "0"
        if algorithm_walltime_limit is None:
            algorithm_walltime_limit = 99999999999999

        results, stdout_, stderr_ = self._call_ta(
            config=config,
            instance=instance,
            instance_specific=instance_specific,
            algorithm_walltime_limit=algorithm_walltime_limit,
            seed=seed,
        )

        if results["status"] in ["SAT", "UNSAT", "SUCCESS"]:
            status = StatusType.SUCCESS
        elif results["status"] in ["TIMEOUT"]:
            status = StatusType.TIMEOUT
        elif results["status"] in ["CRASHED"]:
            status = StatusType.CRASHED
        elif results["status"] in ["ABORT"]:
            status = StatusType.ABORT
        elif results["status"] in ["MEMOUT"]:
            status = StatusType.MEMOUT
        else:
            self.logger.warning(
                "Could not identify status; should be one of the following: "
                "SAT, UNSAT, SUCCESS, TIMEOUT, CRASHED, ABORT or MEMOUT; "
                "Treating as CRASHED run."
            )
            status = StatusType.CRASHED

        if status in [StatusType.CRASHED, StatusType.ABORT]:
            self.logger.warning("Target algorithm crashed. Last 5 lines of stdout and stderr")
            self.logger.warning("\n".join(stdout_.split("\n")[-5:]))
            self.logger.warning("\n".join(stderr_.split("\n")[-5:]))

        if results.get("runtime") is None:
            self.logger.warning("The target algorithm has not returned a" " runtime -- imputed by 0.")
            # (TODO) Check 0
            results["runtime"] = 0

        runtime = float(results["runtime"])

        if self.run_obj == "quality" and results.get("cost") is None:
            self.logger.error("The target algorithm has not returned a quality/cost value although we optimize cost.")
            # (TODO) Do not return 0
            results["cost"] = 0

        if self.run_obj == "runtime":
            cost = float(results["runtime"])
        else:
            cost = float(results["cost"])

        del results["status"]
        try:
            del results["runtime"]
        except KeyError:
            pass
        try:
            del results["cost"]
        except KeyError:
            pass

        return status, cost, runtime, results

    def _call_ta(
        self,
        config: Configuration,
        instance: str,
        instance_specific: str,
        algorithm_walltime_limit: float,
        seed: int,
    ) -> Tuple[Dict, str, str]:

        # TODO: maybe replace fixed instance specific and algorithm_walltime_limit_length (0) to
        # other value
        cmd = []  # type: List[str]
        if not isinstance(self.ta, (list, tuple)):
            raise TypeError("self.ta needs to be of type list or tuple, but is %s" % type(self.ta))
        cmd.extend(self.ta)
        cmd.extend(
            [
                "--instance",
                instance,
                "--algorithm_walltime_limit",
                str(algorithm_walltime_limit),
                "--seed",
                str(seed),
                "--config",
            ]
        )

        for p in config:
            if not config.get(p) is None:
                cmd.extend(["-" + str(p), str(config[p])])

        self.logger.debug("Calling: %s" % (" ".join(cmd)))
        p = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        stdout_, stderr_ = p.communicate()

        self.logger.debug("Stdout: %s" % (stdout_))
        self.logger.debug("Stderr: %s" % (stderr_))

        results = {"status": "CRASHED", "cost": 1234567890}
        for line in stdout_.split("\n"):
            if line.startswith("Result of this algorithm run:"):
                fields = ":".join(line.split(":")[1:])
                results = json.loads(fields)

        return results, stdout_, stderr_
