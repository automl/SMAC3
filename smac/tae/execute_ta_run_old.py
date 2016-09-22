import sys
from subprocess import Popen, PIPE

from smac.tae.execute_ta_run import StatusType, ExecuteTARun

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class ExecuteTARunOld(ExecuteTARun):

    """
        executes a target algorithm run with a given configuration
        on a given instance and some resource limitations
        Uses the original SMAC/PILS format (SMAC < v2.10)

        Attributes
        ----------
        ta : string
            the command line call to the target algorithm (wrapper)
        run_obj: str
            run objective (runtime or quality)
        par_factor: int
            penalized average runtime factor
    """

    def run(self, config, instance=None,
            cutoff=None,
            seed=12345,
            instance_specific="0"
            ):
        """
            runs target algorithm <self.ta> with configuration <config> on
            instance <instance> with instance specifics <specifics>
            for at most <cutoff> seconds and random seed <seed>

            Parameters
            ----------
                config : dictionary (or similar)
                    dictionary param -> value
                instance : string
                    problem instance
                cutoff : double
                    runtime cutoff
                seed : int
                    random seed
                instance_specific: str
                    instance specific information (e.g., domain file or solution)
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

        if instance is None:
            instance = "0"
        if cutoff is None:
            cutoff = 99999999999999.

        # TOOD: maybe replace fixed instance specific and cutoff_length (0) to
        # other value
        cmd = []
        cmd.extend(self.ta)
        cmd.extend([instance, instance_specific, str(cutoff), "0", str(seed)])
        for p in config:
            if not config[p] is None:
                cmd.extend(["-" + str(p), str(config[p])])

        self.logger.debug("Calling: %s" % (" ".join(cmd)))
        p = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        stdout_, stderr_ = p.communicate()

        self.logger.debug("Stdout: %s" % (stdout_))
        self.logger.debug("Stderr: %s" % (stderr_))

        status = "CRASHED"
        quality = 1234567890
        runtime = 1234567890
        additional_info = {}
        for line in stdout_.split("\n"):
            if line.startswith("Result of this algorithm run:") or line.startswith("Result for ParamILS") or line.startswith("Result for SMAC"):
                fields = line.split(":")[1].split(",")
                fields = list(map(lambda x: x.strip(" "), fields))
                if len(fields) == 5:
                    status, runtime, runlength, quality, seed = fields
                    additional_info = {}
                else:
                    status, runtime, runlength, quality, seed, additional_info = fields
                    additional_info = {"additional_info": additional_info}

                runtime = min(float(runtime), cutoff)
                quality = float(quality)
                seed = int(seed)

        if status in ["SAT", "UNSAT", "SUCCESS"]:
            status = StatusType.SUCCESS
        elif status in ["TIMEOUT"]:
            status = StatusType.TIMEOUT
        elif status in ["CRASHED"]:
            status = StatusType.CRASHED
        elif status in ["ABORT"]:
            status = StatusType.ABORT
            self.logger.error("Target algorithm returned ABORT -- Exit!")
            sys.exit(43)
        elif status in ["MEMOUT"]:
            status = StatusType.MEMOUT

        if status in [StatusType.CRASHED, StatusType.ABORT]:
            self.logger.warn(
                "Target algorithm crashed. Last 5 lines of stdout and stderr")
            self.logger.warn("\n".join(stdout_.split("\n")[-5:]))
            self.logger.warn("\n".join(stderr_.split("\n")[-5:]))

        if self.run_obj == "runtime":
            cost = runtime
        else:
            cost = quality

        return status, cost, float(runtime), additional_info
