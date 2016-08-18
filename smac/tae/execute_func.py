import sys
import logging

from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats

import math

import pynisher

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class ExecuteTAFunc(object):

    """
        executes a function  with given inputs (i.e., the configuration)
        and some resource limitations

        Attributes
        ----------
        func : Python function handle 
            function to be optimized
        run_obj: str
            run objective (runtime or quality)
        par_factor: int
            penalized average runtime factor
    """

    def __init__(self, func, stats, run_obj="quality", par_factor=1):
        """
        Constructor

        Parameters
        ----------
            func: function
                target algorithm function
            stats: Stats()
                 stats object to collect statistics about runtime and so on
            run_obj: str
                run objective of SMAC
            par_factor: int
                penalized average runtime factor
        """
        self.func = func
        self.stats = stats
        self.logger = logging.getLogger("ExecuteTAFunc")
        self.run_obj = run_obj
        self.par_factor = par_factor

    def run(self, config, instance=None,
            cutoff=99999999999999.,
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
                    runtime cutoff -- will be casted to int
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

        obj = pynisher.enforce_limits(
            cpu_time_in_s=int(math.ceil(cutoff)), logger=logging.getLogger("pynisher"))(self.func)

        if instance:
            result = obj(config, instance, seed)
        else:
            result = obj(config, seed)

        #self.logger.debug("Function value: %.4f" % (result))

        if obj.exit_status is pynisher.CpuTimeoutException:
            status = StatusType.TIMEOUT
            cost = 1234567890
        elif obj.exit_status == 0:
            status = StatusType.SUCCESS
            cost = result
        else:
            status = StatusType.CRASHED
            cost = 1234567890  # won't be used for the model

        runtime = float(obj.wall_clock_time)

        if self.run_obj == "runtime":
            if status != StatusType.SUCCESS:
                cost = cutoff * self.par_factor
            else:
                cost = runtime

        # update SMAC stats
        self.stats.ta_runs += 1
        self.stats.ta_time_used += float(runtime)

        self.logger.debug("Return: %s,%.4f,%.4f" % (status, cost, runtime))

        return status, cost, runtime, {}
