import sys
import logging
from subprocess import Popen, PIPE

from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats

import pynisher

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class ExecuteTAFunc(object):

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

    def __init__(self, func, run_obj="quality", par_factor=1):
        """
        Constructor

        Parameters
        ----------
            func : function
                target algorithm function 
            run_obj: str
                run objective of SMAC
            par_factor: int
                penalized average runtime factor
        """
        self.func = func
        self.logger = logging.getLogger("ExecuteTARun")
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

        obj = pynisher.enforce_limits(cpu_time_in_s=cutoff)(self.func)

        result = obj(config)
        
        if obj.exit_status is pynisher.CpuTimeoutException:
            status = StatusType.TIMEOUT
            cost = 1234567890
        elif obj.exit_status == 0:
            status = StatusType.SUCCESS
            cost = result
        else:
            status = StatusType.CRASHED
            cost = 1234567890

        runtime = 0 #TODO: replace by real runtime
        
        if self.run_obj == "runtime":
            cost = runtime

        # update SMAC stats
        Stats.ta_runs += 1
        Stats.ta_time_used += float(runtime)

        return status, cost, 0, 0
