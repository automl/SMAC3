import logging
import math

import numpy as np

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class StatusType(object):

    """
        class to define numbers for status types
    """
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4
    MEMOUT = 5


class ExecuteTARun(object):

    """
        executes a target algorithm run with a given configuration
        on a given instance and some resource limitations

        Attributes
        ----------
        ta : string
            the command line call to the target algorithm (wrapper)
    """

    def __init__(self, ta, stats=None, runhistory=None, run_obj="runtime", par_factor=1):
        """
        Constructor

        Parameters
        ----------
            ta : list
                target algorithm command line as list of arguments
            runhistory: RunHistory
                runhistory to keep track of all runs; only used if set
            stats: Stats()
                 stats object to collect statistics about runtime and so on                
            run_obj: str
                run objective of SMAC
            par_factor: int
                penalization factor
        """
        
        self.ta = ta
        self.stats = stats
        self.runhistory = runhistory
        self.run_obj = run_obj
        
        self.par_factor = par_factor

        self.logger = logging.getLogger(self.__class__.__name__)
        self._supports_memory_limit = False

    def start(self, config, instance,
              cutoff=None,
              memory_limit=None,
              seed=12345,
              instance_specific="0"):
        """
            wrapper function for ExecuteTARun.run() to check configuration budget before the runs
            and to update stats after run

            Parameters
            ----------
                config : dictionary
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
                    cost/regret/quality (float) (None, if not returned by TA)
                runtime: float
                    runtime (None if not returned by TA)
                additional_info: dict
                    all further additional run information
        """

        if self.stats.is_budget_exhausted():
            self.logger.debug(
                "Skip target algorithm run due to exhausted configuration budget")
            return StatusType.ABORT, np.nan, 0, {"misc": "exhausted bugdet -- ABORT"}

        if cutoff is not None:
            cutoff = int(math.ceil(cutoff))
        if memory_limit is not None:
            memory_limit = int(math.ceil(memory_limit))

        additional_arguments = {}

        if self._supports_memory_limit is True:
            additional_arguments['memory_limit'] = memory_limit
        elif self._supports_memory_limit is False and memory_limit is not None:
            raise ValueError('Target algorithm executor %s does not support '
                             'restricting the memory usage.' %
                             self.__class__.__name__)

        status, cost, runtime, additional_info = self.run(config=config,
                                                          instance=instance,
                                                          cutoff=cutoff,
                                                          seed=seed,
                                                          instance_specific=instance_specific,
                                                          **additional_arguments)
        # update SMAC stats
        self.stats.ta_runs += 1
        self.stats.ta_time_used += float(runtime)

        if self.run_obj == "runtime":
            if status != StatusType.SUCCESS:
                cost = cutoff * self.par_factor
            else:
                cost = runtime

        self.logger.debug("Return: Status: %d, cost: %f, time. %f, additional: %s" % (
            status, cost, runtime, str(additional_info)))

        if self.runhistory:
            self.runhistory.add(config=config,
                                cost=cost, time=runtime, status=status,
                                instance_id=instance, seed=seed,
                                additional_info=additional_info)
            
        return status, cost, runtime, additional_info

    def run(self, config, instance,
            cutoff=None,
            memory_limit=None,
            seed=12345,
            instance_specific="0"):
        """
            runs target algorithm <self.ta> with configuration <config> on
            instance <instance> with instance specifics <specifics>
            for at most <cutoff> seconds and random seed <seed>

            Parameters
            ----------
                config : dictionary
                    dictionary param -> value
                instance : string
                    problem instance
                cutoff : int, optional
                    Wallclock time limit of the target algorithm. If no value is
                    provided no limit will be enforced.
                memory_limit : int, optional
                    Memory limit in MB enforced on the target algorithm If no
                    value is provided no limit will be enforced.
                seed : int
                    random seed
                instance_specific: str
                    instance specific information (e.g., domain file or solution)

            Returns
            -------
                status: enum of StatusType (int)
                    {SUCCESS, TIMEOUT, CRASHED, ABORT}
                cost: float
                    cost/regret/quality (float) (None, if not returned by TA)
                runtime: float
                    runtime (None if not returned by TA)
                additional_info: dict
                    all further additional run information
        """
        return StatusType.SUCCESS, 12345.0, 1.2345, {}
