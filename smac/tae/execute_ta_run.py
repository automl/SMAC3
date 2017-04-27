import sys
import logging
import math
from enum import Enum

import numpy as np

from smac.configspace import Configuration

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class StatusType(Enum):

    """
        class to define numbers for status types
    """
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4
    MEMOUT = 5
    CAPPED = 6

    def enum_hook(obj):
        """
        hook function passed to json-deserializer as "object_hook"
        """
        if "__enum__" in obj:
            # object is marked as enum
            name, member = obj["__enum__"].split(".")
            if name == "StatusType":
                return getattr(globals()[name], member)
        return obj

class BudgetExhaustedException(Exception):
    """ Exception indicating that time- or memory-budgets are exhausted. """
    pass

class TAEAbortException(Exception):
    """ Exception indicating that the target algorithm suggests an ABORT of
    SMAC, usually because it assumes that all further runs will surely fail.
    """
    pass

class FirstRunCrashedException(TAEAbortException):
    """ Exception indicating that the first run crashed (depending on options
    this could trigger an ABORT of SMAC. """
    pass

class CappedRunException(Exception):
    """ Exception indicating that a run was capped with a cutoff smaller than the actual timeout """
    pass


class ExecuteTARun(object):

    """
        executes a target algorithm run with a given configuration
        on a given instance and some resource limitations

        Attributes
        ----------
        ta : string
            the command line call to the target algorithm (wrapper)
    """

    def __init__(self, ta, stats=None, runhistory=None, run_obj="runtime",
                 par_factor=1):
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

        self.logger = logging.getLogger("smac.tae."+self.__class__.__name__)
        self._supports_memory_limit = False

    def start(self, config:Configuration, 
              instance:str,
              cutoff:float=None,
              seed:int=12345,
              instance_specific:str="0",
              capped:bool=False):
        """
            wrapper function for ExecuteTARun.run() to check configuration budget before the runs
            and to update stats after run

            Parameters
            ----------
                config : Configuration
                    mainly a dictionary param -> value
                instance : string
                    problem instance
                cutoff : float
                    runtime cutoff
                seed : int
                    random seed
                instance_specific: str
                    instance specific information (e.g., domain file or solution)
                capped: bool
                    if true and status is StatusType.TIMEOUT, 
                    uses StatusType.CAPPED 

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
            raise BudgetExhaustedException("Skip target algorithm run due to exhausted configuration budget")

        status, cost, runtime, additional_info = self.run(config=config,
                                                          instance=instance,
                                                          cutoff=cutoff,
                                                          seed=seed,
                                                          instance_specific=instance_specific)

        if self.stats.ta_runs == 0 and status == StatusType.CRASHED:
            raise FirstRunCrashedException("First run crashed, abort. (To "
                                           "prevent this, toggle the "
                                           "'abort_on_first_run_crash'"
                                           "-option!)")
        if status == StatusType.ABORT:
            raise TAEAbortException("Target algorithm status ABORT - SMAC will "
                                    "exit. The last incumbent can be found "
                                    "in the trajectory-file.")

        # update SMAC stats
        self.stats.ta_runs += 1
        self.stats.ta_time_used += float(runtime)

        if self.run_obj == "runtime":
            if status != StatusType.SUCCESS:
                cost = cutoff * self.par_factor
            else:
                cost = runtime
            if status == StatusType.TIMEOUT and capped:
                status = StatusType.CAPPED

        self.logger.debug("Return: Status: %r, cost: %f, time: %f, additional: %s" % (
            status, cost, runtime, str(additional_info)))

        if self.runhistory:
            self.runhistory.add(config=config,
                                cost=cost, time=runtime, status=status,
                                instance_id=instance, seed=seed,
                                additional_info=additional_info)
        
        if status == StatusType.CAPPED:
            raise CappedRunException("")

        return status, cost, runtime, additional_info

    def run(self, config, instance,
            cutoff=None,
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
