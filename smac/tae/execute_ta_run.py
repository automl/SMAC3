import logging
import math
from enum import Enum
import typing

import numpy as np

from smac.configspace import Configuration
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT

if typing.TYPE_CHECKING:
    from smac.runhistory.runhistory import RunHistory  # noqa F401

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class StatusType(Enum):

    """Class to define numbers for status types"""
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4
    MEMOUT = 5
    CAPPED = 6
    # Only relevant for SH/HB. Run might have a results, but should not be considered further.
    # By default, these runs will always be considered for building the model. Potential use cases:
    # 1) The run has converged and does not benefit from a higher budget
    # 2) The run has exhausted given resources and will not benefit from higher budgets
    DONOTADVANCE = 7

    @staticmethod
    def enum_hook(obj: typing.Dict) -> typing.Any:
        """Hook function passed to json-deserializer as "object_hook".
        EnumEncoder in runhistory/runhistory.
        """
        if "__enum__" in obj:
            # object is marked as enum
            name, member = obj["__enum__"].split(".")
            if name == "StatusType":
                return getattr(globals()[name], member)
        return obj


class BudgetExhaustedException(Exception):
    """Exception indicating that time- or memory-budgets are exhausted."""
    pass


class TAEAbortException(Exception):
    """Exception indicating that the target algorithm suggests an ABORT of
    SMAC, usually because it assumes that all further runs will surely fail.
    """
    pass


class FirstRunCrashedException(TAEAbortException):
    """Exception indicating that the first run crashed (depending on options
    this could trigger an ABORT of SMAC.) """
    pass


class CappedRunException(Exception):
    """Exception indicating that a run was capped with a cutoff smaller than
    the actual timeout """
    pass


class ExecuteTARun(object):

    """Executes a target algorithm run with a given configuration on a given
    instance and some resource limitations

    Attributes
    ----------
    ta
    stats
    runhistory
    run_obj

    par_factor
    crash_cost

    logger
    """

    def __init__(
        self,
        ta: typing.Union[typing.List[str], typing.Callable],
        stats: Stats,
        runhistory: typing.Optional['RunHistory'] = None,
        run_obj: str = "runtime",
        par_factor: int = 1,
        cost_for_crash: float = float(MAXINT),
        abort_on_first_run_crash: bool = True,
    ) -> None:
        """Constructor

        Parameters
        ----------
        ta : list
            target algorithm command line as list of arguments
        runhistory: RunHistory, optional
            runhistory to keep track of all runs; only used if set
        stats: Stats()
             stats object to collect statistics about runtime and so on
        run_obj: str
            run objective of SMAC
        par_factor: int
            penalization factor
        crash_cost : float
            cost that is used in case of crashed runs (including runs
            that returned NaN or inf)
        abort_on_first_run_crash: bool
            if true and first run crashes, raise FirstRunCrashedException
        """
        self.ta = ta
        self.stats = stats
        self.runhistory = runhistory
        self.run_obj = run_obj

        self.par_factor = par_factor
        self.crash_cost = cost_for_crash
        self.abort_on_first_run_crash = abort_on_first_run_crash

        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)
        self._supports_memory_limit = False

    def start(
        self,
        config: Configuration,
        instance: str,
        cutoff: typing.Optional[float] = None,
        seed: int = 12345,
        budget: float = 0.0,
        instance_specific: str = "0",
        capped: bool = False,
    ) -> typing.Tuple[StatusType, float, float, typing.Dict]:
        """Wrapper function for ExecuteTARun.run() to check configuration
        budget before the runs and to update stats after run

        Parameters
        ----------
            config : Configuration
                Mainly a dictionary param -> value
            instance : string
                Problem instance
            cutoff : float, optional
                Runtime cutoff
            seed : int
                Random seed
            budget : float, optional
                A positive, real-valued number representing an arbitrary limit to the target algorithm
                Handled by the target algorithm internally
            instance_specific: str
                Instance specific information (e.g., domain file or solution)
            capped: bool
                If true and status is StatusType.TIMEOUT,
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
            raise BudgetExhaustedException(
                "Skip target algorithm run due to exhausted "
                "configuration budget")

        if cutoff is not None:
            cutoff = int(math.ceil(cutoff))
        if cutoff is None and self.run_obj == "runtime":
            self.logger.critical("For scenarios optimizing running time "
                                 "(run objective), a cutoff time is required, "
                                 "but not given to this call.")
            raise ValueError("For scenarios optimizing running time "
                             "(run objective), a cutoff time is required, "
                             "but not given to this call.")

        status, cost, runtime, additional_info = self.run(config=config,
                                                          instance=instance,
                                                          cutoff=cutoff,
                                                          seed=seed,
                                                          budget=budget,
                                                          instance_specific=instance_specific)
        if budget == 0 and status == StatusType.DONOTADVANCE:
            raise ValueError("Cannot handle DONOTADVANCE state when using intensify or SH/HB on "
                             "instances.")
        # update SMAC stats
        self.stats.ta_runs += 1
        self.stats.ta_time_used += float(runtime)

        # Catch NaN or inf.
        if (
            self.run_obj == 'runtime' and not np.isfinite(runtime)
            or self.run_obj == 'quality' and not np.isfinite(cost)
        ):
            self.logger.warning("Target Algorithm returned NaN or inf as {}. "
                                "Algorithm run is treated as CRASHED, cost "
                                "is set to {} for quality scenarios. "
                                "(Change value through \"cost_for_crash\""
                                "-option.)".format(self.run_obj,
                                                   self.crash_cost))
            status = StatusType.CRASHED

        if status == StatusType.ABORT:
            raise TAEAbortException("Target algorithm status ABORT - SMAC will "
                                    "exit. The last incumbent can be found "
                                    "in the trajectory-file.")

        if self.run_obj == "runtime":
            # The following line pleases mypy - we already check for cutoff not being none above, prior to calling
            # run. However, mypy assumes that the data type of cutoff is still Optional[int]
            assert cutoff is not None
            if runtime > self.par_factor * cutoff:
                self.logger.warning("Returned running time is larger "
                                    "than {0} times the passed cutoff time. "
                                    "Clamping to {0} x cutoff.".format(self.par_factor))
                runtime = cutoff * self.par_factor
                status = StatusType.TIMEOUT
            if status == StatusType.SUCCESS:
                cost = runtime
            else:
                cost = cutoff * self.par_factor
            if status == StatusType.TIMEOUT and capped:
                status = StatusType.CAPPED
        else:
            if status == StatusType.CRASHED:
                cost = self.crash_cost

        self.logger.debug("Return: Status: %r, cost: %f, time: %f, additional: %s" % (
            status, cost, runtime, str(additional_info)))

        if self.runhistory:
            self.runhistory.add(config=config,
                                cost=cost, time=runtime, status=status,
                                instance_id=instance, seed=seed,
                                budget=budget,
                                additional_info=additional_info)
            self.stats.n_configs = len(self.runhistory.config_ids)

        if status == StatusType.CAPPED:
            raise CappedRunException("")

        if self.abort_on_first_run_crash and self.stats.ta_runs == 1 and status == StatusType.CRASHED:
            raise FirstRunCrashedException("First run crashed, abort. "
                                           "Please check your setup -- "
                                           "we assume that your default"
                                           "configuration does not crashes. "
                                           "(To deactivate this exception,"
                                           " use the SMAC scenario option "
                                           "'abort_on_first_run_crash')")

        return status, cost, runtime, additional_info

    def run(
        self, config: Configuration,
        instance: str,
        cutoff: typing.Optional[float] = None,
        seed: int = 12345,
        budget: typing.Optional[float] = None,
        instance_specific: str = "0",
    ) -> typing.Tuple[StatusType, float, float, typing.Dict]:
        """Runs target algorithm <self.ta> with configuration <config> on
        instance <instance> with instance specifics <specifics> for at most
        <cutoff> seconds and random seed <seed>

        Parameters
        ----------
            config : Configuration
                dictionary param -> value
            instance : string
                problem instance
            cutoff : float, optional
                Wallclock time limit of the target algorithm. If no value is
                provided no limit will be enforced.
            seed : int
                random seed
            budget : float, optional
                A positive, real-valued number representing an arbitrary limit to the target algorithm
                Handled by the target algorithm internally
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
