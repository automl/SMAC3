import logging
from enum import Enum
import typing

from smac.configspace import Configuration
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT

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
    # In case of budget exception
    BUDGETEXHAUSTED = 8
    # In case a job was submited, but it has not finished
    RUNNING = 9

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
    run_obj

    par_factor
    crash_cost

    logger
    """

    def __init__(
        self,
        ta: typing.Union[typing.List[str], typing.Callable],
        stats: Stats,
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
        stats: Stats()
             stats object to collect statistics about runtime and so on
        run_obj: str
            run objective of SMAC
        par_factor: int
            penalization factor
        cost_for_crash : float
            cost that is used in case of crashed runs (including runs
            that returned NaN or inf)
        abort_on_first_run_crash: bool
            if true and first run crashes, raise FirstRunCrashedException
        """
        self.ta = ta
        self.stats = stats
        self.run_obj = run_obj

        self.par_factor = par_factor
        self.cost_for_crash = cost_for_crash
        self.abort_on_first_run_crash = abort_on_first_run_crash

        self.logger = logging.getLogger(
            self.__module__ + '.' + self.__class__.__name__)
        self._supports_memory_limit = False

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
        raise NotImplementedError()
