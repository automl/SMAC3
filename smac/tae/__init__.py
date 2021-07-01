from enum import Enum
import typing

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
    # Gracefully exit SMAC - wait for currently executed runs to finish
    STOP = 8
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


class TAEAbortException(Exception):
    """Exception indicating that the target algorithm suggests an ABORT of
    SMAC, usually because it assumes that all further runs will surely fail.
    """
    pass


class FirstRunCrashedException(TAEAbortException):
    """Exception indicating that the first run crashed (depending on options
    this could trigger an ABORT of SMAC.) """
    pass
