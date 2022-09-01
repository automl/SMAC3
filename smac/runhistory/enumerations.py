from __future__ import annotations

from enum import Enum, IntEnum

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class StatusType(IntEnum):
    """Class to define numbers for status types."""

    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4
    MEMORYOUT = 5

    # Only relevant for SH/HB. Run might have a results, but should not be considered further.
    # By default, these runs will always be considered for building the model. Potential use cases:
    # 1) The run has converged and does not benefit from a higher budget
    # 2) The run has exhausted given resources and will not benefit from higher budgets
    DONOTADVANCE = 6

    STOP = 7  # Gracefully exit SMAC and wait for currently executed runs to finish.
    RUNNING = 8  # In case a job was submited, but it has not finished.


class DataOrigin(Enum):
    """Definition of how data in the runhistory is used.

    * ``INTERNAL``: internal data which was gathered during the current
      optimization run. It will be saved to disk, used for building EPMs and
      during intensify.
    * ``EXTERNAL_SAME_INSTANCES``: external data, which was gathered by running
       another program on the same instances as the current optimization run
       runs on (for example pSMAC). It will not be saved to disk, but used both
       for EPM building and during intensify.
    * ``EXTERNAL_DIFFERENT_INSTANCES``: external data, which was gathered on a
       different instance set as the one currently used, but due to having the
       same instance features can still provide useful information. Will not be
       saved to disk and only used for EPM building.
    """

    INTERNAL = 1
    EXTERNAL_SAME_INSTANCES = 2
    EXTERNAL_DIFFERENT_INSTANCES = 3


class TrialInfoIntent(Enum):
    """Class to define different requests on how to process the runinfo.

    Gives the flexibility to indicate whether a configuration should be skipped (SKIP) or if the
    SMBO should simple run a generated run_info.
    """

    RUN = 0  # Normal run execution of a run info
    SKIP = 1  # Skip running the run_info
    WAIT = 2  # Wait for more configs to be processed
