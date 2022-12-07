from __future__ import annotations

from enum import IntEnum

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class StatusType(IntEnum):
    """Class to define numbers for status types."""

    RUNNING = 0  # In case a job was submited, but it has not finished.
    SUCCESS = 1
    CRASHED = 2
    TIMEOUT = 3
    MEMORYOUT = 4

    # ABORTED = 4

    # Only relevant for SH/HB. Run might have a results, but should not be considered further.
    # By default, these runs will always be considered for building the model. Potential use cases:
    # 1) The run has converged and does not benefit from a higher budget
    # 2) The run has exhausted given resources and will not benefit from higher budgets
    # DONOTADVANCE = 6

    # STOP = 7  # Gracefully exit SMAC and wait for currently executed runs to finish.
