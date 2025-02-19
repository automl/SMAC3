from __future__ import annotations

from enum import IntEnum

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class StatusType(IntEnum):
    """Class to define status types of configs."""

    RUNNING = 0  # In case a job was submitted, but it has not finished.
    SUCCESS = 1
    CRASHED = 2
    TIMEOUT = 3
    MEMORYOUT = 4
