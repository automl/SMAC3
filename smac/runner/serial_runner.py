from __future__ import annotations

from typing import Iterator

from smac.runhistory import TrialInfo, TrialValue
from smac.runner.abstract_runner import AbstractRunner

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class SerialRunner(AbstractRunner):
    """Interface to submit and collect a job in a serial fashion.

    Dictates what a worker should do to convert a configuration/instance/seed to a result.

    This class is expected to be extended via the implementation of a run() method for
    the desired task.
    """
