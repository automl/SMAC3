from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from smac.smbo import SMBO

from smac.runhistory import RunInfo, RunValue

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Callback:
    def __init__(self):
        pass

    @abstractmethod
    def on_start(self, smbo: SMBO) -> None:
        """Called before the optimization starts."""
        pass

    @abstractmethod
    def on_end(self, smbo: SMBO) -> None:
        """Called after the optimization finished."""
        pass

    @abstractmethod
    def on_iteration_start(self, smbo: SMBO) -> None:
        """Called before the next run is sampled."""
        pass

    @abstractmethod
    def on_iteration_end(self, smbo: SMBO, info: RunInfo, value: RunValue) -> bool | None:
        """Called after the finished run is added to the runhistory. Optionally, return `False` to
        gracefully stop the optimization."""
        pass
