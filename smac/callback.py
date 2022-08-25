from __future__ import annotations

from abc import abstractmethod

import smac
from smac.runhistory import RunInfo, RunValue

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Callback:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def on_start(self, smbo: smac.main.BaseSMBO) -> None:
        """Called before the optimization starts."""
        pass

    @abstractmethod
    def on_end(self, smbo: smac.main.BaseSMBO) -> None:
        """Called after the optimization finished."""
        pass

    @abstractmethod
    def on_ask_start(self, smbo: smac.main.BaseSMBO) -> None:
        """Called before the intensification asks for new configurations. Essentially, this callback is called
        before the surrogate model is trained and before the acquisition function is called.
        """
        pass

    @abstractmethod
    def on_ask_end(self, smbo: smac.main.BaseSMBO, configurations: list[Configuration]) -> None:
        """Called before the intensification asks for new configurations. Essentially, this callback is called
        before the surrogate model is trained and before the acquisition function is called.
        """
        pass

    @abstractmethod
    def on_iteration_start(self, smbo: smac.main.BaseSMBO) -> None:
        """Called before the next run is sampled."""
        pass

    @abstractmethod
    def on_iteration_end(self, smbo: smac.main.BaseSMBO, info: RunInfo, value: RunValue) -> bool | None:
        """Called after the finished run is added to the runhistory. Optionally, return `False` to
        gracefully stop the optimization."""
        pass
