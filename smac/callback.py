from __future__ import annotations

from abc import abstractmethod

import smac
from smac.runhistory import TrialInfo, TrialValue, TrialInfoIntent
from ConfigSpace import Configuration

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Callback:
    """Callback interface with several methods that are called at different stages of the optimization process."""

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
    def on_iteration_start(self, smbo: smac.main.BaseSMBO) -> None:
        """Called before the next run is sampled."""
        pass

    @abstractmethod
    def on_iteration_end(self, smbo: smac.main.BaseSMBO) -> None:
        """Called after an iteration ended."""
        pass

    @abstractmethod
    def on_next_configurations_start(self, smbo: smac.main.BaseSMBO) -> None:
        """Called before the intensification asks for new configurations. Essentially, this callback is called
        before the surrogate model is trained and before the acquisition function is called.
        """
        pass

    @abstractmethod
    def on_next_configurations_end(self, smbo: smac.main.BaseSMBO, configurations: list[Configuration]) -> None:
        """Called after the intensification asks for new configurations. Essentially, this callback is called
        before the surrogate model is trained and before the acquisition function is called.
        """
        pass

    @abstractmethod
    def on_ask_start(self, smbo: smac.main.BaseSMBO) -> None:
        """Called before the intensifier is asked for the next trial."""
        pass

    @abstractmethod
    def on_ask_end(self, smbo: smac.main.BaseSMBO, intent: TrialInfoIntent, info: TrialInfo) -> None:
        """Called after the intensifier is asked for the next trial."""
        pass

    @abstractmethod
    def on_tell_start(self, smbo: smac.main.BaseSMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Called before the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization."""
        pass

    @abstractmethod
    def on_tell_end(self, smbo: smac.main.BaseSMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Called after the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization."""
        pass
