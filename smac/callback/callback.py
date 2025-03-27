from __future__ import annotations

from ConfigSpace import Configuration

import smac
from smac.runhistory import TrialInfo, TrialValue

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Callback:
    """Callback interface with several methods that are called at different stages of the optimization process."""

    def __init__(self) -> None:
        pass

    def on_start(self, smbo: smac.main.smbo.SMBO) -> None:
        """Called before the optimization starts."""
        pass

    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
        """Called after the optimization finished."""
        pass

    def on_iteration_start(self, smbo: smac.main.smbo.SMBO) -> None:
        """Called before the next run is sampled."""
        pass

    def on_iteration_end(self, smbo: smac.main.smbo.SMBO) -> None:
        """Called after an iteration ended."""
        pass

    def on_next_configurations_start(self, config_selector: smac.main.config_selector.ConfigSelector) -> None:
        """Called before the intensification asks for new configurations. Essentially, this callback is called
        before the surrogate model is trained and before the acquisition function is called.
        """
        pass

    def on_next_configurations_end(
        self, config_selector: smac.main.config_selector.ConfigSelector, config: Configuration
    ) -> None:
        """Called after the intensification asks for new configurations. Essentially, this callback is called
        before the surrogate model is trained and before the acquisition function is called.
        """
        pass

    def on_ask_start(self, smbo: smac.main.smbo.SMBO) -> None:
        """Called before the intensifier is asked for the next trial."""
        pass

    def on_ask_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo) -> None:
        """Called after the intensifier is asked for the next trial."""
        pass

    def on_tell_start(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Called before the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization.
        """
        pass

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Called after the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization.
        """
        pass
