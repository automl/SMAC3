from __future__ import annotations

from ConfigSpace import Configuration

from smac.initial_design.abstract_initial_design import AbstractInitialDesign

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class DefaultInitialDesign(AbstractInitialDesign):
    """Initial design that evaluates only the default configuration."""

    def _select_configurations(self) -> list[Configuration]:
        config = self._configspace.get_default_configuration()
        config.origin = "Initial Design: Default"
        return [config]
