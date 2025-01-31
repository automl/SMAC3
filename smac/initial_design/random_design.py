from __future__ import annotations

from ConfigSpace import Configuration

from smac.initial_design.abstract_initial_design import AbstractInitialDesign

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class RandomInitialDesign(AbstractInitialDesign):
    """Initial design that evaluates random configurations."""

    def _select_configurations(self) -> list[Configuration]:
        if self._n_configs == 1:
            configs = [self._configspace.sample_configuration()]
        else:
            configs = self._configspace.sample_configuration(size=self._n_configs)
        for config in configs:
            config.origin = "Initial Design: Random"
        return configs
