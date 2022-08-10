from __future__ import annotations

from typing import List

from ConfigSpace import Configuration

from smac.initial_design.initial_design import InitialDesign

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class RandomInitialDesign(InitialDesign):
    """Initial design that evaluates random configurations."""

    def _select_configurations(self) -> list[Configuration]:
        """Select a random configuration.

        Returns
        -------
        config: Configuration()
            Initial incumbent configuration
        """
        configs = self.configspace.sample_configuration(size=self.n_configs)
        if self.n_configs == 1:
            configs = [configs]
        for config in configs:
            config.origin = "Random Initial Design"
        return configs
