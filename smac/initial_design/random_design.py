from __future__ import annotations

from ConfigSpace import Configuration

from smac.initial_design.initial_design import InitialDesign

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class RandomInitialDesign(InitialDesign):
    """Initial design that evaluates random configurations."""

    def _select_configurations(self) -> list[Configuration]:
        """Select random configurations.

        Returns
        -------
        configs: list[Configuration]
            The list of configurations to be evaluated in the initial design.
        """
        configs = self.configspace.sample_configuration(size=self.n_configs)
        if self.n_configs == 1:
            configs = [configs]
        for config in configs:
            config.origin = "Random Initial Design"
        return configs