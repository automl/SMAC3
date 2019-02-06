from typing import List

from ConfigSpace import Configuration

from smac.initial_design.initial_design import InitialDesign


__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class RandomConfigurations(InitialDesign):
    """Initial design that evaluates random configurations."""

    def _select_configurations(self) -> List[Configuration]:
        """Select a random configuration.

        Returns
        -------
        config: Configuration()
            Initial incumbent configuration
        """

        configs = self.scenario.cs.sample_configuration(size=self.init_budget)
        if self.init_budget == 1:
            configs = [configs]
        for config in configs:
            config.origin = 'Random initial design.'
        return configs
