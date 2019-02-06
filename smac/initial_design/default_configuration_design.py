from typing import List

from ConfigSpace import Configuration

from smac.initial_design.initial_design import InitialDesign


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class DefaultConfiguration(InitialDesign):

    """Initial design that evaluates default configuration"""

    def _select_configurations(self) -> List[Configuration]:

        """Selects the default configuration.

        Returns
        -------
        config: Configuration
            Initial incumbent configuration.
        """

        config = self.scenario.cs.get_default_configuration()
        config.origin = 'Default'
        return [config]
