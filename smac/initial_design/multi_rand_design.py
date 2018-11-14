import typing

from ConfigSpace.configuration_space import Configuration

from smac.initial_design.multi_config_initial_design import \
    MultiConfigInitialDesign


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class MultiRandDesign(MultiConfigInitialDesign):
    """ Initial design that evaluates multiple random configurations

    Attributes
    ----------
    configs : typing.List[Configuration]
        List of configurations to be evaluated
        Don't pass configs to the constructor;
        otherwise factorial design is overwritten
    intensifier
    runhistory
    aggregate_func
    """

    def _select_configurations(self) -> typing.List[Configuration]:
        """Selects a single configuration to run

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """

        cs = self.scenario.cs
        self.logger.debug("Sample %d random configs for initial design" %(self.init_budget))
        return cs.sample_configuration(size=self.init_budget)
