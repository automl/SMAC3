import typing

from pyDOE import lhs

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import FloatHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters

from smac.initial_design.multi_config_initial_design import \
    MultiConfigInitialDesign

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class LHDesign(MultiConfigInitialDesign):
    """ Latin Hypercube design

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
        params = cs.get_hyperparameters()

        lhd = lhs(n=len(params), samples=self.init_budget)

        for idx, param in enumerate(params):
            if isinstance(param, FloatHyperparameter):
                lhd[:,idx] = lhd[:,idx] * (param.upper - param.lower) + param.lower
            else:
                raise ValueError("only FloatHyperparameters supported in LHD")

        self.logger.debug("Initial Design")
        configs = []
        # add middle point in space

        for design in lhd:
            conf_dict = dict([(p.name,v) for p,v in zip(params,design)])
            conf = deactivate_inactive_hyperparameters(conf_dict, cs)
            conf.origin = "LHD"
            configs.append(conf)
            self.logger.debug(conf)

        self.logger.debug("Size of lhd: %d" %(len(configs)))

        return configs
