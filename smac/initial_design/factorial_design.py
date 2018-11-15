import itertools
import typing

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant, NumericalHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters
import numpy as np

from smac.initial_design.multi_config_initial_design import \
    MultiConfigInitialDesign


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class FactorialInitialDesign(MultiConfigInitialDesign):
    """ Factorial initial design

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

    def _select_configurations(self) -> Configuration:
        """Selects a single configuration to run

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """

        cs = self.scenario.cs
        params = cs.get_hyperparameters()

        values = []
        mid = []
        for param in params:
            if isinstance(param, Constant):
                v = [param.value]
                mid.append(param.value)
            elif isinstance(param, NumericalHyperparameter):
                v = [param.lower, param.upper]
                mid.append(np.average([param.lower, param.upper]))
            elif isinstance(param, CategoricalHyperparameter):
                v = list(param.choices)
                mid.append(param.choices[0])
            elif isinstance(param, OrdinalHyperparameter):
                v = [param.sequence[0], param.sequence[-1]]
                l = len(param.sequence)
                mid.append(param.sequence[int(l/2)])
            values.append(v)

        factorial_design = itertools.product(*values)

        self.logger.debug("Initial Design")
        configs = [cs.get_default_configuration()]
        # add middle point in space
        conf_dict = dict([(p.name,v) for p,v in zip(params,mid)])
        middle_conf = deactivate_inactive_hyperparameters(conf_dict, cs)
        configs.append(middle_conf)

        # add corner points
        for design in factorial_design:
            conf_dict = dict([(p.name,v) for p,v in zip(params,design)])
            conf = deactivate_inactive_hyperparameters(conf_dict, cs)
            conf.origin = "Factorial Design"
            configs.append(conf)
            self.logger.debug(conf)

        self.logger.debug("Size of factorial design: %d" %(len(configs)))

        return configs
