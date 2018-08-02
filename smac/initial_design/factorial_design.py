import typing
import itertools

import numpy as np

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant, NumericalHyperparameter, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters

from smac.initial_design.multi_config_initial_design import \
    MultiConfigInitialDesign
from smac.intensification.intensification import Intensifier

from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from symbol import factor
from statsmodels.sandbox.formula import Factor

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

    def _select_configurations(self):
        """Selects a single configuration to run

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """
        
        cs = self.scenario.cs
        params = cs.get_hyperparameters()
        
        values = [] 
        for param in params:
            if isinstance(param, Constant):
                v = [param.value]
            elif isinstance(param, NumericalHyperparameter):
                v = [param.lower, param.upper]
            elif isinstance(param, CategoricalHyperparameter):
                v = list(param.choices)
            elif isinstance(param, OrdinalHyperparameter):
                v = [param.sequence[0], param.sequence[-1]]
            values.append(v)
        factorial_design = itertools.product(*values)
        
        self.logger.debug("Initial Design")
        configs = []
        for design in factorial_design:
            conf_dict = dict([(p.name,v) for p,v in zip(params,design)])
            conf = deactivate_inactive_hyperparameters(conf_dict, cs)
            configs.append(conf)
            self.logger.debug(conf)

        self.logger.debug("Size of factorial design: %d" %(len(configs)))
            
        return configs
            
        