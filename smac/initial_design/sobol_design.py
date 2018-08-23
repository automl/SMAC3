import typing
import itertools

import numpy as np
from sobol_seq import i4_sobol_generate

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import FloatHyperparameter
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


class SobolDesign(MultiConfigInitialDesign):
    """ Sobol sequence design

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
        
        sobol = i4_sobol_generate(len(params), 10*len(params))
        
        for idx, param in enumerate(params):
            if isinstance(param, FloatHyperparameter):
                sobol[:,idx] = sobol[:,idx] * (param.upper - param.lower) + param.lower
            else:
                raise ValueError("only FloatHyperparameters supported in LHD")
        
        self.logger.debug("Initial Design")
        configs = []
        # add middle point in space

        for design in sobol:
            conf_dict = dict([(p.name,v) for p,v in zip(params,design)])
            conf = deactivate_inactive_hyperparameters(conf_dict, cs)
            conf.origin = "Sobol"
            configs.append(conf)
            self.logger.debug(conf)

        self.logger.debug("Length of Sobol sequence: %d" %(len(configs)))
            
        return configs
            
        