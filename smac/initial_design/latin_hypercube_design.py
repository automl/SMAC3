import typing

import numpy as np

from pyDOE import lhs

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import NumericalHyperparameter, \
    Constant, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters

from smac.initial_design.initial_design import InitialDesign

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class LHDesign(InitialDesign):
    """Latin Hypercube design

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

        # manual seeding of lhd design
        np.random.seed(self.rng.randint(1,2*20))

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        lhd = lhs(n=len(params)-constants, samples=self.init_budget)

        for idx, param in enumerate(params):
            if isinstance(param, NumericalHyperparameter):
                continue
                #lhd[:, idx] = lhd[:, idx] * (param.upper - param.lower) + param.lower
            elif isinstance(param, Constant):
                # add a vector with zeros
                lhd_ = np.zeros(np.array(lhd.shape) + np.array((0, 1)))
                lhd_[:, :idx] = lhd[:, :idx]
                lhd_[:, idx+1:] = lhd[:, idx:]
                lhd = lhd_
            elif isinstance(param, CategoricalHyperparameter):
                lhd[:, idx] = np.array(lhd[:, idx] * len(param.choices), dtype=np.int)
            elif isinstance(param, OrdinalHyperparameter):
                lhd[:, idx] = np.array(lhd[:, idx] * len(param.sequence), dtype=np.int)
            else:
                raise ValueError("Hyperparamet not supported in LHD")

        self.logger.debug("Initial Design")
        configs = []
        # add middle point in space
        for design in lhd:
            conf = deactivate_inactive_hyperparameters(configuration=None,
                                                       configuration_space=cs,
                                                       vector=design)
            conf.origin = "LHD"
            configs.append(conf)
            self.logger.debug(conf)

        self.logger.debug("Size of lhd: %d" %(len(configs)))

        return configs
