import typing

import numpy as np

import pyDOE

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant

from smac.initial_design.initial_design import InitialDesign

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2019, AutoML"
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

        # seeding of lhd design
        np.random.seed(self.rng.randint(1,2*20))

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        lhd = pyDOE.lhs(n=len(params)-constants, samples=self.init_budget)

        return self._transform_continuous_designs(design=lhd,
                                                  origin='LHD',
                                                  cs=cs)
