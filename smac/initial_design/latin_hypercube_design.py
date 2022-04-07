import typing

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import LatinHypercube

from smac.initial_design.initial_design import InitialDesign

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2019, AutoML"
__license__ = "3-clause BSD"


class LHDesign(InitialDesign):
    """Latin Hypercube design.

    Attributes
    ----------
    configs : typing.List[Configuration]
        List of configurations to be evaluated
        Don't pass configs to the constructor;
        otherwise factorial design is overwritten
    """

    def _select_configurations(self) -> typing.List[Configuration]:
        """Selects a single configuration to run.

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """
        params = self.cs.get_hyperparameters()

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        lhd = LatinHypercube(d=len(params) - constants, seed=self.rng.randint(0, 1000000)).random(n=self.init_budget)

        return self._transform_continuous_designs(design=lhd, origin="LHD", cs=self.cs)
