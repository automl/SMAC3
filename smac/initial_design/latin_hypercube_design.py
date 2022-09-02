from __future__ import annotations

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import LatinHypercube

from smac.initial_design.initial_design import AbstractInitialDesign

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class LatinHypercubeInitialDesign(AbstractInitialDesign):
    """Latin Hypercube design.

    Attributes
    ----------
    configs : list[Configuration]
        List of configurations to be evaluated
        Don't pass configs to the constructor;
        otherwise factorial design is overwritten
    """

    def _select_configurations(self) -> list[Configuration]:
        """Selects a single configuration to run.

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """
        params = self.configspace.get_hyperparameters()

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        lhd = LatinHypercube(d=len(params) - constants, seed=self.rng.randint(0, 1000000)).random(n=self.n_configs)

        return self._transform_continuous_designs(
            design=lhd, origin="Latin Hypercube Initial Design", configspace=self.configspace
        )
