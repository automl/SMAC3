from __future__ import annotations

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import LatinHypercube

from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.utils.configspace import transform_continuous_designs

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class LatinHypercubeInitialDesign(AbstractInitialDesign):
    """Latin Hypercube initial design. See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html for further information.
    """

    def _select_configurations(self) -> list[Configuration]:
        params = list(self._configspace.values())

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        lhd = LatinHypercube(d=len(params) - constants, seed=self._rng.randint(0, 1000000)).random(n=self._n_configs)

        return transform_continuous_designs(
            design=lhd, origin="Initial Design: Latin Hypercube", configspace=self._configspace
        )
