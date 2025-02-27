from __future__ import annotations

from typing import Any

import warnings

from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import Sobol

from smac.initial_design.abstract_initial_design import AbstractInitialDesign
from smac.utils.configspace import transform_continuous_designs

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class SobolInitialDesign(AbstractInitialDesign):
    """Sobol sequence design with a scrambled Sobol sequence. See
    https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html for further information.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if len(list(self._configspace.values())) > 21201:
            raise ValueError(
                "The default initial design Sobol sequence can only handle up to 21201 dimensions. "
                "Please use a different initial design, such as the Latin Hypercube design."
            )

    def _select_configurations(self) -> list[Configuration]:
        params = list(self._configspace.values())

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        dim = len(params) - constants
        sobol_gen = Sobol(d=dim, scramble=True, seed=self._rng.randint(low=0, high=10000000))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = sobol_gen.random(self._n_configs)

        return transform_continuous_designs(design=sobol, origin="Initial Design: Sobol", configspace=self._configspace)
