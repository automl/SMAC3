from __future__ import annotations

import abc
from typing import Callable, Iterator, Optional, Set, Union

import numpy as np

from smac.acquisition import AbstractAcquisitionOptimizer
from smac.configspace import Configuration
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class DifferentialEvolution(AbstractAcquisitionOptimizer):
    """Get candidate solutions via DifferentialEvolutionSolvers."""

    def _maximize(
        self,
        previous_configs: list[Configuration],
        num_points: int,
        _sorted: bool = False,
    ) -> list[tuple[float, Configuration]]:
        """DifferentialEvolutionSolver.

        Parameters
        ----------
        previous_configs: List[Configuration]
            Previously evaluated configurations.
        num_points: int
            Number of points to be sampled.
        _sorted: bool
            whether random configurations are sorted according to acquisition function

        Returns
        -------
        challengers : list[tuple[float, Configuration]]
            A list consisting of Tuple(acquisition_value, :class:`smac.configspace.Configuration`).
        """
        from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

        configs : tuple[float, Configuration] = []

        def func(x: np.ndarray) -> np.ndarray:
            assert self.acquisition_function is not None
            return -self.acquisition_function([Configuration(self.configspace, vector=x)])

        ds = DifferentialEvolutionSolver(
            func,
            bounds=[[0, 1], [0, 1]],
            args=(),
            strategy="best1bin",
            maxiter=1000,
            popsize=50,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self.rng.randint(1000),
            polish=True,
            callback=None,
            disp=False,
            init="latinhypercube",
            atol=0,
        )

        _ = ds.solve()
        for pop, val in zip(ds.population, ds.population_energies):
            rc = Configuration(self.configspace, vector=pop)
            rc.origin = "DifferentialEvolution"
            configs.append((-val, rc))

        configs.sort(key=lambda t: t[0])
        configs.reverse()
        return configs
