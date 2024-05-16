from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

from smac.acquisition.maximizer import AbstractAcquisitionMaximizer

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class DifferentialEvolution(AbstractAcquisitionMaximizer):
    """Get candidate solutions via `DifferentialEvolutionSolvers` from scipy.

    According to scipy 1.9.2 documentation:

    'Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient methods) to find the minimum,
    and can search large areas of candidate space, but often requires larger numbers of function
    evaluations than conventional gradient-based techniques.
    The algorithm is due to Storn and Price [1].'

    [1] Storn, R and Price, K, Differential Evolution - a Simple and Efficient Heuristic for Global
     Optimization over Continuous Spaces, Journal of Global Optimization, 1997, 11, 341 - 359.
    """

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        # n_points is not used here, but is required by the interface

        configs: list[tuple[float, Configuration]] = []

        def func(x: np.ndarray) -> np.ndarray:
            assert self._acquisition_function is not None
            return -self._acquisition_function([Configuration(self._configspace, vector=x)])

        ds = DifferentialEvolutionSolver(
            func,
            bounds=[[0, 1] for _ in range(len(self._configspace))],
            args=(),
            strategy="best1bin",
            maxiter=1000,
            popsize=50,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=self._rng.randint(1000),
            polish=True,
            callback=None,
            disp=False,
            init="latinhypercube",
            atol=0,
        )

        _ = ds.solve()
        for pop, val in zip(ds.population, ds.population_energies):
            rc = Configuration(self._configspace, vector=pop)
            rc.origin = "Acquisition Function Maximizer: Differential Evolution"
            configs.append((-val, rc))

        configs.sort(key=lambda t: t[0])
        configs.reverse()

        return configs
