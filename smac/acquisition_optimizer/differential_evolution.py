from __future__ import annotations

import abc
from typing import Callable, Iterator, List, Optional, Set, Tuple, Union

import copy
import itertools
import logging
import time

import numpy as np

from smac.acquisition_function import AbstractAcquisitionFunction
from smac.acquisition_optimizer import AbstractAcquisitionOptimizer
from smac.chooser.random_chooser import ChooserNoCoolDown, RandomChooser
from smac.configspace import (
    Configuration,
    ConfigurationSpace,
    ForbiddenValueError,
    convert_configurations_to_array,
    get_one_exchange_neighbourhood,
)
from smac.runhistory.runhistory import RunHistory
from smac.utils.stats import Stats


class DifferentialEvolution(AbstractAcquisitionOptimizer):
    """Get candidate solutions via DifferentialEvolutionSolvers."""

    def _maximize(
        self,
        previous_configs: List[Configuration],
        num_points: int,
        _sorted: bool = False,
    ) -> List[Tuple[float, Configuration]]:
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
        challengers : List[Tuple[float, Configuration]]
            A list consisting of Tuple(acquisition_value, :class:`smac.configspace.Configuration`).
        """
        from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

        configs = []

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
