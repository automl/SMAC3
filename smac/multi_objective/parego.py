from __future__ import annotations
from typing import Any

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import AbstractMultiObjectiveAlgorithm
from smac.scenario import Scenario


class ParEGO(AbstractMultiObjectiveAlgorithm):
    """ParEGO implementation based on https://www.cs.bham.ac.uk/~jdk/UKCI-2015.pdf.

    Parameters
    ----------
    scenario : Scenario
    rho : float, defaults to 0.05
        A small positive value.
    seed : int | None, defaults to None
    """

    def __init__(
        self,
        scenario: Scenario,
        rho: float = 0.05,
        seed: int | None = None,
    ):
        super(ParEGO, self).__init__(scenario=scenario, seed=seed)
        self._rho = rho
        self._theta = self._rng.rand(self._n_objectives)
        self.update_on_iteration_start()

    def get_meta(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "rho": self._rho,
            "seed": self._seed,
        }

    def update_on_iteration_start(self) -> None:
        self._theta = self._rng.rand(self._n_objectives)

        # Normalize so that all theta values sum up to 1
        self._theta = self._theta / (np.sum(self._theta) + 1e-10)

    def __call__(self, values: list[float]) -> float:
        # Weight the values
        theta_f = self._theta * values
        return np.max(theta_f, axis=0) + self._rho * np.sum(theta_f, axis=0)
