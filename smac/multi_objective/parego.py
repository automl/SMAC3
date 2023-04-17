from __future__ import annotations

from typing import Any

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
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
        super(ParEGO, self).__init__()

        if seed is None:
            seed = scenario.seed

        self._n_objectives = scenario.count_objectives()
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        self._rho = rho
        # Will be set on starting an SMBO iteration
        self._theta: np.ndarray | None = None

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "name": self.__class__.__name__,
                "rho": self._rho,
                "seed": self._seed,
            }
        )

        return meta

    def update_on_iteration_start(self) -> None:  # noqa: D102
        self._theta = self._rng.rand(self._n_objectives)

        # Normalize so that all theta values sum up to 1
        self._theta = self._theta / (np.sum(self._theta) + 1e-10)

    def __call__(self, values: list[float]) -> float:  # noqa: D102
        # Weight the values
        if self._theta is None:
            raise ValueError("Iteration not yet initalized; Call `update_on_iteration_start()` first")

        theta_f = self._theta * values
        return float(np.max(theta_f, axis=0) + self._rho * np.sum(theta_f, axis=0))
