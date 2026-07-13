from __future__ import annotations

from typing import Any

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.scenario import Scenario


class ParEGO(AbstractMultiObjectiveAlgorithm):
    """
    ParEGO implementation based on https://ieeexplore.ieee.org/abstract/document/1583627.
    If `objective_weights` are provided via the scenario, scalarization weights
    are sampled from a Dirichlet distribution centered around these weights.

    Parameters
    ----------
    scenario : Scenario
    rho : float, defaults to 0.05
        A small positive value.
    seed : int | None, defaults to None
    concentration_scale : float, defaults to 10.0
        Scaling factor for the Dirichlet distribution when `objective_weights` are provided:
        - Low values -> more exploration (weights vary strongly)
        - High values -> stronger focus on the scenario-provided objective_weights
    """

    def __init__(
        self,
        scenario: Scenario,
        rho: float = 0.05,
        seed: int | None = None,
        concentration_scale: float = 10.0,
    ):
        super(ParEGO, self).__init__()

        if seed is None:
            seed = scenario.seed

        self._n_objectives = scenario.count_objectives()
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self.concentration_scale = concentration_scale

        self._objective_weights = None
        if scenario.objective_weights is not None:
            w = np.asarray(scenario.objective_weights, dtype=float)
            self._objective_weights = w / np.sum(w)

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
                "objective_weights": self._objective_weights,
                "concentration_scale": self.concentration_scale,
            }
        )

        return meta

    def update_on_iteration_start(self) -> None:
        """Sample new scalarization weights for the current iteration."""
        if self._objective_weights is None:
            # Sample uniformly and normalize to simplex
            self._theta = self._rng.rand(self._n_objectives)
            self._theta = self._theta / (np.sum(self._theta) + 1e-10)
        else:
            # Dirichlet sampling around user preference vector
            w = self._objective_weights
            alpha = self.concentration_scale * w
            self._theta = self._rng.dirichlet(alpha)

    def __call__(self, values: list[float]) -> float:  # noqa: D102
        # Weight the values
        if self._theta is None:
            raise ValueError("Iteration not yet initialized; Call `update_on_iteration_start()` first")

        theta_f = self._theta * values
        return float(np.max(theta_f, axis=0) + self._rho * np.sum(theta_f, axis=0))
