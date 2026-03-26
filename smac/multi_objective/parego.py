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
    If `objective_weights` are provided, scalarization weights
    are sampled from a Dirichlet distribution centered around these preferences.

    Parameters
    ----------
    scenario : Scenario
    rho : float, defaults to 0.05
        A small positive value.
    seed : int | None, defaults to None
    objective_weights : list[float] | None, defaults to None
        Optional preference weights to bias the search towards user preference.
        Must be non-negative and match the number of objectives.
    concentration_scale : float, defaults to 10.0
        Scaling factor used for the Dirichlet distribution used to sample
        scalarization weights when user preferences provided.
        - Low values -> more exploration (weights vary strongly)
        - High values -> stronger focus on user preferences
    """

    def __init__(
        self,
        scenario: Scenario,
        rho: float = 0.05,
        seed: int | None = None,
        objective_weights: list[float] | None = None,
        concentration_scale: float = 10.0,
    ):
        super(ParEGO, self).__init__()

        if seed is None:
            seed = scenario.seed

        self._n_objectives = scenario.count_objectives()
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self.concentration_scale = concentration_scale

        # Validate and normalize objective_weights
        if objective_weights is not None:
            if self._n_objectives != len(objective_weights):
                raise ValueError("Number of objectives and number of weights must be equal.")
            if any(w < 0 for w in objective_weights):
                raise ValueError("objective_weights must be non-negative.")

            w = np.asarray(objective_weights, dtype=float)
            self._objective_weights = w / np.sum(w)
        else:
            self._objective_weights = None

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
        """Sample new scalarization weights for the current iteration.

        Behavior depends on whether user preferences are provided:
        - No preferences:
            Uniform random weights on the simplex (classic ParEGO)
        - With preferences:
            Weights are sampled from a Dirichlet distribution centered
            around the user-defined objective weights.
        """
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
