from __future__ import annotations
from typing import Any

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import AbstractMultiObjectiveAlgorithm
from smac.scenario import Scenario


class ParEGO(AbstractMultiObjectiveAlgorithm):

    def __init__(
        self,
        scenario: Scenario,
        seed: int | None = None,
        rho: float = 0.05,
    ):
        super(ParEGO, self).__init__(scenario=scenario, seed=seed)
        self.rho = rho
        self.theta = None
        self.update_on_iteration_start()

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self.seed,
            "rho": self.rho,
        }

    def update_on_iteration_start(self) -> None:
        """Update the internal state for each SMAC SMBO iteration."""
        self.theta = self.rng.rand(self.num_objectives)
        # Normalize st all theta values sum up to 1
        self.theta = self.theta / (np.sum(self.theta) + 1e-10)

    def __call__(self, values: list[float]) -> float:
        """Transform a multi-objective loss to a single loss.

        Parameters
        ----------
        values : list[float]
            Normalized values.

        Returns
        -------
        cost : float
            Combined cost.
        """
        # Weight the values
        theta_f = self.theta * values
        return np.max(theta_f, axis=0) + self.rho * np.sum(theta_f, axis=0)
