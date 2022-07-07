from __future__ import annotations

from typing import Optional

import numpy as np

from smac.multi_objective.aggregation_strategy import AggregationStrategy


class ParEGO(AggregationStrategy):
    def __init__(
        self,
        rng: Optional[np.random.RandomState] = None,
        rho: float = 0.05,
    ):
        super(ParEGO, self).__init__(rng=rng)
        self.rho = rho

    def __call__(self, values: list[float]) -> float:
        """
        Transform a multi-objective loss to a single loss.

        Parameters
        ----------
        values : list[float]
            Normalized values.

        Returns
        -------
        cost : float
            Combined cost.
        """
        # Then we have to compute the weight
        theta = self.rng.rand(len(values))

        # Normalize st all theta values sum up to 1
        theta = theta / (np.sum(theta) + 1e-10)

        # Weight the values
        theta_f = theta * values
        return np.max(theta_f, axis=0) + self.rho * np.sum(theta_f, axis=0)
