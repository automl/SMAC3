from __future__ import annotations

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)


class MeanAggregationStrategy(AbstractMultiObjectiveAlgorithm):
    """A class to mean-aggregate multi-objective losses to a single loss"""

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
        return np.mean(values, axis=0)
