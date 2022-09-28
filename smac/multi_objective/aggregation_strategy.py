from __future__ import annotations

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)


class MeanAggregationStrategy(AbstractMultiObjectiveAlgorithm):
    """A class to mean-aggregate multi-objective costs to a single cost."""

    def __call__(self, values: list[float]) -> float:  # noqa: D102
        return np.mean(values, axis=0)
