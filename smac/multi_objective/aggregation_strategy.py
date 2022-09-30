from __future__ import annotations

from typing import Any

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.scenario import Scenario


class MeanAggregationStrategy(AbstractMultiObjectiveAlgorithm):
    """A class to mean-aggregate multi-objective costs to a single cost.

    Parameters
    ----------
    scenario : Scenario
    objective_weights : list[float] | None, defaults to None
        Weights for an weighted average. Must be of the same length as the number of objectives.
    """

    def __init__(
        self,
        scenario: Scenario,
        objective_weights: list[float] | None = None,
    ):
        super(MeanAggregationStrategy, self).__init__()

        if objective_weights is not None and scenario.count_objectives() != len(objective_weights):
            raise ValueError("Number of objectives and number of weights must be equal.")

        self._objective_weights = objective_weights

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "objective_weights": self._objective_weights,
        }

    def __call__(self, values: list[float]) -> float:  # noqa: D102
        return float(np.average(values, axis=0, weights=self._objective_weights))
