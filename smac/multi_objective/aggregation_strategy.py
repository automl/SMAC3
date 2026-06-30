from __future__ import annotations

from typing import Any

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.scenario import Scenario


class MeanAggregationStrategy(AbstractMultiObjectiveAlgorithm):
    """A class to mean-aggregate multi-objective costs to a single cost.
    If `objective_weights` are provided via the scenario, each objective is weighted
    accordingly when computing the mean; otherwise, all objectives are treated equally.

    Parameters
    ----------
    scenario : Scenario
    """

    def __init__(
        self,
        scenario: Scenario,
    ):
        super(MeanAggregationStrategy, self).__init__()

        self._objective_weights = scenario.objective_weights

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "objective_weights": self._objective_weights,
        }

    def __call__(self, values: list[float]) -> float:  # noqa: D102
        return float(np.average(values, axis=0, weights=self._objective_weights))


class NoAggregationStrategy(AbstractMultiObjectiveAlgorithm):
    """A class to not aggregate multi-objective losses into a single objective losses."""

    def __call__(self, values: list[float]) -> list[float]:  # type: ignore[override]
        """
        Not transform a multi-objective loss to a single loss.

        Parameters
        ----------
        values : list[float]
            Normalized cost values.

        Returns
        -------
        costs : list[float]
            costs.
        """
        return values
