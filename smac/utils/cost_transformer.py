from __future__ import annotations

from typing import List, Tuple

import numpy as np

from smac.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.utils.multi_objective import normalize_costs


class CostTransformer:
    """
    Utility Class for transforming raw cost values retrieved from the RunHistory.

    This class is stateless and operates purely on already extracted cost data.
    It does not access the RunHistory directly.

    Typical Workflow:
    -----------------
    1. Retrieve raw costs from RunHistory:
        raw_costs = runhistory.get_costs(config, isb_keys)

    2. Transform them using CostTransformer:
        cost = CostTransformer.aggregate(
            raw_costs,
            method="mean",
            normalize=True,
            bounds=runhistory.objective_bounds,
            scalarize=True,
            algorithm=runhistory.multi_objective_algorithm,
        )
    """

    @staticmethod
    def mean(costs: List[float | List[float]]) -> float | List[float]:
        """
        Compute the mean of costs.
        - Single-objective: returns float
        - Multi-objective: returns list of floats (mean per objective)
        """
        if not costs:
            return np.nan

        if isinstance(costs[0], list):
            return np.mean(np.array(costs), axis=0).tolist()

        return float(np.mean(costs))

    @staticmethod
    def sum(costs: List[float | List[float]]) -> float | List[float]:
        """
        Compute the sum of costs.
        - Single-objective: returns float
        - Multi-objective: returns list of floats (sum per objective)
        """
        if not costs:
            return np.nan

        if isinstance(costs[0], list):
            return np.sum(np.array(costs), axis=0).tolist()

        return float(np.sum(costs))

    @staticmethod
    def min(costs: List[float | List[float]]) -> float | List[float]:
        """
        Compute minimum of costs.
        - Single-objective: returns float
        - Multi-objective: returns list of floats (min per objective)
        """
        if not costs:
            return np.nan

        if isinstance(costs[0], list):
            return np.min(np.array(costs), axis=0).tolist()

        return float(np.min(costs))

    @staticmethod
    def normalize(
        costs: List[float],
        bounds: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Normalize multi-objective costs using given bounds.
        Each objective is scaled independently to [0, 1].
        """
        return normalize_costs(costs, bounds)

    @staticmethod
    def scalarize(
        costs: List[float],
        algorithm: AbstractMultiObjectiveAlgorithm,
    ) -> float:
        """Scalarize multi-objective costs into single value."""
        return algorithm(costs)

    @staticmethod
    def aggregate(
        costs: List[float | List[float]],
        *,
        method: str = "mean",
        normalize: bool = False,
        bounds: List[Tuple[float, float]] | None = None,
        scalarize: bool = False,
        algorithm: AbstractMultiObjectiveAlgorithm | None = None,
    ) -> float | List[float]:
        """
        High-level convenience function to transform raw costs.

        This function combines:
        - aggregation (mean / sum / min)
        - optional normalization
        - optional scalarization

        Parameters
        ----------
        costs : list[float | list[float]]
            Raw costs from RunHistory. Each entry corresponds to one trial.
        method : str, defaults to "mean"
            Aggregation method: "mean", "sum", or "min"
        normalize : bool, defaults to False
            Whether to normalize multi-objective costs
        bounds : list[Tuple[float, float]] | None, defaults to None
            Required if normalize=True
        scalarize: bool, defaults to False
            Whether to scalarize multi-objective costs
        algorithm : AbstractMultiObjectiveAlgorithm | None, defaults to None
            Required if scalarize=True

        Returns
        -------
        float or list of floats

        Notes
        -----
        - Normalization and scalarization are only applied to multi-objective costs.
        - Operations are applied in this order:
            1. Aggregation
            2. normalization (optional)
            3. scalarization (optional)
        """
        if not costs:
            return np.nan

        if method == "mean":
            result = CostTransformer.mean(costs)
        elif method == "min":
            result = CostTransformer.min(costs)
        elif method == "sum":
            result = CostTransformer.sum(costs)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        if isinstance(result, list):
            if normalize:
                if bounds is None:
                    raise ValueError("Bounds must be provided when normalize=True")
                result = CostTransformer.normalize(result, bounds)

            if scalarize:
                if algorithm is None:
                    raise ValueError("algorithm must be provided when scalarize=True")
                result = CostTransformer.scalarize(result, algorithm)

        return result
