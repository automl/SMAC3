from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.weighted_acquisition_function import (
    WeightedAcquisitionFunction,
)
from smac.acquisition.weight.feasibility import FeasibilityWeight
from smac.model.abstract_model import AbstractModel
from smac.runhistory.runhistory import RunHistory
from smac.utils.constraints import OutcomeConstraint
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class ConstrainedAcquisitionFunction(WeightedAcquisitionFunction):
    r"""Weight an acquisition function by the probability that the output constraints are satisfied.

    The constrained outputs are modelled separately from the objective, and the acquisition value is multiplied
    by the probability that every bound holds:

    $$
    a_{c}(\mathbf{X}) = a(\mathbf{X}) \prod_i P(y_i(\mathbf{X}) \text{ satisfies constraint } i)
    $$

    See "Bayesian Optimization with Inequality Constraints" by Jacob Gardner et al. [[GKZ+14][GKZ+14]] for
    further details.

    This is a `WeightedAcquisitionFunction` carrying a single `FeasibilityWeight`, which is where the mechanism
    lives and where it is documented. Reach for `WeightedAcquisitionFunction` directly when the acquisition
    function should carry more than the constraints, such as a user prior over the optimum as well; wrapping this
    class around another wrapper is refused.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction
        The acquisition function to weight.
    constraints : list[OutcomeConstraint]
        The constraints to enforce, as parsed from ``Scenario.constraints``.
    constraint_model : AbstractModel
        Surrogate model for the constrained outputs, predicting one column per constraint in the order the
        constraints are given. It is trained on the raw observed values, because the bounds are stated in raw
        units, and is therefore kept separate from the objective's own model and encoder.
    feasibility_floor : float, defaults to 1e-12
        Lowest possible value of the feasibility weight. Keeps the ranking of configurations intact when every
        probability underflows to zero.
    """

    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        constraints: list[OutcomeConstraint],
        constraint_model: AbstractModel,
        feasibility_floor: float = 1e-12,
    ) -> None:
        if len(constraints) == 0:
            raise ValueError("A constrained acquisition function needs at least one constraint.")

        feasibility = FeasibilityWeight(
            constraints=constraints,
            constraint_model=constraint_model,
            floor=feasibility_floor,
        )

        super().__init__(acquisition_function, [feasibility])

        self._feasibility = feasibility

    @property
    def name(self) -> str:  # noqa: D102
        return f"Constrained Acquisition Function ({self._acquisition_function.__class__.__name__})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        # Deliberately not the metadata of the weighted acquisition function: this dictionary ends up in the name
        # of the output directory, and changing it would keep existing runs from being continued.
        return {
            "name": self.__class__.__name__,
            "acquisition_function": self._acquisition_function.meta,
            "constraints": [str(constraint) for constraint in self._constraints],
            "constraint_model": self._constraint_model.meta,
            "feasibility_floor": self._feasibility_floor,
        }

    @property
    def feasibility(self) -> FeasibilityWeight:
        """The weight carrying the constraints."""
        return self._feasibility

    @property
    def _constraints(self) -> list[OutcomeConstraint]:
        return self._feasibility._constraints

    @property
    def _constraint_model(self) -> AbstractModel:
        return self._feasibility._constraint_model

    @property
    def _feasibility_floor(self) -> float:
        return self._feasibility._floor

    @property
    def _trained(self) -> bool:
        return self._feasibility._trained

    @property
    def _has_feasible(self) -> bool:
        return self._feasibility._has_feasible

    def _collect_constraint_data(self, runhistory: RunHistory) -> tuple[list[Configuration], np.ndarray]:
        return self._feasibility._collect_constraint_data(runhistory)

    def _as_dicts(self, constraint_values: np.ndarray) -> list[dict[str, float]]:
        return self._feasibility._as_dicts(constraint_values)
