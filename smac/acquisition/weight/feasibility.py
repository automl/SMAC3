from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import Configuration

from smac.acquisition.weight.abstract_weight import AbstractAcquisitionWeight
from smac.model.abstract_model import AbstractModel
from smac.runhistory.runhistory import RunHistory
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.constraints import (
    OutcomeConstraint,
    is_feasible,
    probability_of_feasibility,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class FeasibilityWeight(AbstractAcquisitionWeight):
    r"""The probability that every output constraint holds.

    $$w(\mathbf{X}) = \prod_i P(y_i(\mathbf{X}) \text{ satisfies constraint } i)$$

    See "Bayesian Optimization with Inequality Constraints" by Jacob Gardner et al. [[GKZ+14][GKZ+14]] for further
    details.

    Weighting rather than penalizing keeps the objective model clean: a penalty would make the surrogate fit a
    cliff that is not a feature of the objective, degrading its predictions inside the feasible region as well,
    and would discard the measured value that locates the boundary.

    The weight does not decay. A bound states a fact about the problem rather than a belief about where the optimum
    is, and a fact does not become less true as the run progresses.

    Two details make the difference between this working and not, and both are why a weight can do more than return
    a number:

    * The incumbent handed to the acquisition function is the best *feasible* one. The unconstrained incumbent is
      over-optimistic, because the best configuration seen so far may well violate a bound.
    * While nothing feasible has been observed there is no improvement to expect over, so the acquisition function
      is set aside and the search maximizes the probability of feasibility until the first feasible configuration
      turns up.

    Parameters
    ----------
    constraints : list[OutcomeConstraint]
        The constraints to enforce, as parsed from ``Scenario.constraints``.
    constraint_model : AbstractModel
        Surrogate model for the constrained outputs, predicting one column per constraint in the order the
        constraints are given. It is trained on the raw observed values, because the bounds are stated in raw
        units, and is therefore kept separate from the objective's own model and encoder.
    floor : float, defaults to 1e-12
        Lowest possible value of the weight. Keeps the ranking of configurations intact when every probability
        underflows to zero.
    """

    def __init__(
        self,
        constraints: list[OutcomeConstraint],
        constraint_model: AbstractModel,
        *,
        floor: float = 1e-12,
    ) -> None:
        super().__init__(floor=floor)

        if len(constraints) == 0:
            raise ValueError("A feasibility weight needs at least one constraint.")

        self._constraints = constraints

        # Not to be confused with `self.model`, which is the surrogate model of the objective.
        self._constraint_model = constraint_model

        self._trained = False
        self._has_feasible = False
        self._feasible_eta: float | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return f"Feasibility Weight ({', '.join(str(constraint) for constraint in self._constraints)})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "constraints": [str(constraint) for constraint in self._constraints],
                "constraint_model": self._constraint_model.meta,
            }
        )

        return meta

    @property
    def constraints(self) -> list[OutcomeConstraint]:
        """The enforced constraints."""
        return list(self._constraints)

    @property
    def constraint_model(self) -> AbstractModel:
        """The surrogate model of the constrained outputs."""
        return self._constraint_model

    def _update(self, **kwargs: Any) -> None:
        """Trains the constraint models and determines the best feasible incumbent.

        Parameters
        ----------
        runhistory : RunHistory
            Used to read the observed constraint values.
        """
        assert "runhistory" in kwargs
        runhistory: RunHistory = kwargs["runhistory"]

        configs, constraint_values = self._collect_constraint_data(runhistory)
        self._trained = False

        if len(configs) > 0:
            X = convert_configurations_to_array(configs)
            self._constraint_model.train(X, constraint_values)
            self._trained = True

        feasible_configs = [
            config
            for config, values in zip(configs, self._as_dicts(constraint_values))
            if is_feasible(self._constraints, values)
        ]
        self._has_feasible = len(feasible_configs) > 0
        self._feasible_eta = self._compute_feasible_eta(feasible_configs) if self._has_feasible else None

    def _collect_constraint_data(self, runhistory: RunHistory) -> tuple[list[Configuration], np.ndarray]:
        """Gathers the observed constraint values, averaged over the trials of each configuration.

        Configurations which did not report every constrained output are skipped: there is nothing to train on
        for them. They are infeasible as far as the incumbent is concerned, which ``is_feasible`` decides
        separately.
        """
        names = [constraint.name for constraint in self._constraints]
        observations: dict[Configuration, list[list[float]]] = {}

        for trial_key in runhistory:
            trial_value = runhistory[trial_key]
            values = trial_value.constraint_values
            if values is None:
                continue

            if any(name not in values for name in names):
                continue

            row = [float(values[name]) for name in names]
            if not np.all(np.isfinite(row)):
                continue

            config = runhistory.ids_config[trial_key.config_id]
            observations.setdefault(config, []).append(row)

        configs = list(observations.keys())
        if len(configs) == 0:
            return [], np.empty((0, len(names)))

        averaged = np.array([np.mean(observations[config], axis=0) for config in configs])

        return configs, averaged

    def _as_dicts(self, constraint_values: np.ndarray) -> list[dict[str, float]]:
        names = [constraint.name for constraint in self._constraints]

        return [dict(zip(names, row)) for row in constraint_values]

    def _compute_feasible_eta(self, feasible_configs: list[Configuration]) -> float | None:
        """Returns the best objective value among the feasible configurations.

        The value is predicted rather than observed, matching how the unconstrained incumbent is determined, so
        that it lives in the same space as the objective model's output and can be handed to the acquisition
        function unchanged.
        """
        if self.model is None:
            return None

        X = convert_configurations_to_array(feasible_configs)
        means, _ = self.model.predict_marginalized(X)

        return float(np.min(means[:, 0]))

    def _compute(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        means, variances = self._constraint_model.predict_marginalized(X)

        return probability_of_feasibility(self._constraints, means, variances)

    def is_active(self) -> bool:
        """Whether any constrained output has been observed yet."""
        return self._trained

    def suppresses_acquisition(self) -> bool:
        """Whether the search should look for a feasible configuration before improving on anything."""
        return self._trained and not self._has_feasible

    def adjust_eta(self, eta: float | None) -> float | None:
        """Returns the best feasible incumbent value, or the given one while nothing feasible is known."""
        if self._has_feasible and self._feasible_eta is not None:
            return self._feasible_eta

        return eta
