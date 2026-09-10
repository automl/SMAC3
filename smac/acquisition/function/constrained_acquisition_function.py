from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.confidence_bound import AbstractConfidenceBound
from smac.acquisition.function.integrated_acquisition_function import (
    IntegratedAcquisitionFunction,
)
from smac.acquisition.function.thompson import TS
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


class ConstrainedAcquisitionFunction(AbstractAcquisitionFunction):
    r"""Weight an acquisition function by the probability that the output constraints are satisfied.

    The constrained outputs are modelled separately from the objective, and the acquisition value is multiplied
    by the probability that every bound holds:

    $$
    a_{c}(\mathbf{X}) = a(\mathbf{X}) \prod_i P(y_i(\mathbf{X}) \text{ satisfies constraint } i)
    $$

    See "Bayesian Optimization with Inequality Constraints" by Jacob Gardner et al. [[GKZ+14][GKZ+14]] for
    further details.

    Weighting rather than penalizing keeps the objective model clean: a penalty would make the surrogate fit a
    cliff that is not a feature of the objective, degrading its predictions inside the feasible region as well,
    and would discard the measured value that locates the boundary.

    Two details make the difference between this working and not:

    * The incumbent handed to the wrapped acquisition function is the best *feasible* one. The unconstrained
      incumbent is over-optimistic, because the best configuration seen so far may well violate a bound.
    * While nothing feasible has been observed there is no improvement to expect over, so the acquisition
      function degenerates to the probability of feasibility alone until the first feasible configuration
      turns up.

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
        super().__init__()

        if len(constraints) == 0:
            raise ValueError("A constrained acquisition function needs at least one constraint.")

        self._acquisition_function = acquisition_function
        self._constraints = constraints
        self._constraint_model = constraint_model
        self._feasibility_floor = feasibility_floor

        # LCB and TS are negative by design, so they have to be shifted before a multiplicative weight is
        # meaningful. This mirrors how BoTorch shifts by an infeasible cost before applying its feasibility
        # weight.
        if isinstance(acquisition_function, IntegratedAcquisitionFunction):
            acquisition_type = acquisition_function._acquisition_function
        else:
            acquisition_type = acquisition_function

        self._rescale = isinstance(acquisition_type, (AbstractConfidenceBound, TS))

        self._eta: float | None = None
        self._has_feasible = False
        self._trained = False

    @property
    def name(self) -> str:  # noqa: D102
        return f"Constrained Acquisition Function ({self._acquisition_function.__class__.__name__})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "acquisition_function": self._acquisition_function.meta,
                "constraints": [str(constraint) for constraint in self._constraints],
                "constraint_model": self._constraint_model.meta,
                "feasibility_floor": self._feasibility_floor,
            }
        )

        return meta

    def _update(self, **kwargs: Any) -> None:
        """Trains the constraint models and replaces the incumbent with the best feasible one.

        Parameters
        ----------
        runhistory : RunHistory
            Used to read the observed constraint values.
        eta : float
            Current incumbent value, ignoring feasibility.
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

        kwargs = dict(kwargs)
        if self._has_feasible:
            kwargs["eta"] = self._compute_feasible_eta(feasible_configs, kwargs.get("eta"))

        self._eta = kwargs.get("eta")

        assert self.model is not None
        self._acquisition_function.update(model=self.model, **kwargs)

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

    def _compute_feasible_eta(self, feasible_configs: list[Configuration], eta: float | None) -> float | None:
        """Returns the best objective value among the feasible configurations.

        The value is predicted rather than observed, matching how the unconstrained incumbent is determined, so
        that it lives in the same space as the objective model's output and can be handed to the wrapped
        acquisition function unchanged.
        """
        if self.model is None:
            return eta

        X = convert_configurations_to_array(feasible_configs)
        means, _ = self.model.predict_marginalized(X)

        return float(np.min(means[:, 0]))

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the feasibility-weighted acquisition values.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated.

        Returns
        -------
        np.ndarray [N, 1]
            Feasibility-weighted acquisition values of X.
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if not self._trained:
            # Nothing has reported a constrained output yet, so there is no feasibility to weight by.
            return self._acquisition_function._compute(X)

        means, variances = self._constraint_model.predict_marginalized(X)
        feasibility = probability_of_feasibility(self._constraints, means, variances)

        if not self._has_feasible:
            # Without a feasible incumbent there is no improvement to expect over, so search for a feasible
            # configuration first.
            return feasibility

        if self._rescale:
            assert self._eta is not None
            acquisition_values = np.clip(self._acquisition_function._compute(X) + self._eta, 0, np.inf)
        else:
            acquisition_values = self._acquisition_function._compute(X)

        return acquisition_values.reshape((-1, 1)) * (feasibility + self._feasibility_floor)
