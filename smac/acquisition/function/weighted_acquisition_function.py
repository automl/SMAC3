from __future__ import annotations

from typing import Any, Sequence, TypeVar

import numpy as np

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.weight.abstract_weight import AbstractAcquisitionWeight
from smac.acquisition.weight.feasibility import FeasibilityWeight
from smac.model.abstract_model import AbstractModel
from smac.utils.constraints import OutcomeConstraint
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

W = TypeVar("W", bound=AbstractAcquisitionWeight)


class WeightedAcquisitionFunction(AbstractAcquisitionFunction):
    r"""An acquisition function multiplied by a list of non-negative weights.

    $$a_w(\mathbf{X}) = a(\mathbf{X}) \prod_i w_i(\mathbf{X})$$

    This is how SMAC steers the search without touching the surrogate model of the objective: output constraints
    contribute the probability that their bounds hold, user priors over the optimum contribute a density, and both
    only reweight how interesting each configuration looks.

    Holding the weights in a list rather than nesting one wrapper inside another is what makes them compose.
    Nested wrappers each shift the acquisition values of a confidence bound by the incumbent, so the shift is
    applied twice and the values no longer mean anything. Here it happens in exactly one place.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction
        The acquisition function to weight.
    weights : Sequence[AbstractAcquisitionWeight] | None, defaults to None
        The weights to apply. Weights can be added and removed later, which is how a user prior supplied during a
        run reaches the acquisition function.
    """

    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        weights: Sequence[AbstractAcquisitionWeight] | None = None,
    ) -> None:
        super().__init__()

        if isinstance(acquisition_function, WeightedAcquisitionFunction):
            raise ValueError(
                "A weighted acquisition function must not wrap another one: both would shift the acquisition "
                "values of a confidence bound by the incumbent, and the ranking would be meaningless. Add the "
                "weights to the existing wrapper with add_weight instead."
            )

        self._acquisition_function = acquisition_function
        self._weights: list[AbstractAcquisitionWeight] = list(weights) if weights is not None else []
        self._rescale = acquisition_function.requires_rescaling
        self._eta: float | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return f"Weighted Acquisition Function ({self._acquisition_function.__class__.__name__})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "acquisition_function": self._acquisition_function.meta,
                "weights": [weight.meta for weight in self._weights],
            }
        )

        return meta

    @property
    def requires_rescaling(self) -> bool:  # noqa: D102
        # The weighted values are non-negative, whether or not the wrapped function needed shifting.
        return False

    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction:
        """The wrapped acquisition function."""
        return self._acquisition_function

    @property
    def weights(self) -> tuple[AbstractAcquisitionWeight, ...]:
        """The applied weights, in the order they are multiplied."""
        return tuple(self._weights)

    @property
    def model(self) -> AbstractModel | None:  # noqa: D102
        return self._model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        self._model = model

        for weight in self._weights:
            weight.model = model

    def add_weight(self, weight: AbstractAcquisitionWeight) -> None:
        """Adds a weight, which takes effect at the next acquisition function maximization."""
        if self._model is not None:
            weight.model = self._model

        self._weights.append(weight)

    def remove_weight(self, weight: AbstractAcquisitionWeight) -> None:
        """Removes a previously added weight."""
        self._weights.remove(weight)

    def get_weight(self, kind: type[W]) -> W | None:
        """Returns the first weight of the given type, or `None` if there is none.

        Used to find the one weight which collects the user priors, or the one which enforces the output
        constraints, without the caller having to remember where in the list it was put.
        """
        for weight in self._weights:
            if isinstance(weight, kind):
                return weight

        return None

    def _update(self, **kwargs: Any) -> None:
        """Updates the weights, then the wrapped acquisition function.

        The weights go first so that they can correct the incumbent value before the acquisition function is told
        what to improve over.
        """
        assert self.model is not None

        for weight in self._weights:
            weight.update(model=self.model, **kwargs)

        kwargs = dict(kwargs)
        for weight in self._weights:
            kwargs["eta"] = weight.adjust_eta(kwargs.get("eta"))

        self._eta = kwargs.get("eta")

        self._acquisition_function.update(model=self.model, **kwargs)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the weighted acquisition values.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated.

        Returns
        -------
        np.ndarray [N, 1]
            Weighted acquisition values of X.
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        active = [weight for weight in self._weights if weight.is_active()]

        if len(active) == 0:
            # Nothing to weight by, so hand the acquisition values through untouched. Deliberately without the
            # rescaling, which clips at zero and would change the ranking for no reason.
            return self._acquisition_function._compute(X)

        if any(weight.suppresses_acquisition() for weight in active):
            # The acquisition function has nothing to say, so let the weights rank the candidates on their own.
            values = np.ones((X.shape[0], 1))
        elif self._rescale:
            assert self._eta is not None
            values = np.clip(self._acquisition_function._compute(X).reshape((-1, 1)) + self._eta, 0, np.inf)
        else:
            values = self._acquisition_function._compute(X).reshape((-1, 1))

        for weight in active:
            values = values * weight(X)

        return values


def ensure_feasibility_weight(
    acquisition_function: AbstractAcquisitionFunction,
    constraints: list[OutcomeConstraint],
    constraint_model: AbstractModel,
    feasibility_floor: float = 1e-12,
) -> AbstractAcquisitionFunction:
    """Makes sure the given acquisition function is weighted by the probability that the constraints hold.

    An acquisition function which already carries weights - a user prior over the optimum, say - gains one more,
    so that the two mechanisms end up side by side in one wrapper. A plain acquisition function is wrapped in a
    `ConstrainedAcquisitionFunction`, which is what the constraints-only case has always produced and what keeps
    its metadata, and therefore the name of its output directory, unchanged.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction
        The acquisition function to weight.
    constraints : list[OutcomeConstraint]
        The constraints to enforce.
    constraint_model : AbstractModel
        Surrogate model for the constrained outputs.
    feasibility_floor : float, defaults to 1e-12
        Lowest possible value of the feasibility weight.

    Returns
    -------
    AbstractAcquisitionFunction
        The weighted acquisition function.
    """
    from smac.acquisition.function.constrained_acquisition_function import (
        ConstrainedAcquisitionFunction,
    )

    if isinstance(acquisition_function, WeightedAcquisitionFunction):
        if acquisition_function.get_weight(FeasibilityWeight) is not None:
            return acquisition_function

        acquisition_function.add_weight(
            FeasibilityWeight(
                constraints=constraints,
                constraint_model=constraint_model,
                floor=feasibility_floor,
            )
        )

        return acquisition_function

    return ConstrainedAcquisitionFunction(
        acquisition_function=acquisition_function,
        constraints=constraints,
        constraint_model=constraint_model,
        feasibility_floor=feasibility_floor,
    )
