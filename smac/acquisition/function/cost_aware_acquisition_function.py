from __future__ import annotations

from typing import Any

import numpy as np

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.callback.cost_surrogate_callback import CostSurrogateCallback
from smac.model.abstract_model import AbstractModel


class CostAwareAcquisitionFunction(AbstractAcquisitionFunction):
    r"""
    A wrapper to make any acquisition function cost-aware, using the EI-Cool strategy.
    The cost-awareness is controlled by a dynamic parameter alpha, which decreases as the
    optimization budget is spent.

    \[
    \mathrm{Acq}(x)_{\text{cost-aware}} := \frac{\mathrm{Acq}(x)}{c(x)^\alpha}
    where \alpha = \frac{\tau - \tau_k}{\tau - \tau_{\mathrm{init}}},
    \]

    where:
    - :math:`\mathrm{Acq}(x)` is the value of the wrapped acquisition function.
    - :math:`c(x)` is the predicted cost of configuration :math:`x`.
    - :math:`\tau` is the total optimization budget.
    - :math:`\tau_k` is the budget spent up to iteration :math:`k`.
    - :math:`\tau_{\mathrm{init}}` is the budget spent on the initial design.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction
        The acquisition function to be wrapped.
    cost_surrogate_callback : CostSurrogateCallback
        Callback to get the cost model.
    """

    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        cost_surrogate_callback: CostSurrogateCallback,
    ):
        super().__init__()
        self._acquisition_function = acquisition_function
        self._cost_callback = cost_surrogate_callback
        self._alpha: float = 1.0
        # Add attributes to store budget info
        self._total_budget: float | None = None
        self._cumulative_cost: float | None = None
        self._initial_design_budget: float | None = None

    @property
    def name(self) -> str:
        """Returns the full name of the acquisition function."""
        return f"{self._acquisition_function.name} (Cost-Aware)"

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the acquisition function."""
        # Start with the meta from the wrapped acquisition function
        meta = self._acquisition_function.meta.copy()
        # Update with cost-aware specific information
        meta.update({"name": self.name})
        meta.update({"alpha": self._alpha})
        return meta

    def set_budget_info(
        self,
        total_budget: float,
        cumulative_cost: float,
        initial_design_budget: float,
    ) -> None:
        """
        Sets the budget information required for alpha calculation.
        This method is called externally before `ask` is called on the facade.
        """
        self._total_budget = total_budget
        self._cumulative_cost = cumulative_cost
        self._initial_design_budget = initial_design_budget

    def update(self, model: AbstractModel, **kwargs: Any) -> None:
        """
        Update the acquisition function with new budget information and pass updates
        to the wrapped acquisition function.
        """
        if self._total_budget is None or self._cumulative_cost is None or self._initial_design_budget is None:
            raise ValueError("Budget information not set. Call `set_budget_info` before running optimization.")

        denominator = self._total_budget - self._initial_design_budget
        if denominator <= 0:
            # Avoid division by zero or negative alpha
            self._alpha = 0.0
        else:
            self._alpha = max(0.0, (self._total_budget - self._cumulative_cost) / denominator)

        # Update the wrapped acquisition function
        self._acquisition_function.update(model, **kwargs)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the cost-aware acquisition values."""
        if self._cost_callback.cost_model is None:
            raise RuntimeError("Cost model not initialized.")

        # Get standard acquisition values by calling the wrapped function's internal compute
        acq_values = self._acquisition_function._compute(X)

        # If alpha is 0, no need to compute cost
        if self._alpha == 0.0:
            return acq_values

        # Get predicted costs
        cost_values, _ = self._cost_callback.cost_model.predict(X)
        cost_values = np.maximum(cost_values, 1e-9)  # Avoid division by zero

        # Return acq_value / (cost^alpha)
        return acq_values / (cost_values**self._alpha)
