from __future__ import annotations

from smac.acquisition.function.cost_aware_acquisition_function import (
    CostAwareAcquisitionFunction,
)
from smac.callback.callback import Callback
from smac.main.smbo import SMBO

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


class UpdateCostCallback(Callback):
    """Callback to update the cost-aware acquisition function with budget info."""

    def __init__(
        self,
        acquisition_function: CostAwareAcquisitionFunction,
        total_budget: float,
        initial_design_budget: float,
        cumulative_cost_tracker: list[float],
    ):
        self._acquisition_function = acquisition_function
        self._total_budget = total_budget
        self._initial_design_budget = initial_design_budget
        self._cumulative_cost_tracker = cumulative_cost_tracker

    def on_ask_start(self, smbo: SMBO) -> None:
        """
        Called before the intensifier is asked for the next trial.
        This is the ideal place to update the acquisition function with the current
        cumulative cost before it's used to select a configuration.
        """
        self._acquisition_function.set_budget_info(
            total_budget=self._total_budget,
            cumulative_cost=self._cumulative_cost_tracker[0],
            initial_design_budget=self._initial_design_budget,
        )
