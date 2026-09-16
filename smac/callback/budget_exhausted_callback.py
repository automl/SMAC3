from __future__ import annotations

from smac.callback.callback import Callback
from smac.main.smbo import SMBO
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class BudgetExhaustedCallback(Callback):
    """Callback to stop the optimization if the budget is exhausted."""

    def __init__(self, total_resource_budget: float, cumulative_cost_tracker: list[float]):
        self._total_resource_budget = total_resource_budget
        self._cumulative_cost_tracker = cumulative_cost_tracker

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """
        Called after the stats are updated and the trial is added to the runhistory.
        Optionally, returns false to gracefully stop the optimization.
        """
        # The resource cost is passed in additional_info
        resource_cost = value.additional_info.get("resource_cost", 0.0)
        self._cumulative_cost_tracker[0] += resource_cost

        logger.info(
            f"Origin: {info.config.origin}, Cost: {resource_cost: .2f}, "
            f"Cumulative Cost: {self._cumulative_cost_tracker[0]: .2f}/{self._total_resource_budget: .2f}"
        )

        if self._cumulative_cost_tracker[0] >= self._total_resource_budget:
            logger.info("Total resource budget exhausted. Stopping optimization.")
            # Mark optimization as finished before stopping
            smbo._finished = True
            return False

        return None
