from __future__ import annotations

from typing import Any

import numpy as np

from smac.intensifier.successive_halving import SuccessiveHalving


class Hyperband(SuccessiveHalving):
    """See ``SuccessiveHalving`` for documentation."""

    def reset(self) -> None:
        """Resets the internal variables of the intensifier, including the tracker and the next bracket."""
        super().reset()

        # Reset current bracket
        self._next_bracket: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()

        min_budget = self._min_budget
        max_budget = self._max_budget
        eta = self._eta

        # The only difference we have to do is change max_iterations, n_configs_in_stage, budgets_in_stage
        s_max = int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))

        max_iterations: dict[int, int] = {}
        n_configs_in_stage: dict[int, list] = {}
        budgets_in_stage: dict[int, list] = {}

        for i in range(s_max + 1):
            max_iter = s_max - i
            n_initial_challengers = int(eta**max_iter)

            # How many configs in each stage
            linspace = -np.linspace(0, max_iter, max_iter + 1)
            n_configs_ = n_initial_challengers * np.power(eta, linspace)
            n_configs = np.array(np.round(n_configs_), dtype=int).tolist()

            # How many budgets in each stage
            linspace = -np.linspace(max_iter, 0, max_iter + 1)
            budgets = (max_budget * np.power(eta, linspace)).tolist()

            max_iterations[i] = max_iter + 1
            n_configs_in_stage[i] = n_configs
            budgets_in_stage[i] = budgets

        self._s_max = s_max
        self._max_iterations = max_iterations
        self._n_configs_in_stage = n_configs_in_stage
        self._budgets_in_stage = budgets_in_stage

    def get_state(self) -> dict[str, Any]:  # noqa: D102
        state = super().get_state()
        state["next_bracket"] = self._next_bracket

        return state

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: D102
        super().set_state(state)
        self._next_bracket = state["next_bracket"]

    def _get_next_bracket(self) -> int:
        """In contrast to Successive Halving, Hyperband uses multiple brackets. Each time a new batch
        is added to the tracker, the bracket is increased.
        """
        current_bracket = self._next_bracket
        next_bracket = current_bracket + 1

        if next_bracket > self._s_max or next_bracket < 0:
            next_bracket = 0

        self._next_bracket = next_bracket

        return current_bracket
