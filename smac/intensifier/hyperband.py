from __future__ import annotations

from typing import Any

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
        assert min_budget is not None and max_budget is not None
        eta = self._eta

        # The only difference we have to do is change max_iterations, n_configs_in_stage, budgets_in_stage
        self._s_max = self._get_max_iterations(eta, max_budget, min_budget)  # type: ignore[operator]
        self._max_iterations: dict[int, int] = {}
        self._n_configs_in_stage: dict[int, list] = {}
        self._budgets_in_stage: dict[int, list] = {}

        for i in range(self._s_max + 1):
            max_iter = self._s_max - i

            self._budgets_in_stage[i], self._n_configs_in_stage[i] = self._compute_configs_and_budgets_for_stages(
                eta, max_budget, max_iter, self._s_max
            )
            self._max_iterations[i] = max_iter + 1

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
