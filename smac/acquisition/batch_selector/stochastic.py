from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import Configuration

from smac.acquisition.batch_selector.abstract_batch_selector import (
    AbstractBatchSelector,
)
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class StochasticBatchSelector(AbstractBatchSelector):
    r"""Selects a batch by sampling candidates instead of taking the highest scoring ones.

    Sorting by acquisition value and taking the first ``batch_size`` candidates ignores that the
    acquisition value of a candidate would change once its neighbours are part of the batch. If
    that change is modelled as independent Gumbel noise on the scores, the resulting selection rule
    is to sample the batch without replacement with a probability derived from the scores rather
    than to take the maximum. Sampling without replacement proportional to weights $w_i$ is done
    with the Gumbel-top-k trick: perturb $\log w_i$ with independent Gumbel noise and sort.

    Three ways of turning acquisition values into weights are supported:

    * ``soft_rank``: $w_i \propto \mathrm{rank}_i^{-\alpha}$, with rank 1 for the best candidate.
    * ``power``: $w_i \propto (a_i - \min_j a_j)^{\alpha}$.
    * ``softmax``: $w_i \propto \exp((a_i - \bar{a}) / (\tau \cdot \mathrm{std}(a)))$.

    ``soft_rank`` is the default because it only depends on the ordering of the acquisition values.
    SMAC's acquisition functions live on very different scales - a negated cost for confidence
    bounds, a non-negative improvement for expected improvement, a probability for probability of
    improvement - so an exponent or temperature applied to the raw values does not carry over
    between them, while a rank-based weighting needs no retuning.

    The default exponent of 2 keeps the selector's behaviour stable in the size of the candidate
    pool, which is a property of the acquisition maximizer rather than of the problem. Under
    $w_i \propto \mathrm{rank}_i^{-\alpha}$ the expected rank of a draw grows roughly linearly in
    the pool size for $\alpha = 1$ but only logarithmically for $\alpha = 2$: going from 20 to
    5000 candidates moves it from 5.6 to 550 in the first case and from 2.3 to 5.5 in the second.

    This selector needs neither joint posterior samples nor a conditionable model, so it works with
    every surrogate and every acquisition function.

    Note
    ----
    A Gumbel-perturbed ordering of the whole candidate list is simultaneously a
    sampling-without-replacement draw for every batch size, so ``batch_size`` does not enter the
    computation. Pending configurations are removed from the pool; they are already part of the
    batch being formed and are not available to be picked again.

    Parameters
    ----------
    mode : str, defaults to "soft_rank"
        One of "soft_rank", "power" or "softmax".
    alpha : float, defaults to 2.0
        Exponent for "soft_rank" and "power". Larger values concentrate the batch on the best
        candidates; 0 makes the selection uniform.
    temperature : float, defaults to 1.0
        Temperature for "softmax", in units of the standard deviation of the acquisition values.
        Smaller values concentrate the batch on the best candidates.
    seed : int, defaults to 0
        Random seed.
    """

    _modes = ("soft_rank", "power", "softmax")

    def __init__(
        self,
        mode: str = "soft_rank",
        alpha: float = 2.0,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)

        if mode not in self._modes:
            raise ValueError(f"`mode` must be one of {self._modes}, got {mode!r}.")

        if alpha < 0:
            raise ValueError(f"`alpha` must not be negative, got {alpha}.")

        if temperature <= 0:
            raise ValueError(f"`temperature` must be positive, got {temperature}.")

        self._mode = mode
        self._alpha = alpha
        self._temperature = temperature

    @property
    def name(self) -> str:  # noqa: D102
        return f"Stochastic Batch Selection ({self._mode})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "mode": self._mode,
                "alpha": self._alpha,
                "temperature": self._temperature,
            }
        )

        return meta

    def _log_weights(self, acquisition_values: np.ndarray) -> np.ndarray:
        """Turns acquisition values into unnormalized log weights according to ``mode``."""
        if self._mode == "soft_rank":
            ranks = np.empty(len(acquisition_values), dtype=float)
            ranks[np.argsort(-acquisition_values, kind="stable")] = np.arange(
                1, len(acquisition_values) + 1
            )

            return -self._alpha * np.log(ranks)

        if self._mode == "power":
            shifted = acquisition_values - acquisition_values.min()

            return self._alpha * np.log(shifted + np.finfo(float).eps)

        spread = acquisition_values.std()
        if spread == 0:
            return np.zeros_like(acquisition_values)

        return (acquisition_values - acquisition_values.mean()) / (self._temperature * spread)

    def _select(
        self,
        candidates: list[Configuration],
        acquisition_values: np.ndarray,
        batch_size: int,
        model: AbstractModel | None,
        acquisition_function: AbstractAcquisitionFunction | None,
        pending: list[Configuration],
    ) -> list[Configuration]:
        in_flight = set(pending)
        available = [
            (config, value)
            for config, value in zip(candidates, acquisition_values)
            if config not in in_flight
        ]

        if len(available) <= 1:
            return list(candidates)

        available_candidates = [config for config, _ in available]
        available_values = np.array([value for _, value in available])

        log_weights = self._log_weights(available_values)

        # Gumbel-top-k: the ordering of log w + Gumbel noise is a draw without replacement with
        # probability proportional to w.
        uniform = self._rng.uniform(low=np.finfo(float).tiny, high=1.0, size=len(log_weights))
        keys = log_weights - np.log(-np.log(uniform))

        return self._order_by(available_candidates, keys)
