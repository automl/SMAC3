from __future__ import annotations

from typing import Any

import numpy as np

from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class CosineAnnealingRandomDesign(AbstractRandomDesign):
    """Interleaves a random configuration according to a given probability which is decreased
    according to a cosine annealing schedule.

    Parameters
    ----------
    max_probability : float
        Initial (maximum) probability of a random configuration.
    min_probability : float
        Final (minimal) probability of a random configuration used in iteration `restart_iteration`.
    restart_iteration : int
        Restart the annealing schedule every `restart_iteration` iterations.
    seed : int
        Integer used to initialize random state.
    """

    def __init__(self, min_probability: float, max_probability: float, restart_iteration: int, seed: int = 0):
        super().__init__(seed)
        assert 0 <= min_probability <= 1
        assert 0 <= max_probability <= 1
        assert max_probability > min_probability
        assert restart_iteration > 2
        self._max_probability = max_probability
        self._min_probability = min_probability

        # Internally, iteration indices start at 0, so we need to decrease this
        self._restart_iteration = restart_iteration - 1
        self._iteration = 0
        self._probability = max_probability

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "max_probability": self._max_probability,
                "min_probability": self._min_probability,
                "restart_iteration": self._restart_iteration,
            }
        )

        return meta

    def next_iteration(self) -> None:  # noqa: D102
        """Moves to the next iteration and set ``self._probability``."""
        self._iteration += 1
        if self._iteration > self._restart_iteration:
            self._iteration = 0
            logger.debug("Perform a restart.")

        self._probability = self._min_probability + (
            0.5
            * (self._max_probability - self._min_probability)
            * (1 + np.cos(self._iteration * np.pi / self._restart_iteration))
        )
        logger.debug(f"Probability for random configs: {self._probability}")

    def check(self, iteration: int) -> bool:  # noqa: D102
        assert iteration >= 0

        if self._rng.rand() <= self._probability:
            return True
        else:
            return False
