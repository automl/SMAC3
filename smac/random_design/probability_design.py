from __future__ import annotations

from typing import Any

from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class ProbabilityRandomDesign(AbstractRandomDesign):
    """Interleave a random configuration according to a given probability.

    Parameters
    ----------
    probability : float
        Probability that a configuration will be drawn at random.
    seed : int, defaults to 0
        Integer used to initialize the random state.
    """

    def __init__(self, probability: float, seed: int = 0):
        super().__init__(seed=seed)
        assert 0 <= probability <= 1
        self._probability = probability

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"probability": self._probability})

        return meta

    def check(self, iteration: int) -> bool:  # noqa: D102
        assert iteration >= 0

        if self._rng.rand() < self._probability:
            return True
        else:
            return False


class DynamicProbabilityRandomDesign(AbstractRandomDesign):
    """Interleave a random configuration according to a given probability which is decreased over time.

    Parameters
    ----------
    probability : float
        Probability that a configuration will be drawn at random.
    factor : float
        Multiply the `probability` by `factor` in each iteration.
    seed : int, defaults to 0
        Integer used to initialize the random state.
    """

    def __init__(self, probability: float, factor: float, seed: int = 0):
        super().__init__(seed)
        assert 0 <= probability <= 1
        assert factor > 0

        self._probability = probability
        self._factor = factor

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"probability": self._probability, "factor": self._factor})

        return meta

    def next_iteration(self) -> None:
        """Sets the probability to the current value multiplied by ``factor``."""
        self._probability *= self._factor

    def check(self, iteration: int) -> bool:  # noqa: D102
        assert iteration >= 0

        if self._rng.rand() <= self._probability:
            return True
        else:
            return False
