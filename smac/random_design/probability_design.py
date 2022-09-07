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
    """

    def __init__(self, probability: float, seed: int = 0):
        super().__init__(seed=seed)
        assert 0 <= probability <= 1
        self._probability = probability

    def get_meta(self) -> dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "probability": self._probability,
            "seed": self._seed,
        }

    def check(self, iteration: int) -> bool:
        assert iteration >= 0

        if self._rng.rand() < self._probability:
            return True
        else:
            return False


class ProbabilityCoolDownRandomDesign(AbstractRandomDesign):
    """Interleave a random configuration according to a given probability which is decreased over
    time.

    Parameters
    ----------
    probability : float
        Probability that a configuration will be drawn at random.
    factor : float
        Multiply the `probability` by `factor` in each iteration.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        probability: float,
        factor: float,
        seed: int = 0,
    ):
        super().__init__(seed)
        assert 0 <= probability <= 1
        assert factor > 0

        self._probability = probability
        self._factor = factor

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "probability": self._probability,
            "factor": self._factor,
            "seed": self._seed,
        }

    def next_iteration(self) -> None:
        """Sets the probability to the current value multiplied by `factor`."""
        self._probability *= self._factor

    def check(self, iteration: int) -> bool:
        assert iteration >= 0

        if self._rng.rand() <= self._probability:
            return True
        else:
            return False
