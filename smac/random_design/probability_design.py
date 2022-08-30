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
    prob : float
        Probability that a configuration will be drawn at random
    seed : int
        Integer used to initialize random state
    """

    def __init__(self, probability: float, seed: int = 0):
        super().__init__(seed)
        assert 0 <= probability <= 1
        self.prob = probability

    def next_iteration(self) -> None:
        """Does nothing."""
        ...

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "probability": self.prob,
            "seed": self.seed,
        }

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random. Iteration here relates
        to the ith configuration evaluated in an SMBO iteration."""
        assert iteration >= 0
        if self.rng.rand() < self.prob:
            return True
        else:
            return False


class ProbabilityCoolDownRandomDesign(AbstractRandomDesign):
    """Interleave a random configuration according to a given probability which is decreased over
    time.

    Parameters
    ----------
    prob : float
        Probility of a random configuration
    factor : float
        Multiply the ``prob`` by ``cool_down_fac`` in each iteration
    seed : int
        Integer used to initialize random state
    """

    def __init__(self, probability: float, factor: float, seed: int = 0):
        super().__init__(seed)
        assert 0 <= probability <= 1
        assert factor > 0
        self.prob = probability
        self.cool_down_fac = factor

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "probability": self.prob,
            "factor": self.cool_down_fac,
            "seed": self.seed,
        }

    def next_iteration(self) -> None:
        """Set the probability to the current value multiplied by the `cool_down_fac`."""
        self.prob *= self.cool_down_fac

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random. Iteration here relates
        to the ith configuration evaluated in an SMBO iteration."""
        assert iteration >= 0
        if self.rng.rand() <= self.prob:
            return True
        else:
            return False
