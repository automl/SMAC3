from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import logging

import numpy as np

from smac.random_design.random_design import RandomDesign

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class ProbabilityRandomDesign(RandomDesign):
    """Interleave a random configuration according to a given probability.

    Parameters
    ----------
    prob : float
        Probility of a random configuration
    rng : np.random.RandomState
        Random state
    """

    def __init__(self, probability: float, seed: int = 0):
        super().__init__(seed)
        self.prob = probability

    def next_iteration(self) -> None:
        """Does nothing."""
        ...

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        if self.rng.rand() < self.prob:
            return True
        else:
            return False


class ProbabilityCoolDownRandomDesign(RandomDesign):
    """Interleave a random configuration according to a given probability which is decreased over
    time.

    Parameters
    ----------
    prob : float
        Probility of a random configuration
    cool_down_fac : float
        Multiply the ``prob`` by ``cool_down_fac`` in each iteration
    rng : np.random.RandomState
        Random state
    """

    def __init__(self, probability: float, factor: float, seed: int = 0):
        super().__init__(seed)
        self.prob = probability
        self.cool_down_fac = factor

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def next_iteration(self) -> None:
        """Set the probability to the current value multiplied by the `cool_down_fac`."""
        self.prob *= self.cool_down_fac

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        if self.rng.rand() < self.prob:
            return True
        else:
            return False
