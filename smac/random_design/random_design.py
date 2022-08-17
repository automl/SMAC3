from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import logging

import numpy as np

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class RandomDesign:
    """Abstract base of helper classes to configure interleaving of random configurations in a list
    of challengers.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self.seed,
        }

    @abstractmethod
    def next_iteration(self) -> None:
        """Indicate beginning of next SMBO iteration."""
        pass

    @abstractmethod
    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        pass
