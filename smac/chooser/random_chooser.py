from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import logging

import numpy as np

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class RandomConfigurationChooser(ABC):
    """Abstract base of helper classes to configure interleaving of random configurations in a list
    of challengers.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed=seed)

    @abstractmethod
    def next_smbo_iteration(self) -> None:
        """Indicate beginning of next SMBO iteration."""
        pass

    @abstractmethod
    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be at random."""
        pass
