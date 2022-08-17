from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import logging

import numpy as np

from smac.random_design.random_design import RandomDesign

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class CosineAnnealingRandomDesign(RandomDesign):
    """Interleave a random configuration according to a given probability which is decreased
    according to a cosine annealing schedule.

    Parameters
    ----------
    max_probability : float
        Initial probility of a random configuration
    min_probability : float
        Lowest probility of a random configuration
    restart_iteration : int
        Restart the annealing schedule every ``restart_iteration`` iterations.
    rng : np.random.RandomState
        Random state
    """

    def __init__(
        self,
        min_probability: float,
        max_probability: float,
        restart_iteration: int,
        seed: int = 0,
    ):
        super().__init__(seed)
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.max_probability = max_probability
        self.min_probability = min_probability
        self.restart_iteration = restart_iteration
        self.iteration = 0
        self.probability = max_probability

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self.seed,
        }

    def next_iteration(self) -> None:
        """Set `self.probability` and increases the iteration counter."""
        self.probability = self.min_probability + (
            0.5
            * (self.max_probability - self.min_probability)
            * (1 + np.cos(self.iteration * np.pi / self.restart_iteration))
        )
        self.logger.error("Probability for random configs: %f" % self.probability)
        self.iteration += 1
        if self.iteration > self.restart_iteration:
            self.iteration = 0
            self.logger.error("Perform restart in next iteration!")

    def check(self, iteration: int) -> bool:
        """Check if a random configuration should be interleaved."""
        if self.rng.rand() < self.probability:
            self.logger.error("Random Config")
            return True
        else:
            self.logger.error("Acq Config")
            return False
