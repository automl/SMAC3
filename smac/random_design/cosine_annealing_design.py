from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
    prob_max : float
        Initial probility of a random configuration
    prob_min : float
        Lowest probility of a random configuration
    restart_iteration : int
        Restart the annealing schedule every ``restart_iteration`` iterations.
    rng : np.random.RandomState
        Random state
    """

    def __init__(
        self,
        prob_max: float,
        prob_min: float,
        restart_iteration: int,
        seed: int = 0,
    ):
        super().__init__(seed)
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.prob_max = prob_max
        self.prob_min = prob_min
        self.restart_iteration = restart_iteration
        self.iteration = 0
        self.prob = prob_max

    def next_smbo_iteration(self) -> None:
        """Set `self.prob` and increases the iteration counter."""
        self.prob = self.prob_min + (
            0.5 * (self.prob_max - self.prob_min) * (1 + np.cos(self.iteration * np.pi / self.restart_iteration))
        )
        self.logger.error("Probability for random configs: %f" % self.prob)
        self.iteration += 1
        if self.iteration > self.restart_iteration:
            self.iteration = 0
            self.logger.error("Perform restart in next iteration!")

    def check(self, iteration: int) -> bool:
        """Check if a random configuration should be interleaved."""
        if self.rng.rand() < self.prob:
            self.logger.error("Random Config")
            return True
        else:
            self.logger.error("Acq Config")
            return False
