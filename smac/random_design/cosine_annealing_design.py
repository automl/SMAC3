from __future__ import annotations

from typing import Any

import numpy as np

from smac.random_design.random_design import RandomDesign
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class CosineAnnealingRandomDesign(RandomDesign):
    """Interleave a random configuration according to a given probability which is decreased
    according to a cosine annealing schedule.

    Parameters
    ----------
    max_probability : float
        Initial (maximum) probability of a random configuration
    min_probability : float
        Final (minimal) probability of a random configuration
    restart_iteration : int
        Restart the annealing schedule every ``restart_iteration`` iterations.
    seed : int
        integer used to initialize the random state
    """

    def __init__(
        self,
        min_probability: float,
        max_probability: float,
        restart_iteration: int,
        seed: int = 0,
    ):
        super().__init__(seed)
        assert 0 < min_probability <= 1
        assert 0 < max_probability <= 1
        assert max_probability > min_probability
        assert restart_iteration > 2
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
            "max_probability": self.max_probability,
            "min_probability": self.min_probability,
            "restart_iteration": self.restart_iteration,
        }

    def next_iteration(self) -> None:
        """Set `self.probability` and increases the iteration counter."""
        self.probability = self.min_probability + (
            0.5
            * (self.max_probability - self.min_probability)
            * (1 + np.cos(self.iteration * np.pi / self.restart_iteration))
        )
        logger.error(f"Probability for random configs: {self.probability}")
        self.iteration += 1
        if self.iteration > self.restart_iteration:
            self.iteration = 0
            logger.error("Perform restart in next iteration!")

    def check(self, iteration: int) -> bool:
        """Check if a random configuration should be interleaved. Iteration here relates
        to the ith configuration evaluated in an SMBO iteration."""
        assert iteration > 0
        if self.rng.rand() < self.probability:
            logger.error("Random Config")
            return True
        else:
            logger.error("Acq Config")
            return False
