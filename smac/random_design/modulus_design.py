from __future__ import annotations

from typing import Any

import logging

import numpy as np

from smac.random_design.random_design import RandomDesign
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class NoCoolDownRandomDesign(RandomDesign):
    """Interleave a random configuration after a constant number of configurations found by
    Bayesian optimization.

    Parameters
    ----------
    modulus : float
        Every modulus-th configuration will be at random.
    seed: int
        integer used to initialize random state (not used)
    """

    def __init__(self, modulus: float = 2.0, seed: int = 0):
        super().__init__(seed)

        logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        if modulus <= 1.0:
            logger.warning("Using SMAC with random configurations only. ROAR is the better choice for this.")
        self.modulus = modulus

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "modulus": self.modulus,
            "seed": self.seed,
        }

    def next_iteration(self) -> None:
        """Does nothing."""
        ...

    def check(self, iteration: int) -> bool:
        """Checks if the next configuration should be at random."""
        return iteration % self.modulus < 1


class LinearCoolDownRandomDesign(RandomDesign):
    """Interleave a random configuration, decreasing the fraction of random configurations over
    time.

    Parameters
    ----------
    start_modulus : float
       Initially, every modulus-th configuration will be at random
    modulus_increment : float
       Increase modulus by this amount in every iteration
    end_modulus : float
       The maximum modulus ever used in the chooser. If the value is reached before the optimization
       is over, it is not further increased. If it is not reached before the optimization is over,
       there will be no adjustment to make sure that the ``end_modulus`` is reached.
    seed : int
        Integer used to initialize the random state (not used)
    """

    def __init__(
        self,
        start_modulus: float = 2.0,
        modulus_increment: float = 0.3,
        end_modulus: float = np.inf,
        seed: int = 0,
    ):
        super().__init__(seed)

        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        if start_modulus <= 1.0 and modulus_increment <= 0.0:
            logger.warning("Using SMAC with random configurations only. ROAR is the better choice for this.")
        self.modulus = start_modulus
        self.start_modulus = start_modulus
        self.modulus_increment = modulus_increment
        self.end_modulus = end_modulus
        self.last_iteration = 0

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "start_modulus": self.start_modulus,
            "end_modulus": self.end_modulus,
            "modulus_increment": self.modulus_increment,
            "seed": self.seed,
        }

    def next_iteration(self) -> None:
        """Increase modulus."""
        self.modulus += self.modulus_increment
        self.modulus = min(self.modulus, self.end_modulus)
        self.last_iteration = 0

    def check(self, iteration: int) -> bool:
        """Check if the next configuration should be interleaved based on modulus."""
        if (iteration - self.last_iteration) % self.modulus < 1:
            self.last_iteration = iteration
            return True
        else:
            return False
