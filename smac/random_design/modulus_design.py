from __future__ import annotations

from typing import Any

import numpy as np

from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class ModulusRandomDesign(AbstractRandomDesign):
    """Interleave a random configuration after a constant number of configurations found by
    Bayesian optimization.

    Parameters
    ----------
    modulus : float
        Every modulus-th configuration will be at random.
    seed : int
        Integer used to initialize random state. This class does not use the seed.
    """

    def __init__(self, modulus: float = 2.0, seed: int = 0):
        super().__init__(seed)
        assert modulus > 0
        if modulus <= 1.0:
            logger.warning("Using SMAC with random configurations only. ROAR is the better choice for this.")

        self._modulus = modulus

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"modulus": self._modulus})

        return meta

    def check(self, iteration: int) -> bool:  # noqa: D102
        assert iteration >= 0
        return iteration % self._modulus < 1


class DynamicModulusRandomDesign(AbstractRandomDesign):
    """Interleave a random configuration, decreasing the fraction of random configurations over time.

    Parameters
    ----------
    start_modulus : float, defaults to 2.0
       Initially, every modulus-th configuration will be at random.
    modulus_increment : float, defaults to 0.3
       Increase modulus by this amount in every iteration.
    end_modulus : float, defaults to np.inf
       The maximum modulus ever used. If the value is reached before the optimization
       is over, it is not further increased. If it is not reached before the optimization is over,
       there will be no adjustment to make sure that the `end_modulus` is reached.
    seed : int, defaults to 0
        Integer used to initialize the random state. This class does not use the seed.
    """

    def __init__(
        self, start_modulus: float = 2.0, modulus_increment: float = 0.3, end_modulus: float = np.inf, seed: int = 0
    ):
        super().__init__(seed)
        assert start_modulus > 0
        assert modulus_increment > 0
        assert end_modulus > 0
        assert end_modulus > start_modulus

        if start_modulus <= 1.0 and modulus_increment <= 0.0:
            logger.warning("Using SMAC with random configurations only. ROAR is the better choice for this.")

        self._modulus = start_modulus
        self._start_modulus = start_modulus
        self._modulus_increment = modulus_increment
        self._end_modulus = end_modulus

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "start_modulus": self._start_modulus,
                "end_modulus": self._end_modulus,
                "modulus_increment": self._modulus_increment,
            }
        )

        return meta

    def next_iteration(self) -> None:  # noqa: D102
        self._modulus += self._modulus_increment
        self._modulus = min(self._modulus, self._end_modulus)

    def check(self, iteration: int) -> bool:  # noqa: D102
        assert iteration >= 0

        if iteration % self._modulus < 1:
            return True
        else:
            return False
