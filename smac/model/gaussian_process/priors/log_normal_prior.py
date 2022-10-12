from __future__ import annotations

from typing import Any

import math

import numpy as np

from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class LogNormalPrior(AbstractPrior):
    """Implements the log normal prior.

    Parameters
    ----------
    sigma : float
        Specifies the standard deviation of the normal distribution.
    mean : float
        Specifies the mean of the normal distribution.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        sigma: float,
        mean: float = 0,
        seed: int = 0,
    ):
        super().__init__(seed=seed)

        if mean != 0:
            raise NotImplementedError(mean)

        self._sigma = sigma
        self._sigma_square = sigma**2
        self._mean = mean
        self._sqrt_2_pi = np.sqrt(2 * np.pi)

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"sigma": self._sigma, "mean": self._mean})

        return meta

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        return self._rng.lognormal(mean=self._mean, sigma=self._sigma, size=n_samples)

    def _get_log_probability(self, theta: float) -> float:
        if theta <= self._mean:
            return -1e25
        else:
            rval = -((math.log(theta) - self._mean) ** 2) / (2 * self._sigma_square) - math.log(
                self._sqrt_2_pi * self._sigma * theta
            )
            return rval

    def _get_gradient(self, theta: float) -> float:
        if theta <= 0:
            return 0
        else:
            # Derivative of log(1 / (x * s^2 * sqrt(2 pi)) * exp( - 0.5 * (log(x ) / s^2))^2))
            # This is without the mean!
            return -(self._sigma_square + math.log(theta)) / (self._sigma_square * (theta)) * theta
