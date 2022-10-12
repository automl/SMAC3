from __future__ import annotations

from typing import Any

import math

import numpy as np

from smac.constants import VERY_SMALL_NUMBER
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HorseshoePrior(AbstractPrior):
    """Horseshoe Prior as it is used in spearmint.

    Parameters
    ----------
    scale: float
        Scaling parameter.
    seed : int, defaults to 0
    """

    def __init__(self, scale: float, seed: int = 0):
        super().__init__(seed=seed)
        self._scale = scale
        self._scale_square = scale**2

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"scale": self._scale})

        return meta

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        # This is copied from RoBO - scale is most likely the tau parameter
        lamda = np.abs(self._rng.standard_cauchy(size=n_samples))
        p0 = np.abs(self._rng.randn() * lamda * self._scale)
        return p0

    def _get_log_probability(self, theta: float) -> float:
        # We computed it exactly as in the original spearmint code, they basically say that there's no analytical form
        # of the horseshoe prior, but that the multiplier is bounded between 2 and 4 and that they used the middle
        # See "The horseshoe estimator for sparse signals" by Carvalho, Poloson and Scott (2010), Equation 1.
        # https://www.jstor.org/stable/25734098
        # Compared to the paper by Carvalho, there's a constant multiplicator missing
        # Compared to Spearmint we first have to undo the log space transformation of the theta
        # Note: "undo log space transformation" is done in parent class
        if theta == 0:
            return np.inf  # POSITIVE infinity (this is the "spike")
        else:
            a = math.log(1 + 3.0 * (self._scale_square / theta**2))
            return math.log(a + VERY_SMALL_NUMBER)

    def _get_gradient(self, theta: float) -> float:
        if theta == 0:
            return np.inf  # POSITIVE infinity (this is the "spike")
        else:
            a = -(6 * self._scale_square)
            b = 3 * self._scale_square + theta**2
            b *= math.log(3 * self._scale_square * theta ** (-2) + 1)
            b = max(b, 1e-14)

            return a / b
