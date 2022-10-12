from __future__ import annotations

from typing import Any

import numpy as np
import scipy.stats as sps

from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class GammaPrior(AbstractPrior):
    """Implementation of gamma prior.

    f(x) = (x-loc)**(a-1) * e**(-(x-loc)) * (1/scale)**a / gamma(a)

    Parameters
    ----------
    a : float
        The shape parameter. Must be greater than 0.
    scale : float
        The scale parameter (1/scale corresponds to parameter p in canonical form). Must be greather than 0.
    loc : float
        Mean parameter for the distribution.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        a: float,
        scale: float,
        loc: float,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        assert a > 0
        assert scale > 0

        self._a = a
        self._loc = loc
        self._scale = scale

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "a": self._a,
                "loc": self._loc,
                "scale": self._scale,
            }
        )

        return meta

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        return self._rng.gamma(shape=self._a, scale=self._scale, size=n_samples)

    def _get_log_probability(self, theta: float) -> float:
        """Returns the log pdf of theta."""
        if np.ndim(theta) != 0:
            raise NotImplementedError()

        return sps.gamma.logpdf(theta, a=self._a, scale=self._scale, loc=self._loc)

    def _get_gradient(self, theta: float) -> float:
        """Get gradient as computed by Wolfram Alpha."""
        if np.ndim(theta) == 0:
            # Multiply by theta because of the chain rule...
            return ((self._a - 1) / theta - (1 / self._scale)) * theta
        else:
            raise NotImplementedError()
