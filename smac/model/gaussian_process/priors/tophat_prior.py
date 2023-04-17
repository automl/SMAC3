from __future__ import annotations

from typing import Any

import warnings

import numpy as np

from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class TophatPrior(AbstractPrior):
    """Tophat prior as it used in the original spearmint code.

    Parameters
    ----------
    lower_bound : float
        Lower bound of the prior. In original scale.
    upper_bound : float
        Upper bound of the prior. In original scale.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self._min = lower_bound
        self._log_min = np.log(lower_bound)
        self._max = upper_bound
        self._log_max = np.log(upper_bound)
        self._prob = 1 / (self._max - self._min)
        self._log_prob = np.log(self._prob)

        if not (self._max > self._min):
            raise Exception("Upper bound of Tophat prior must be greater than the lower bound.")

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"lower_bound": self._min, "upper_bound": self._max})

        return meta

    def _get_log_probability(self, theta: float) -> float:
        if theta < self._min or theta > self._max:
            return -np.inf
        else:
            return self._log_prob

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        if np.ndim(n_samples) != 0:
            raise ValueError("The argument `n_samples` needs to be a scalar (is %s)." % n_samples)

        if n_samples <= 0:
            raise ValueError("The argument `n_samples` needs to be positive (is %d)." % n_samples)

        p0 = np.exp(self._rng.uniform(low=self._log_min, high=self._log_max, size=(n_samples,)))

        return p0

    def _get_gradient(self, theta: float) -> float:
        return 0

    def get_gradient(self, theta: float) -> float:  # noqa: D102
        return 0


class SoftTopHatPrior(AbstractPrior):
    """Soft Tophat prior as it used in the original spearmint code.

    Parameters
    ----------
    lower_bound : float
        Lower bound of the prior. In original scale.
    upper_bound : float
        Upper bound of the prior. In original scale.
    exponent : float
        Exponent of the prior.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        exponent: float,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._lower_bound = lower_bound
            try:
                self._log_lower_bound = np.log(lower_bound)
            except RuntimeWarning as w:
                if "invalid value encountered in log" in w.args[0]:
                    raise ValueError("Invalid lower bound %f (cannot compute log)" % lower_bound)

                raise w
            self._upper_bound = upper_bound
            try:
                self._log_upper_bound = np.log(upper_bound)
            except RuntimeWarning as w:
                if "invalid value encountered in log" in w.args[0]:
                    raise ValueError("Invalid lower bound %f (cannot compute log)" % lower_bound)

                raise w

        if exponent <= 0:
            raise ValueError("Exponent cannot be less or equal than zero (but is %f)" % exponent)

        self._exponent = exponent

    def __repr__(self) -> str:
        return "SoftTopHatPrior(lower_bound=%f, upper_bound=%f)" % (
            self._lower_bound,
            self._upper_bound,
        )

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"lower_bound": self._lower_bound, "upper_bound": self._upper_bound, "exponent": self._exponent})

        return meta

    def get_log_probability(self, theta: float) -> float:  # noqa: D102
        # We need to use lnprob here instead of _lnprob to have the squared function work
        # in the logarithmic space, too.
        if np.ndim(theta) == 0:
            if theta < self._log_lower_bound:
                return -((theta - self._log_lower_bound) ** self._exponent)
            elif theta > self._log_upper_bound:
                return -((self._log_upper_bound - theta) ** self._exponent)
            else:
                return 0
        else:
            raise NotImplementedError()

    def get_gradient(self, theta: float) -> float:  # noqa: D102
        if np.ndim(theta) == 0:
            if theta < self._log_lower_bound:
                return -self._exponent * (theta - self._log_lower_bound)
            elif theta > self._log_upper_bound:
                return self._exponent * (self._log_upper_bound - theta)
            else:
                return 0
        else:
            raise NotImplementedError()

    def _get_log_probability(self, theta: float) -> float:
        return 0

    def _get_gradient(self, theta: float) -> float:
        return 0

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        return np.exp(self._rng.uniform(self._log_lower_bound, self._log_upper_bound, size=(n_samples,)))
