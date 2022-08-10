from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.stats as sps

from smac.model.gaussian_process.priors.prior import Prior
from smac.constants import VERY_SMALL_NUMBER

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class TophatPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float, seed: int = 0):
        """Tophat prior as it used in the original spearmint code.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        lower_bound : float
            Lower bound of the prior. In original scale.
        upper_bound : float
            Upper bound of the prior. In original scale.
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(seed=seed)
        self.min = lower_bound
        self._log_min = np.log(lower_bound)
        self.max = upper_bound
        self._log_max = np.log(upper_bound)
        if not (self.max > self.min):
            raise Exception("Upper bound of Tophat prior must be greater than the lower bound!")

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Parameters
        ----------
        theta : float
            A hyperparameter configuration

        Returns
        -------
        float
        """
        if theta < self.min or theta > self.max:
            return -np.inf
        else:
            return 0

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Return ``n_samples`` from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        if np.ndim(n_samples) != 0:
            raise ValueError("argument n_samples needs to be a scalar (is %s)" % n_samples)
        if n_samples <= 0:
            raise ValueError("argument n_samples needs to be positive (is %d)" % n_samples)

        p0 = np.exp(self.rng.uniform(low=self._log_min, high=self._log_max, size=(n_samples,)))
        return p0

    def gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array

            The gradient of the prior at theta.
        """
        return 0


class SoftTopHatPrior(Prior):
    def __init__(self, lower_bound: float, upper_bound: float, exponent: float, seed: int = 0) -> None:
        super().__init__(seed=seed)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.lower_bound = lower_bound
            try:
                self._log_lower_bound = np.log(lower_bound)
            except RuntimeWarning as w:
                if "invalid value encountered in log" in w.args[0]:
                    raise ValueError("Invalid lower bound %f (cannot compute log)" % lower_bound)

                raise w
            self.upper_bound = upper_bound
            try:
                self._log_upper_bound = np.log(upper_bound)
            except RuntimeWarning as w:
                if "invalid value encountered in log" in w.args[0]:
                    raise ValueError("Invalid lower bound %f (cannot compute log)" % lower_bound)

                raise w

        if exponent <= 0:
            raise ValueError("Exponent cannot be less or equal than zero (but is %f)" % exponent)
        self.exponent = exponent

    def lnprob(self, theta: float) -> float:
        """Return the log probability of theta."""
        # We need to use lnprob here instead of _lnprob to have the squared function work
        # in the logarithmic space, too.
        if np.ndim(theta) == 0:
            if theta < self._log_lower_bound:
                return -((theta - self._log_lower_bound) ** self.exponent)
            elif theta > self._log_upper_bound:
                return -((self._log_upper_bound - theta) ** self.exponent)
            else:
                return 0
        else:
            raise NotImplementedError()

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns N samples from the prior.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        return np.exp(self.rng.uniform(self._log_lower_bound, self._log_upper_bound, size=(n_samples,)))

    def gradient(self, theta: float) -> float:
        """Returns the gradient of the prior at theta."""
        if np.ndim(theta) == 0:
            if theta < self._log_lower_bound:
                return -self.exponent * (theta - self._log_lower_bound)
            elif theta > self._log_upper_bound:
                return self.exponent * (self._log_upper_bound - theta)
            else:
                return 0
        else:
            raise NotImplementedError()

    def __repr__(self) -> str:
        return "SoftTopHatPrior(lower_bound=%f, upper_bound=%f)" % (
            self.lower_bound,
            self.upper_bound,
        )
