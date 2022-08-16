from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.stats as sps

from smac.constants import VERY_SMALL_NUMBER
from smac.model.gaussian_process.priors.prior import Prior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class LogNormalPrior(Prior):
    def __init__(
        self,
        sigma: float,
        mean: float = 0,
        seed: int = 0,
    ):
        """Log normal prior.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        sigma: float
            Specifies the standard deviation of the normal
            distribution.
        rng: np.random.RandomState
            Random number generator
        mean: float
            Specifies the mean of the normal distribution
        """
        super().__init__(seed=seed)

        if mean != 0:
            raise NotImplementedError(mean)

        self.sigma = sigma
        self.sigma_square = sigma**2
        self.mean = mean
        self.sqrt_2_pi = np.sqrt(2 * np.pi)

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
        if theta <= self.mean:
            return -1e25
        else:
            rval = -((math.log(theta) - self.mean) ** 2) / (2 * self.sigma_square) - math.log(
                self.sqrt_2_pi * self.sigma * theta
            )
            return rval

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
        return self.rng.lognormal(mean=self.mean, sigma=self.sigma, size=n_samples)

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration in log space

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        if theta <= 0:
            return 0
        else:
            # derivative of log(1 / (x * s^2 * sqrt(2 pi)) * exp( - 0.5 * (log(x ) / s^2))^2))
            # This is without the mean!!!
            return -(self.sigma_square + math.log(theta)) / (self.sigma_square * (theta)) * theta
