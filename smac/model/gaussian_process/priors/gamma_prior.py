from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.stats as sps

from smac.model.gaussian_process.priors.prior import Prior
from smac.constants import VERY_SMALL_NUMBER

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class GammaPrior(Prior):
    def __init__(self, a: float, scale: float, loc: float, seed: int = 0):
        """Gamma prior.

        f(x) = (x-loc)**(a-1) * e**(-(x-loc)) * (1/scale)**a / gamma(a)

        Parameters
        ----------
        a: float > 0
            shape parameter
        scale: float > 0
            scale parameter (1/scale corresponds to parameter p in canonical form)
        loc: float
            mean parameter for the distribution
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(seed=seed)

        self.a = a
        self.loc = loc
        self.scale = scale

    def _lnprob(self, theta: float) -> float:
        """Returns the logpdf of theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration

        Returns
        -------
        float
        """
        if np.ndim(theta) != 0:
            raise NotImplementedError()
        return sps.gamma.logpdf(theta, a=self.a, scale=self.scale, loc=self.loc)

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
        return self.rng.gamma(shape=self.a, scale=self.scale, size=n_samples)

    def _gradient(self, theta: float) -> float:
        """As computed by Wolfram Alpha.

        Parameters
        ----------
        theta: float
            A hyperparameter configuration

        Returns
        -------
        float
        """
        if np.ndim(theta) == 0:
            # Multiply by theta because of the chain rule...
            return ((self.a - 1) / theta - (1 / self.scale)) * theta
        else:
            raise NotImplementedError()
