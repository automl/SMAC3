from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.stats as sps

from smac.constants import VERY_SMALL_NUMBER
from smac.model.gaussian_process.priors.prior import Prior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HorseshoePrior(Prior):
    def __init__(self, scale: float, seed: int = 0):
        """Horseshoe Prior as it is used in spearmint.

        This class is adapted from RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        scale: float
            Scaling parameter. See below how it is influencing the distribution.
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(seed=seed)
        self.scale = scale
        self.scale_square = scale**2

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Parameters
        ----------
        theta : (D,) numpy array
            A hyperparameter configuration

        Returns
        -------
        float
        """
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
            a = math.log(1 + 3.0 * (self.scale_square / theta**2))
            return math.log(a + VERY_SMALL_NUMBER)

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
        # This is copied from RoBO - scale is most likely the tau parameter
        lamda = np.abs(self.rng.standard_cauchy(size=n_samples))
        p0 = np.abs(self.rng.randn() * lamda * self.scale)
        return p0

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : (D,) numpy array
            Hyperparameter configuration

        Returns
        -------
        (D) np.array
            The gradient of the prior at theta.
        """
        if theta == 0:
            return np.inf  # POSITIVE infinity (this is the "spike")
        else:
            a = -(6 * self.scale_square)
            b = 3 * self.scale_square + theta**2
            b *= math.log(3 * self.scale_square * theta ** (-2) + 1)
            b = max(b, 1e-14)
            return a / b
