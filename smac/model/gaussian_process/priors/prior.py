from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.stats as sps

from smac.constants import VERY_SMALL_NUMBER

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Prior:
    """Abstract base class to define the interface for priors of GP hyperparameter.

    This class is adapted from RoBO:

    Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
    RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
    In: NIPS 2017 Bayesian Optimization Workshop

    [16.04.2019]: Whenever lnprob or the gradient is computed for a scalar input, we use math.* rather than np.*

    Parameters
    ----------
    rng: np.random.RandomState
        Random number generator
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    def lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Theta must be on a log scale! This method exponentiates theta and calls ``self._lnprob``.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        return self._lnprob(np.exp(theta))

    def _lnprob(self, theta: float) -> float:
        """Return the log probability of theta.

        Theta must be on the original scale.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration on the original scale.

        Returns
        -------
        float
            The log probability of theta
        """
        raise NotImplementedError()

    def sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns ``n_samples`` from the prior.

        All samples are on a log scale. This method calls ``self._sample_from_prior`` and applies a log transformation
        to the obtained values.

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

        sample = np.log(self._sample_from_prior(n_samples=n_samples))

        if np.any(~np.isfinite(sample)):
            raise ValueError("Sample %s from prior %s contains infinite values!" % (sample, self))

        return sample

    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns ``n_samples`` from the prior.

        All samples are on a original scale.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        raise NotImplementedError()

    def gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Theta must be on the original scale.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space

        Returns
        -------
        float
            The gradient of the prior at theta.
        """
        return self._gradient(np.exp(theta))

    def _gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in the original space space

        Returns
        -------
        float
            The gradient of the prior at theta.
        """
        raise NotImplementedError()
