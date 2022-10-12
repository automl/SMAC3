from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractPrior:
    """Abstract base class to define the interface for priors of Gaussian process hyperparameters.

    This class is adapted from RoBO:

    Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
    RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
    In: NIPS 2017 Bayesian Optimization Workshop

    Note
    ----
    Whenever lnprob or the gradient is computed for a scalar input, we use math.* rather than np.*.

    Parameters
    ----------
    seed : int, defaults to 0
    """

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self._seed,
        }

    def sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns `n_samples` from the prior. All samples are on a log scale. This method calls
        `self._sample_from_prior` and applies a log transformation to the obtained values.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        samples : np.ndarray
        """
        if np.ndim(n_samples) != 0:
            raise ValueError("argument n_samples needs to be a scalar (is %s)" % n_samples)

        if n_samples <= 0:
            raise ValueError("argument n_samples needs to be positive (is %d)" % n_samples)

        sample = np.log(self._sample_from_prior(n_samples=n_samples))

        if np.any(~np.isfinite(sample)):
            raise ValueError("Sample %s from prior %s contains infinite values!" % (sample, self))

        return sample

    def get_log_probability(self, theta: float) -> float:
        """Returns the log probability of theta. This method exponentiates theta and calls `self._get_log_probability`.

        Warning
        -------
        Theta must be on a log scale!

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space.

        Returns
        -------
        float
            The log probability of theta
        """
        return self._get_log_probability(np.exp(theta))

    def get_gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta. Internally, his method calls `self._get_gradient`.

        Warning
        -------
        Theta must be on the original scale.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in log space

        Returns
        -------
        gradient : float
            The gradient of the prior at theta.
        """
        return self._get_gradient(np.exp(theta))

    @abstractmethod
    def _get_log_probability(self, theta: float) -> float:
        """Return the log probability of theta.

        Warning
        -------
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

    @abstractmethod
    def _get_gradient(self, theta: float) -> float:
        """Computes the gradient of the prior with respect to theta.

        Parameters
        ----------
        theta : float
            Hyperparameter configuration in the original space space

        Returns
        -------
        gradient : float
            The gradient of the prior at theta.
        """
        raise NotImplementedError()

    @abstractmethod
    def _sample_from_prior(self, n_samples: int) -> np.ndarray:
        """Returns `n_samples` from the prior. All samples are on a original scale.

        Parameters
        ----------
        n_samples : int
            The number of samples that will be drawn.

        Returns
        -------
        np.ndarray
        """
        raise NotImplementedError()
