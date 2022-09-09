from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
from inspect import Signature, signature

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import scipy.special
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.kernels.base_kernels import MagicMixinKernel
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class Matern(MagicMixinKernel, kernels.Matern):
    def __init__(
        self,
        length_scale: Union[float, Tuple[float, ...], np.ndarray] = 1.0,
        length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray] = (
            1e-5,
            1e5,
        ),
        nu: float = 1.5,
        operate_on: Optional[np.ndarray] = None,
        prior: Optional[AbstractPrior] = None,
        has_conditions: bool = False,
    ) -> None:

        super(Matern, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def _call(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = kernels._check_length_scale(X, self.length_scale)

        if Y is None:
            dists = scipy.spatial.distance.pdist(X / length_scale, metric="euclidean")
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = scipy.spatial.distance.cdist(X / length_scale, Y / length_scale, metric="euclidean")

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / scipy.special.gamma(self.nu))
            K *= tmp**self.nu
            K *= scipy.special.kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = scipy.spatial.distance.squareform(K)
            np.fill_diagonal(K, 1)

        if active is not None:
            K = K * active

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale**2)
            else:
                D = scipy.spatial.distance.squareform(dists**2)[:, :, np.newaxis]

            if self.nu == 0.5:
                K_gradient = K[..., np.newaxis] * D / np.sqrt(D.sum(2))[:, :, np.newaxis]
                K_gradient[~np.isfinite(K_gradient)] = 0
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            else:
                # original sklearn code would approximate gradient numerically, but this would violate our assumption
                # that the kernel hyperparameters are not changed within __call__
                raise ValueError(self.nu)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K
