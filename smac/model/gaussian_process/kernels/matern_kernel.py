from __future__ import annotations

import math

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import scipy.special
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.kernels.base_kernels import AbstractKernel
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class MaternKernel(AbstractKernel, kernels.Matern):
    """Matern kernel implementation."""

    def __init__(
        self,
        length_scale: float | tuple[float, ...] | np.ndarray = 1.0,
        length_scale_bounds: tuple[float, float] | list[tuple[float, float]] | np.ndarray = (1e-5, 1e5),
        nu: float = 1.5,
        operate_on: np.ndarray | None = None,
        has_conditions: bool = False,
        prior: AbstractPrior | None = None,
    ) -> None:
        super().__init__(
            operate_on=operate_on,
            has_conditions=has_conditions,
            prior=prior,
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=nu,
        )

    def _call(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
        active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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
