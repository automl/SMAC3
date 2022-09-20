from __future__ import annotations

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import scipy.special
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.kernels.base_kernels import AbstractKernel
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class RBFKernel(AbstractKernel, kernels.RBF):
    """RBF kernel implementation."""

    def __init__(
        self,
        length_scale: float | tuple[float, ...] | np.ndarray = 1.0,
        length_scale_bounds: tuple[float, float] | list[tuple[float, float]] | np.ndarray = (1e-5, 1e5),
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
            dists = scipy.spatial.distance.pdist(X / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = scipy.spatial.distance.squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = scipy.spatial.distance.cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)

        if active is not None:
            K = K * active

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * scipy.spatial.distance.squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale**2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient

        return K
