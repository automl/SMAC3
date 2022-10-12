from __future__ import annotations

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.kernels.base_kernels import AbstractKernel
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class WhiteKernel(AbstractKernel, kernels.WhiteKernel):
    """White kernel implementation."""

    def __init__(
        self,
        noise_level: float | tuple[float, ...] = 1.0,
        noise_level_bounds: tuple[float, float] | list[tuple[float, float]] = (1e-5, 1e5),
        operate_on: np.ndarray | None = None,
        has_conditions: bool = False,
        prior: AbstractPrior | None = None,
    ) -> None:

        super().__init__(
            operate_on=operate_on,
            has_conditions=has_conditions,
            prior=prior,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
        )

    def _call(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
        active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(X)

        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.noise_level * np.eye(X.shape[0])

            if active is not None:
                K = K * active

            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return (K, self.noise_level * np.eye(X.shape[0])[:, :, np.newaxis])
                else:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))
