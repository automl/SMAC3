from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.kernels.base_kernels import AbstractKernel
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class HammingKernel(
    AbstractKernel,
    kernels.StationaryKernelMixin,
    kernels.NormalizedKernelMixin,
    kernels.Kernel,
):
    """Hamming kernel implementation."""

    def __init__(
        self,
        length_scale: float | tuple[float, ...] | np.ndarray = 1.0,
        length_scale_bounds: tuple[float, float] | list[tuple[float, float]] | np.ndarray = (1e-5, 1e5),
        operate_on: np.ndarray | None = None,
        has_conditions: bool = False,
        prior: AbstractPrior | None = None,
    ) -> None:
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

        super().__init__(
            operate_on=operate_on,
            has_conditions=has_conditions,
            prior=prior,
        )

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta

        length_scale = self.length_scale
        if isinstance(length_scale, np.ndarray):
            length_scale = length_scale.tolist()

        length_scale_bounds = self.length_scale_bounds
        if isinstance(length_scale_bounds, np.ndarray):
            length_scale_bounds = length_scale_bounds.tolist()

        meta.update(
            {
                "length_scale": length_scale,
                "lengthscale_bounds": length_scale_bounds,
            }
        )

        return meta

    @property
    def hyperparameter_length_scale(self) -> kernels.Hyperparameter:
        """Hyperparameter of the length scale."""
        length_scale = self.length_scale
        anisotropic = np.iterable(length_scale) and len(length_scale) > 1  # type: ignore

        if anisotropic:
            return kernels.Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(length_scale),  # type: ignore
            )

        return kernels.Hyperparameter(
            "length_scale",
            "numeric",
            self.length_scale_bounds,
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
            Y = X
        elif eval_gradient:
            raise ValueError("gradient can be evaluated only when Y != X")
        else:
            Y = np.atleast_2d(Y)

        indicator = np.expand_dims(X, axis=1) != Y
        K = (-1 / (2 * length_scale**2) * indicator).sum(axis=2)
        K = np.exp(K)

        if active is not None:
            K = K * active

        if eval_gradient:
            # dK / d theta = (dK / dl) * (dl / d theta)
            # theta = log(l) => dl / d (theta) = e^theta = l
            # dK / d theta = l * dK / dl

            # dK / dL computation
            if np.iterable(length_scale) and length_scale.shape[0] > 1:  # type: ignore
                grad = np.expand_dims(K, axis=-1) * np.array(indicator, dtype=np.float32)
            else:
                grad = np.expand_dims(K * np.sum(indicator, axis=2), axis=-1)

            grad *= 1 / length_scale**3

            return K, grad

        return K
