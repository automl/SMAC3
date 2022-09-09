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


class HammingKernel(
    MagicMixinKernel,
    kernels.StationaryKernelMixin,
    kernels.NormalizedKernelMixin,
    kernels.Kernel,
):
    def __init__(
        self,
        length_scale: Union[float, Tuple[float, ...], np.ndarray] = 1.0,
        length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray] = (
            1e-5,
            1e5,
        ),
        operate_on: Optional[np.ndarray] = None,
        prior: Optional[AbstractPrior] = None,
        has_conditions: bool = False,
    ) -> None:
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    @property
    def hyperparameter_length_scale(self) -> kernels.Hyperparameter:
        """Hyperparameter of the length scale."""
        length_scale = self.length_scale
        anisotropic = np.iterable(length_scale) and len(length_scale) > 1  # type: ignore
        if anisotropic:
            return kernels.Hyperparameter("length_scale", "numeric", self.length_scale_bounds, len(length_scale))  # type: ignore  # noqa: E501
        return kernels.Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

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
        X : [array-like, shape=(n_samples_X, n_features)]
            Left argument of the returned kernel k(X, Y)
        Y : [array-like, shape=(n_samples_Y, n_features) or None(default)]
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : [bool, False(default)]
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : [array-like, shape=(n_samples_X, n_samples_Y)]
            Kernel k(X, Y)

        K_gradient : [array-like, shape=(n_samples_X, n_samples_X, n_dims)]
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.

        Note
        ----
        Code partially copied from skopt (https://github.com/scikit-optimize).
        Made small changes to only compute necessary values and use scikit-learn helper functions.
        """
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
