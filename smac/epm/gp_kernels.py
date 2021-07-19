import copy
from inspect import signature, Signature
import math
from typing import Optional, Union, Tuple, List, Callable, Dict, Any

import numpy as np
import sklearn.gaussian_process.kernels
import scipy.optimize
import scipy.spatial.distance
import scipy.special

import torch
from gpytorch import settings
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.means.mean import Mean
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, PsdSumLazyTensor, RootLazyTensor, delazify
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood

from smac.epm.gp_base_prior import Prior
import skopt.learning.gaussian_process.kernels as kernels


# This file contains almost no type annotations to simplify comparing it to the original scikit-learn version!


def get_conditional_hyperparameters(X: np.ndarray, Y: Optional[np.ndarray]) -> np.ndarray:
    # Taking care of conditional hyperparameters according to Levesque et al.
    X_cond = X <= -1
    if Y is not None:
        Y_cond = Y <= -1
    else:
        Y_cond = X <= -1
    active = ~((np.expand_dims(X_cond, axis=1) != Y_cond).any(axis=2))
    return active


class MagicMixin:
    # This is a mixin for a kernel to override functions of the kernel. Because it overrides functions of the kernel,
    # it needs to be placed first in the inheritance hierarchy. For this reason it is not possible to subclass the
    # Mixin from the kernel class because this will prevent it from being instantiatable. Therefore, mypy won't know
    # about anything related to the superclass and I had to add a few type:ignore statements when accessing a member
    # that is declared in the superclass such as self.has_conditions, self._call, super().get_params etc.

    prior = None  # type: Optional[Prior]

    def __call__(
            self,
            X: np.ndarray,
            Y: Optional[np.ndarray] = None,
            eval_gradient: bool = False,
            active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        if active is None and self.has_conditions:  # type: ignore[attr-defined] # noqa F821
            if self.operate_on is None:
                active = get_conditional_hyperparameters(X, Y)
            else:
                if Y is None:
                    active = get_conditional_hyperparameters(X[:, self.operate_on], None)
                else:
                    active = get_conditional_hyperparameters(X[:, self.operate_on], Y[:, self.operate_on])

        if self.operate_on is None:
            rval = self._call(X, Y, eval_gradient, active)  # type: ignore[attr-defined] # noqa F821
        else:
            if Y is None:
                rval = self._call(  # type: ignore[attr-defined] # noqa F821
                    X=X[:, self.operate_on].reshape([-1, self.len_active]),
                    Y=None,
                    eval_gradient=eval_gradient,
                    active=active,
                )
                X = X[:, self.operate_on].reshape((-1, self.len_active))
            else:
                rval = self._call(  # type: ignore[attr-defined] # noqa F821
                    X=X[:, self.operate_on].reshape([-1, self.len_active]),
                    Y=Y[:, self.operate_on].reshape([-1, self.len_active]),
                    eval_gradient=eval_gradient,
                    active=active,
                )
                X = X[:, self.operate_on].reshape((-1, self.len_active))
                Y = Y[:, self.operate_on].reshape((-1, self.len_active))

        return rval

    def __add__(self, b: Union[kernels.Kernel, float]) -> kernels.Sum:
        if not isinstance(b, kernels.Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b: Union[kernels.Kernel, float]) -> kernels.Sum:
        if not isinstance(b, kernels.Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b: Union[kernels.Kernel, float]) -> kernels.Product:
        if not isinstance(b, kernels.Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b: Union[kernels.Kernel, float]) -> kernels.Product:
        if not isinstance(b, kernels.Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def _signature(self, func: Callable) -> Signature:
        try:
            sig_ = self._signature_cache.get(func)  # type: Optional[Signature]
        except AttributeError:
            self._signature_cache = {}  # type: Dict[Callable, Signature]
            sig_ = None
        if sig_ is None:
            sig = signature(func)
            self._signature_cache[func] = sig
            return sig
        else:
            return sig_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict()

        try:
            args = self._args_cache
        except AttributeError:
            # ignore[misc] looks like it catches all kinds of errors, but misc is actually a category from mypy:
            # https://mypy.readthedocs.io/en/latest/error_code_list.html#miscellaneous-checks-misc
            tmp = super().get_params(deep)  # type: ignore[misc] # noqa F821
            args = list(tmp.keys())
            # Sum and Product do not clone the 'has_conditions' attribute by default. Instead of changing their
            # get_params() method, we simply add the attribute here!
            if 'has_conditions' not in args:
                args.append('has_conditions')
            self._args_cache = args  # type: List[Union[str, Any]]

        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    @property
    def hyperparameters(self) -> List[kernels.Hyperparameter]:
        """Returns a list of all hyperparameter specifications."""
        try:
            return self._hyperparameters_cache
        except AttributeError:
            pass

        r = super().hyperparameters  # type: ignore[misc] # noqa F821
        self._hyperparameters_cache = r  # type: List[kernels.Hyperparameter]

        return r

    @property
    def n_dims(self) -> int:
        """Returns the number of non-fixed hyperparameters of the kernel."""

        try:
            return self._n_dims_cache
        except AttributeError:
            pass

        self._n_dims_cache = -1  # type: int # I cannot use `varname: type = value` syntax because that's >=Python3.6
        self._n_dims_cache = super().n_dims  # type: ignore[misc] # noqa F821
        return self._n_dims_cache

    def clone_with_theta(self, theta: np.ndarray) -> kernels.Kernel:
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The hyperparameters
        """
        self.theta = theta
        return self

    def set_active_dims(self, operate_on: Optional[np.ndarray] = None) -> None:
        """Sets dimensions this kernel should work on

        Parameters
        ----------
        operate_on : None, list or array, shape (n_dims,)
        """
        if operate_on is not None and type(operate_on) in (list, np.ndarray):
            if not isinstance(operate_on, np.ndarray):
                raise TypeError('argument operate_on needs to be of type np.ndarray, but is %s' % type(operate_on))
            if operate_on.dtype != np.int:
                raise ValueError('dtype of argument operate_on needs to be np.int, but is %s' % operate_on.dtype)
            self.operate_on = operate_on  # type: Optional[np.ndarray]
            self.len_active = len(operate_on)  # type: Optional[int]
        else:
            self.operate_on = None
            self.len_active = None


class Sum(MagicMixin, kernels.Sum):

    def __init__(
            self,
            k1: kernels.Kernel,
            k2: kernels.Kernel,
            operate_on: np.ndarray = None,
            has_conditions: bool = False,
    ) -> None:
        super(Sum, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)
        self.has_conditions = has_conditions

    def _call(
            self,
            X: np.ndarray,
            Y: Optional[np.ndarray] = None,
            eval_gradient: bool = False,
            active: np.ndarray = None,
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
            hyperparameter is determined.

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
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True, active=active)
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True, active=active)
            return K1 + K2, np.dstack((K1_gradient, K2_gradient))
        else:
            return self.k1(X, Y, active=active) + self.k2(X, Y, active=active)


class Product(MagicMixin, kernels.Product):

    def __init__(
            self,
            k1: kernels.Kernel,
            k2: kernels.Kernel,
            operate_on: np.ndarray = None,
            has_conditions: bool = False,
    ) -> None:
        super(Product, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)
        self.has_conditions = has_conditions

    def _call(
            self,
            X: np.ndarray,
            Y: Optional[np.ndarray] = None,
            eval_gradient: bool = False,
            active: np.ndarray = None,
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
            hyperparameter is determined.

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
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, eval_gradient=True, active=active)
            K2, K2_gradient = self.k2(X, Y, eval_gradient=True, active=active)
            return K1 * K2, np.dstack((K1_gradient * K2[:, :, np.newaxis],
                                       K2_gradient * K1[:, :, np.newaxis]))
        else:
            return self.k1(X, Y, active=active) * self.k2(X, Y, active=active)


class ConstantKernel(MagicMixin, kernels.ConstantKernel):

    def __init__(
            self,
            constant_value: float = 1.0,
            constant_value_bounds: Tuple[float, float] = (1e-5, 1e5),
            operate_on: Optional[np.ndarray] = None,
            prior: Optional[Prior] = None,
            has_conditions: bool = False,
    ) -> None:

        super(ConstantKernel, self).__init__(constant_value=constant_value, constant_value_bounds=constant_value_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

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
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        K = np.full((X.shape[0], Y.shape[0]), self.constant_value,
                    dtype=np.array(self.constant_value).dtype)
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (K, np.full((X.shape[0], X.shape[0], 1),
                                   self.constant_value,
                                   dtype=np.array(self.constant_value).dtype))
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K


class Matern(MagicMixin, kernels.Matern):

    def __init__(
            self,
            length_scale: Union[float, Tuple[float, ...]] = 1.0,
            length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (1e-5, 1e5),
            nu: float = 1.5,
            operate_on: Optional[np.ndarray] = None,
            prior: Optional[Prior] = None,
            has_conditions: bool = False,
    ) -> None:

        super(Matern, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

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
        length_scale = sklearn.gaussian_process.kernels._check_length_scale(X, self.length_scale)

        if Y is None:
            dists = scipy.spatial.distance.pdist(X / length_scale, metric='euclidean')
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = scipy.spatial.distance.cdist(X / length_scale, Y / length_scale, metric='euclidean')

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1. + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * self.nu) * K)
            K.fill((2 ** (1. - self.nu)) / scipy.special.gamma(self.nu))
            K *= tmp ** self.nu
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
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale ** 2)
            else:
                D = scipy.spatial.distance.squareform(dists ** 2)[:, :, np.newaxis]

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


class RBF(MagicMixin, kernels.RBF):

    def __init__(
            self,
            length_scale: Union[float, Tuple[float, ...]] = 1.0,
            length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (1e-5, 1e5),
            operate_on: Optional[np.ndarray] = None,
            prior: Optional[Prior] = None,
            has_conditions: bool = False,
    ) -> None:

        super(RBF, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

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
        length_scale = sklearn.gaussian_process.kernels._check_length_scale(X, self.length_scale)

        if Y is None:
            dists = scipy.spatial.distance.pdist(X / length_scale, metric='sqeuclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = scipy.spatial.distance.squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = scipy.spatial.distance.cdist(X / length_scale, Y / length_scale, metric='sqeuclidean')
            K = np.exp(-.5 * dists)

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
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient

        return K


class WhiteKernel(MagicMixin, kernels.WhiteKernel):

    def __init__(
            self,
            noise_level: Union[float, Tuple[float, ...]] = 1.0,
            noise_level_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (1e-5, 1e5),
            operate_on: Optional[np.ndarray] = None,
            prior: Optional[Prior] = None,
            has_conditions: bool = False,
    ) -> None:

        super(WhiteKernel, self).__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

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


class HammingKernel(MagicMixin, kernels.HammingKernel):

    def __init__(
            self,
            length_scale: Union[float, Tuple[float, ...]] = 1.0,
            length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]]] = (1e-5, 1e5),
            operate_on: Optional[np.ndarray] = None,
            prior: Optional[Prior] = None,
            has_conditions: bool = False,
    ) -> None:
        super(HammingKernel, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

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
        length_scale = sklearn.gaussian_process.kernels._check_length_scale(X, self.length_scale)

        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("gradient can be evaluated only when Y != X")
        else:
            Y = np.atleast_2d(Y)

        indicator = np.expand_dims(X, axis=1) != Y
        K = (-1 / (2 * length_scale ** 2) * indicator).sum(axis=2)
        K = np.exp(K)

        if active is not None:
            K = K * active

        if eval_gradient:
            # dK / d theta = (dK / dl) * (dl / d theta)
            # theta = log(l) => dl / d (theta) = e^theta = l
            # dK / d theta = l * dK / dl

            # dK / dL computation
            if np.iterable(length_scale) and length_scale.shape[0] > 1:
                grad = (np.expand_dims(K, axis=-1) * np.array(indicator, dtype=np.float32))
            else:
                grad = np.expand_dims(K * np.sum(indicator, axis=2), axis=-1)

            grad *= (1 / length_scale ** 3)

            return K, grad
        return K


class PartialSparseKernel(Kernel):
    def __init__(self,
                 base_kernel: Kernel,
                 X_inducing: torch.tensor,
                 likelihood: GaussianLikelihood,
                 X_out: torch.tensor,
                 y_out: torch.tensor,
                 active_dims: Optional[Tuple[int]] = None):
        """
        A kernel for partial sparse gaussian process. When doing forward, it needs to pass two GP kernel where the
        two kernels share the same hyperparameters (kernel length, kernel scale and noises), the first one is a sparse
        GP kernel and has inducing points as its hyperparameter. When computing the posterior of the partial sparse
        kernel, we first compute the posterior w.r.t. outer_X and outer_y. Then we consider this posterior as a prior
        of the second stage where we compute the posterior distribution of X_in (the input of forward function)
        Parameters
        Mean value is computed with:
        \mathbf{\mu_{l'}}  = \mathbf{K_{l',u} \Sigma K_{u,1} \Lambda}^{-1}\mathbf{y_g} \label{eq:mean_sgp}
        and variance value:
        \mathbf{\sigma}^2_{l'} = \mathbf{K_{l',l'}} - \mathbf{Q_{l', l'} + \mathbf{K_{l', u}\Sigma K_{u, l'}}}
        \mathbf{\Sigma} = (\mathbf{K_{u,u}} + \mathbf{K_{u, g} \Lambda}^{-1}\mathbf{K_{g,u}})^{-1}
        \mathbf{\Lambda} = diag[\mathbf{K_{g,g}-Q_{g,g}} + \sigma^2_{noise}\idenmat]
        ----------
        base_kernel: Kernel
            base kernel function
        X_inducing: torch.tensor (N_inducing, D)
            inducing points, should be of size (N_inducing, D), N_inducing is the number of the inducing points
        likelihood: GaussianLikelihood
            GP likelihood
        X_out: torch.tensor (N_out,D)
            data features outside the subregion, needs to be of size (N_out, D), N_out is the number of points outside
            the subspace
        y_out: torch.tensor
            data observations outside the subregion
        active_dims: typing.Optional[typing.Tuple[int]] = None
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        """
        super(PartialSparseKernel, self).__init__(active_dims=active_dims)
        self.has_lengthscale = base_kernel.has_lengthscale
        self.base_kernel = base_kernel
        self.likelihood = likelihood

        if X_inducing.ndimension() == 1:
            X_inducing = X_inducing.unsqueeze(-1)

        self.X_out = X_out
        self.y_out = y_out
        self.register_parameter(name="X_inducing", parameter=torch.nn.Parameter(X_inducing))

    def train(self, mode: bool = True) -> None:
        """
        turn the model into training mode, needs to clear all the cached value as they are not required when doing
        training
        Parameters
        ----------
        mode: bool
        if the model is under training mode
        """
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        if hasattr(self, "_cached_inducing_sigma"):
            del self._cached_inducing_sigma
        if hasattr(self, "_cached_poster_mean_mat"):
            del self._cached_poster_mean_mat
        if hasattr(self, "_train_cached_k_u1"):
            del self._train_cached_k_u1
        if hasattr(self, "_train_cached_inducing_sigma_inv_root"):
            del self._train_cached_inducing_sigma_inv_root
        if hasattr(self, "_train_cached_lambda_diag_inv"):
            del self._train_cached_lambda_diag_inv
        if hasattr(self, "_cached_posterior_mean"):
            del self._cached_posterior_mean
        return super(PartialSparseKernel, self).train(mode)

    @property
    def _inducing_mat(self) -> torch.tensor:
        """
        computes inducing matrix, K(X_inducing, X_inducing)
        Returns
        -------
        res: torch.tensor (N_inducing, N_inducing)
        K(X_inducing, X_inducing)
        """
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = delazify(self.base_kernel(self.X_inducing, self.X_inducing))
            if not self.training:
                self._cached_kernel_mat = res
            return res

    @property
    def _inducing_inv_root(self) -> torch.tensor:
        """
        computes the inverse of the inducing matrix: K_inv(X_inducing, X_inducing) = K(X_inducing, X_inducing)^(-1)
        Returns
        -------
        res: torch.tensor (N_inducing, N_inducing)
        K_inv(X_inducing, X_inducing)
        """
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            chol = psd_safe_cholesky(self._inducing_mat, upper=True, jitter=settings.cholesky_jitter.value())
            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]

            res = inv_root
            if not self.training:
                self._cached_kernel_inv_root = res
            return res

    @property
    def _k_u1(self) -> torch.tensor:
        """
        computes the covariance matrix between the X_inducing and X_out : K(X_inducing, X_out)
        Returns
        -------
        res: torch.tensor (N_inducing, N_out)
         K(X_inducing, X_out)
        """
        if not self.training and hasattr(self, "_cached_k_u1"):
            return self._cached_k_u1
        else:
            res = delazify(self.base_kernel(self.X_inducing, self.X_out))
            if not self.training:
                self._cached_k_u1 = res
            else:
                self._train_cached_k_u1 = res.clone()
            return res

    @property
    def _lambda_diag_inv(self):
        """
        computes the inverse of lambda matrix, is computed by
        \Lambda = diag[\mathbf{K_{X_out,X_out}-Q_{X_out,X_out}} + \sigma^2_{noise}\idenmat] and
        Q{X_out, X_out} = K(X_out, X_inducing) K^{-1}(X_inducing,X_inducing) K(X_inducing, X_out)
        Returns
        -------
        res: torch.tensor (N_out, N_out)
        inverse of the diagonal matrix lambda
        """
        if not self.training and hasattr(self, "_cached_lambda_diag_inv"):
            return self._cached_lambda_diag_inv
        else:
            diag_k11 = delazify(self.base_kernel(self.X_out, diag=True))

            diag_q11 = delazify(RootLazyTensor(self._k_u1.transpose(-1, -2).matmul(self._inducing_inv_root))).diag()

            # Diagonal correction for predictive posterior
            correction = (diag_k11 - diag_q11).clamp(0, math.inf)

            sigma = self.likelihood._shaped_noise_covar(correction.shape).diag()

            res = delazify(DiagLazyTensor((correction + sigma).reciprocal()))

            if not self.training:
                self._cached_lambda_diag_inv = res
            else:
                self._train_cached_lambda_diag_inv = res.clone()
            return res

    @property
    def _inducing_sigma(self):
        """
        computes the inverse of lambda matrix, is computed by
        \mathbf{\Sigma} = (\mathbf{K_{X_inducing,X_inducing}} +
         \mathbf{K_{X_inducing, X_out} \Lambda}^{-1}\mathbf{K_{X_out,X_inducing}})
        Returns
        -------
        res: torch.tensor (N_inducing, N_inducing)
        \Sigma
        """
        if not self.training and hasattr(self, "_cached_inducing_sigma"):
            return self._cached_inducing_sigma
        else:
            k_u1 = self._k_u1
            res = PsdSumLazyTensor(self._inducing_mat, MatmulLazyTensor(k_u1, MatmulLazyTensor(self._lambda_diag_inv,
                                                                                               k_u1.transpose(-1, -2))))
            res = delazify(res)
            if not self.training:
                self._cached_inducing_sigma = res

            return res

    @property
    def _inducing_sigma_inv_root(self):
        """
        inverse of Sigma matrix:
        Returns
        -------
        res: torch.tensor (N_inducing, N_inducing)
        \Sigma ^{-1}
        """
        if not self.training and hasattr(self, "_cached_inducing_sigma_inv_root"):
            return self._cached_inducing_sigma_inv_root
        else:
            chol = psd_safe_cholesky(self._inducing_sigma, upper=True, jitter=settings.cholesky_jitter.value())

            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]
            res = inv_root
            if not self.training:
                self._cached_inducing_sigma_inv_root = res
            else:
                self._train_cached_inducing_sigma_inv_root = res.clone()
            return res

    @property
    def _poster_mean_mat(self):
        """
        A cached value for computing posterior mean of a sparse kernel:
        Returns
        -------
        res: torch.tensor (N_inducing, 1)
        a cached value for computing the posterior mean,
        is defined by  \Sigma K_{u, 1} \Lambda}^{-1}\mathbf{y_out}
        """
        if not self.training and hasattr(self, "_cached_poster_mean_mat"):
            return self._cached_poster_mean_mat
        else:
            inducing_sigma_inv_root = self._inducing_sigma_inv_root
            sigma = RootLazyTensor(inducing_sigma_inv_root)

            k_u1 = self._k_u1
            lambda_diag_inv = self._lambda_diag_inv

            res_mat = delazify(MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))

            res = torch.matmul(res_mat, self.y_out)

            if not self.training:
                self._cached_poster_mean_mat = res
            return res

    def _get_covariance(self, x1: torch.tensor, x2: torch.tensor):
        """
        Compute the posterior covariance matrix of the sparse kernel (will serve as the prior for the GP
        kernel in the second stage)
        Parameters
        ----------
        x1: torch.tensor(N_x1, D)
        first input of the partial sparse kernel
        x2: torch.tensor(N_x2, D)
        second input of the partial sparse kernel
        Returns
        -------
        res: torch.tensor (N_x1, 1) or PsdSumLazyTensor
        a cached value for computing the posterior mean,
        is defined by  \Sigma K_{u, 1} \Lambda}^{-1}\mathbf{y_out}
        """
        k_x1x2 = self.base_kernel(x1, x2)
        k_x1u = delazify(self.base_kernel(x1, self.X_inducing))
        inducing_inv_root = self._inducing_inv_root
        inducing_sigma_inv_root = self._inducing_sigma_inv_root
        if torch.equal(x1, x2):
            q_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_inv_root))

            s_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_sigma_inv_root))
        else:
            k_x2u = delazify(self.base_kernel(x2, self.X_inducing))
            q_x1x2 = MatmulLazyTensor(
                k_x1u.matmul(inducing_inv_root), k_x2u.matmul(inducing_inv_root).transpose(-1, -2)
            )
            s_x1x2 = MatmulLazyTensor(
                k_x1u.matmul(inducing_sigma_inv_root), k_x2u.matmul(inducing_sigma_inv_root).transpose(-1, -2)
            )
        covar = PsdSumLazyTensor(k_x1x2, -1. * q_x1x2, s_x1x2)

        if self.training:
            k_iu = self.base_kernel(x1, self.inducing_points)
            sigma = RootLazyTensor(inducing_sigma_inv_root)

            k_u1 = self._train_cached_k_u1 if hasattr(self, "_train_cached_k_u1") else self._k_u1
            lambda_diag_inv = self._train_cached_lambda_diag_inv \
                if hasattr(self, "_train_cached_lambda_diag_inv") else self._lambda_diag_inv

            mean = torch.matmul(
                delazify(MatmulLazyTensor(k_iu, MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))),
                self.outer_y)

            self._cached_posterior_mean = mean
        return covar

    def _covar_diag(self, inputs):
        """
        covar matrix diagonal
        Parameters
        ----------
        inputs: torch.tensor(N_inputs, D)
        input of the partial sparse kernel
        Returns
        -------
        res: DiagLazyTensor (N_inputs, 1)
        a diagional matrix
        """
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        # Get diagonal of covar
        covar_diag = delazify(self.base_kernel(inputs, diag=True))
        return DiagLazyTensor(covar_diag)

    def posterior_mean(self, inputs):
        """
        posterior mean of the sparse kernel, will serve as the prior mean of the dense kernel
        Parameters
        ----------
        inputs: torch.tensor(N_inputs, D)
        input of the partial sparse kernel
        Returns
        -------
        res: Torch.tensor (N_inputs, 1)
        posterior mean of sparse Kernel
        """
        if self.training and hasattr(self, "_cached_posterior_mean"):
            return self._cached_posterior_mean
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        k_iu = delazify(self.base_kernel(inputs, self.X_inducing))
        poster_mean = self._poster_mean_mat
        res = torch.matmul(k_iu, poster_mean)
        return res

    def forward(self, x1, x2, diag=False, **kwargs):
        covar = self._get_covariance(x1, x2)
        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")

        if diag:
            return covar.diag()
        else:
            return covar

    def num_outputs_per_input(self, x1, x2) -> int:
        """
        Number of outputs given the inputs
        Parameters
        ----------
        x1: torch.tensor(N_x1, D)
        input of the partial sparse kernel
        Returns
        -------
        res: int
        for base kernels such as matern or RBF kernels, this value needs to be 1.
        """
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def __deepcopy__(self, memo):
        replace_inv_root = False
        replace_kernel_mat = False
        replace_k_u1 = False
        replace_lambda_diag_inv = False
        replace_inducing_sigma = False
        replace_inducing_sigma_inv_root = False
        replace_poster_mean = False

        if hasattr(self, "_cached_kernel_inv_root"):
            replace_inv_root = True
            kernel_inv_root = self._cached_kernel_inv_root
        if hasattr(self, "_cached_kernel_mat"):
            replace_kernel_mat = True
            kernel_mat = self._cached_kernel_mat
        if hasattr(self, "_cached_k_u1"):
            replace_k_u1 = True
            k_u1 = self._cached_k_u1
        if hasattr(self, "_cached_lambda_diag_inv"):
            replace_lambda_diag_inv = True
            lambda_diag_inv = self._cached_lambda_diag_inv
        if hasattr(self, "_cached_inducing_sigma"):
            replace_inducing_sigma = True
            inducing_sigma = self._cached_inducing_sigma
        if hasattr(self, "_cached_inducing_sigma_inv_root"):
            replace_inducing_sigma_inv_root = True
            inducing_sigma_inv_root = self._cached_inducing_sigma_inv_root
        if hasattr(self, "_cached_poster_mean_mat"):
            replace_poster_mean = True
            poster_mean_mat = self._cached_poster_mean_mat

        cp = self.__class__(
            base_kernel=copy.deepcopy(self.base_kernel),
            inducing_points=copy.deepcopy(self.X_inducing),
            outer_points=self.X_out,
            outer_y=self.y_out,
            likelihood=self.likelihood,
            active_dims=self.active_dims,
        )

        if replace_inv_root:
            cp._cached_kernel_inv_root = kernel_inv_root

        if replace_kernel_mat:
            cp._cached_kernel_mat = kernel_mat

        if replace_k_u1:
            cp._cached_k_u1 = k_u1

        if replace_lambda_diag_inv:
            cp._cached_lambda_diag_inv = lambda_diag_inv

        if replace_inducing_sigma:
            cp._inducing_sigma = inducing_sigma

        if replace_inducing_sigma_inv_root:
            cp._cached_inducing_sigma_inv_root = inducing_sigma_inv_root

        if replace_poster_mean:
            cp._cached_poster_mean_mat = poster_mean_mat

        return cp


class PartialSparseMean(Mean):
    def __init__(self, covar_module: PartialSparseKernel, batch_shape=torch.Size(), **kwargs):
        """
        Read the posterior mean value of the given partial sparse kernel and serve as a prior mean value for the
        second stage
        Parameters
        ----------
        covar_module: PartialSparseKernel
        a partial sparse kernel
        batch_shape: torch.size
        batch size
        """
        super(PartialSparseMean, self).__init__()
        self.covar_module = covar_module
        self.batch_shape = batch_shape
        self.covar_module = covar_module

    def forward(self, input: torch.tensor):
        """
        Compute the posterior mean from the cached value of partial sparse kernel
        Parameters
        ----------
        input: torch.tensor(N_xin, D)
        input torch tensor
        Returns
        -------
        res: torch.tensor(N_xin)
        posterior mean value of sparse GP model
        """
        # detach is applied here to avoid updating the same parameter twice in the same iteration
        # which might result in an error
        res = self.covar_module.posterior_mean(input).detach()
        return res
