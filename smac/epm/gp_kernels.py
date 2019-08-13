from inspect import signature
import math
from typing import Optional

import numpy as np
import sklearn.gaussian_process.kernels
import scipy.optimize
import scipy.spatial.distance
import scipy.special

from lazy_import import lazy_module
kernels = lazy_module('skopt.learning.gaussian_process.kernels')

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

    prior = None

    def __call__(self, X, Y=None, eval_gradient=False, active=None):

        if active is None and self.has_conditions:
            if self.operate_on is None:
                active = get_conditional_hyperparameters(X, Y)
            else:
                if Y is None:
                    active = get_conditional_hyperparameters(X[:, self.operate_on], None)
                else:
                    active = get_conditional_hyperparameters(X[:, self.operate_on], Y[:, self.operate_on])

        if self.operate_on is None:
            rval = self._call(X, Y, eval_gradient, active)
        else:
            if Y is None:
                rval = self._call(
                    X=X[:, self.operate_on].reshape([-1, self.len_active]),
                    Y=None,
                    eval_gradient=eval_gradient,
                    active=active,
                )
                X = X[:, self.operate_on].reshape((-1, self.len_active))
            else:
                rval = self._call(
                    X=X[:, self.operate_on].reshape([-1, self.len_active]),
                    Y=Y[:, self.operate_on].reshape([-1, self.len_active]),
                    eval_gradient=eval_gradient,
                    active=active,
                )
                X = X[:, self.operate_on].reshape((-1, self.len_active))
                Y = Y[:, self.operate_on].reshape((-1, self.len_active))

        return rval

    def __add__(self, b):
        if not isinstance(b, kernels.Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, kernels.Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, kernels.Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, kernels.Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)

    def _signature(self, func):
        try:
            sig = self._signature_cache.get(func)
        except AttributeError:
            self._signature_cache = dict()
            sig = None
        if sig is None:
            sig = signature(func)
            self._signature_cache[func] = sig
        return sig

    def get_params(self, deep=True):
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
            tmp = super().get_params(deep)
            args = list(tmp.keys())
            # Sum and Product do not clone the 'has_conditions' attribute by default. Instead of changing their
            # get_params() method, we simply add the attribute here!
            if 'has_conditions' not in args:
                args.append('has_conditions')
            self._args_cache = args

        for arg in args:
            params[arg] = getattr(self, arg, None)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter specifications."""
        try:
            return self._hyperparameters_cache
        except AttributeError:
            pass

        r = super().hyperparameters
        self._hyperparameters_cache = r

        return r

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""

        try:
            return self._n_dims_cache
        except AttributeError:
            pass

        self._n_dims_cache = super().n_dims
        return self._n_dims_cache

    def clone_with_theta(self, theta):
        """Returns a clone of self with given hyperparameters theta.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The hyperparameters
        """
        self.theta = theta
        return self

    def set_active_dims(self, operate_on=None):
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
            self.operate_on = operate_on
            self.len_active = len(operate_on)
        else:
            self.operate_on = None
            self.len_active = None


class Sum(MagicMixin, kernels.Sum):

    def __init__(self, k1, k2, operate_on=None, has_conditions=False):
        super(Sum, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)
        self.has_conditions = has_conditions

    def _call(self, X, Y=None, eval_gradient=False, active=None):
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

    def __init__(self, k1, k2, operate_on=None, has_conditions=False):
        super(Product, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)
        self.has_conditions = has_conditions

    def _call(self, X, Y=None, eval_gradient=False, active=None):
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
            constant_value=1.0,
            constant_value_bounds=(1e-5, 1e5),
            operate_on=None,
            prior=None,
            has_conditions=False,
    ):
        super(ConstantKernel, self).__init__(constant_value=constant_value, constant_value_bounds=constant_value_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(self, X, Y=None, eval_gradient=False, active=None):
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
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        nu=1.5,
        operate_on=None,
        prior=None,
        has_conditions=False
    ):
        super(Matern, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(self, X, Y=None, eval_gradient=False, active=None):
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
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        operate_on=None,
        prior=None,
        has_conditions=False,
    ):
        super(RBF, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(self, X, Y=None, eval_gradient=False, active=None):
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
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                             / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K


class WhiteKernel(MagicMixin, kernels.WhiteKernel):

    def __init__(
        self,
        noise_level=1.0,
        noise_level_bounds=(1e-5, 1e5),
        operate_on=None,
        prior=None,
        has_conditions=False,
    ):
        super(WhiteKernel, self).__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(self, X, Y=None, eval_gradient=False, active=None):
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
                    return (K, self.noise_level
                            * np.eye(X.shape[0])[:, :, np.newaxis])
                else:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K
        else:
            return np.zeros((X.shape[0], Y.shape[0]))


class HammingKernel(MagicMixin, kernels.HammingKernel):

    def __init__(
        self,
        length_scale=1.0,
        length_scale_bounds=(1e-5, 1e5),
        operate_on=None,
        prior=None,
        has_conditions=False,
    ):
        super(HammingKernel, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.set_active_dims(operate_on)
        self.prior = prior
        self.has_conditions = has_conditions

    def _call(self, X, Y=None, eval_gradient=False, active=None):
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
        K = (-1/(2*length_scale**2) * indicator).sum(axis=2)
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
