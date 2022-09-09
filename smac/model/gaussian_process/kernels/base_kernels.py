from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
from inspect import Signature, signature

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import scipy.special
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.kernels.utils import get_conditional_hyperparameters
from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class MagicMixinKernel:
    # This is a mixin for a kernel to override functions of the kernel.
    # Because it overrides functions of the kernel, it needs to be placed first in the inheritance
    # hierarchy. For this reason it is not possible to subclass the
    # Mixin from the kernel class because this will prevent it from being instantiatable.
    # Therefore, mypy won't know about anything related to the superclass and I had
    # to add a few type:ignore statements when accessing a member that is declared in the
    # superclass such as self.has_conditions, self._call, super().get_params etc.

    prior = None  # type: Optional[AbstractPrior]

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        eval_gradient: bool = False,
        active: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Call the kernel function."""
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
            if self.len_active is None:
                raise RuntimeError("len_active is not set.")

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
            return SumKernel(self, ConstantKernel(b))
        return SumKernel(self, b)

    def __radd__(self, b: Union[kernels.Kernel, float]) -> kernels.Sum:
        if not isinstance(b, kernels.Kernel):
            return SumKernel(ConstantKernel(b), self)
        return SumKernel(b, self)

    def __mul__(self, b: Union[kernels.Kernel, float]) -> kernels.Product:
        if not isinstance(b, kernels.Kernel):
            return ProductKernel(self, ConstantKernel(b))
        return ProductKernel(self, b)

    def __rmul__(self, b: Union[kernels.Kernel, float]) -> kernels.Product:
        if not isinstance(b, kernels.Kernel):
            return ProductKernel(ConstantKernel(b), self)
        return ProductKernel(b, self)

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
            if "has_conditions" not in args:
                args.append("has_conditions")
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

        self._n_dims_cache = -1  # type: int
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
        """Sets dimensions this kernel should work on.

        Parameters
        ----------
        operate_on : None, list or array, shape (n_dims,)
        """
        if operate_on is not None and type(operate_on) in (list, np.ndarray):
            if not isinstance(operate_on, np.ndarray):
                raise TypeError("argument operate_on needs to be of type np.ndarray, but is %s" % type(operate_on))
            if operate_on.dtype != int:
                raise ValueError("dtype of argument operate_on needs to be int, but is %s" % operate_on.dtype)
            self.operate_on = operate_on  # type: Optional[np.ndarray]
            self.len_active = len(operate_on)  # type: Optional[int]
        else:
            self.operate_on = None
            self.len_active = None


class SumKernel(MagicMixinKernel, kernels.Sum):
    def __init__(
        self,
        k1: kernels.Kernel,
        k2: kernels.Kernel,
        operate_on: np.ndarray = None,
        has_conditions: bool = False,
    ) -> None:
        super(SumKernel, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)
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


class ProductKernel(MagicMixinKernel, kernels.Product):
    def __init__(
        self,
        k1: kernels.Kernel,
        k2: kernels.Kernel,
        operate_on: np.ndarray = None,
        has_conditions: bool = False,
    ) -> None:
        super(ProductKernel, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)
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
            return K1 * K2, np.dstack((K1_gradient * K2[:, :, np.newaxis], K2_gradient * K1[:, :, np.newaxis]))
        else:
            return self.k1(X, Y, active=active) * self.k2(X, Y, active=active)


class ConstantKernel(MagicMixinKernel, kernels.ConstantKernel):
    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: Tuple[float, float] = (1e-5, 1e5),
        operate_on: Optional[np.ndarray] = None,
        prior: Optional[AbstractPrior] = None,
        has_conditions: bool = False,
    ) -> None:

        super(ConstantKernel, self).__init__(constant_value=constant_value, constant_value_bounds=constant_value_bounds)
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
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        K = np.full(
            (X.shape[0], Y.shape[0]),
            self.constant_value,
            dtype=np.array(self.constant_value).dtype,
        )
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return (
                    K,
                    np.full(
                        (X.shape[0], X.shape[0], 1),
                        self.constant_value,
                        dtype=np.array(self.constant_value).dtype,
                    ),
                )
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K
