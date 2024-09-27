from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

from inspect import Signature, signature

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.priors.abstract_prior import AbstractPrior
from smac.utils.configspace import get_conditional_hyperparameters

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractKernel:
    """
    This is a mixin for a kernel to override functions of the kernel. Because it overrides functions of the kernel,
    it needs to be placed first in the inheritance hierarchy. For this reason it is not possible to subclass the
    Mixin from the kernel class because this will prevent it from being instantiatable. Therefore, mypy won't know about
    anything related to the superclass and some type:ignore statements has to be added when accessing a member that is
    declared in the superclass such as `self.has_conditions`, `self._call`, `super().get_params`, etc.

    Parameters
    ----------
    operate_on : np.ndarray, defaults to None
        On which numpy array should be operated on.
    has_conditions : bool, defaults to False
        Whether the kernel has conditions.
    prior : AbstractPrior, defaults to None
        Which prior the kernel is using.

    Attributes
    ----------
    operate_on : np.ndarray, defaults to None
        On which numpy array should be operated on.
    has_conditions : bool, defaults to False
        Whether the kernel has conditions. Might be changed by the gaussian process.
    prior : AbstractPrior, defaults to None
        Which prior the kernel is using. Primarily used by sklearn.
    """

    def __init__(
        self,
        *,
        operate_on: np.ndarray | None = None,
        has_conditions: bool = False,
        prior: AbstractPrior | None = None,
        **kwargs: Any,
    ) -> None:
        self.operate_on = operate_on
        self.has_conditions = has_conditions
        self.prior = prior
        self._set_active_dims(operate_on)

        # Since this class is a mixin, we just pass all the other parameters to the next class.
        super().__init__(**kwargs)

        # Get variables from next class:
        # We make it explicit here to make sure the next class really has this attributes.
        self._hyperparameters: list[kernels.Hyperparameter] = super().hyperparameters  # type: ignore
        self._n_dims: int = super().n_dims  # type: ignore
        self._len_active: int | None

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object. This method calls the `get_params` method to collect the
        parameters of the kernel.
        """
        meta: dict[str, Any] = {"name": self.__class__.__name__}
        meta.update(self.get_params(deep=False))

        # We have to handle some special cases to make the meta data serializable
        for k in meta:
            v = meta[k]
            if isinstance(v, AbstractKernel):
                meta[k] = v.meta

            if isinstance(v, AbstractPrior):
                meta[k] = v.meta

            if isinstance(v, np.ndarray):
                meta[k] = v.tolist()

        return meta

    @property
    def hyperparameters(self) -> list[kernels.Hyperparameter]:
        """Returns a list of all hyperparameter specifications."""
        return self._hyperparameters

    @property
    def n_dims(self) -> int:
        """Returns the number of non-fixed hyperparameters of the kernel."""
        return self._n_dims

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters of this kernel.

        Parameters
        ----------
        deep : bool, defaults to True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict[str, Any]
            Parameter names mapped to their values.
        """
        params = {}

        # ignore[misc] looks like it catches all kinds of errors, but misc is actually a category from mypy:
        # https://mypy.readthedocs.io/en/latest/error_code_list.html#miscellaneous-checks-misc
        tmp = super().get_params(deep)  # type: ignore[misc] # noqa F821
        args = list(tmp.keys())

        # Sum and Product do not clone the 'has_conditions' attribute by default. Instead of changing their
        # get_params() method, we simply add the attribute here!
        if "has_conditions" not in args:
            args.append("has_conditions")

        for arg in args:
            params[arg] = getattr(self, arg, None)

        return params

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
        active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Call the kernel function. Internally, `self._call` is called, which must be specified by a subclass."""
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
            if self._len_active is None:
                raise RuntimeError("The internal variable `_len_active` is not set.")

            if Y is None:
                rval = self._call(
                    X=X[:, self.operate_on].reshape([-1, self._len_active]),
                    Y=None,
                    eval_gradient=eval_gradient,
                    active=active,
                )
                X = X[:, self.operate_on].reshape((-1, self._len_active))
            else:
                rval = self._call(
                    X=X[:, self.operate_on].reshape([-1, self._len_active]),
                    Y=Y[:, self.operate_on].reshape([-1, self._len_active]),
                    eval_gradient=eval_gradient,
                    active=active,
                )
                X = X[:, self.operate_on].reshape((-1, self._len_active))
                Y = Y[:, self.operate_on].reshape((-1, self._len_active))

        return rval

    def __add__(self, b: kernels.Kernel | float) -> kernels.Sum:
        if not isinstance(b, kernels.Kernel):
            return SumKernel(self, ConstantKernel(b))

        return SumKernel(self, b)

    def __radd__(self, b: kernels.Kernel | float) -> kernels.Sum:
        if not isinstance(b, kernels.Kernel):
            return SumKernel(ConstantKernel(b), self)

        return SumKernel(b, self)

    def __mul__(self, b: kernels.Kernel | float) -> kernels.Product:
        if not isinstance(b, kernels.Kernel):
            return ProductKernel(self, ConstantKernel(b))

        return ProductKernel(self, b)

    def __rmul__(self, b: kernels.Kernel | float) -> kernels.Product:
        if not isinstance(b, kernels.Kernel):
            return ProductKernel(ConstantKernel(b), self)

        return ProductKernel(b, self)

    @abstractmethod
    def _call(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
        active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Note
        ----
        Code partially copied from skopt (https://github.com/scikit-optimize).
        Made small changes to only compute necessary values and use scikit-learn helper functions.

        Parameters
        ----------
        X : np.ndarray [#samples, #features]
            Left argument of the returned kernel k(X, Y).
        Y : np.ndarray [#samples, #features], defaults to None
            Right argument of the returned kernel k(X, Y). If None, k(X, X) is evaluated instead.
        eval_gradient : bool, defaults to False
            Determines whether the gradient with respect to the kernel hyperparameter is determined.
            Only supported when `Y` is None.
        active : np.ndarray [#samples, #features], defaults to None
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : np.ndarray [#X_samples, #Y_samples]
            Kernel k(X, Y).
        K_gradient : np.ndarray [#X_samples, #X_samples, #dimensions]
            The gradient of the kernel k(X, X) with respect to the hyperparameter of the kernel.
            Only returned when `eval_gradient` is True.
        """
        raise NotImplementedError

    def _signature(self, func: Callable) -> Signature:
        sig_: Signature | None

        try:
            sig_ = self._signature_cache.get(func)
        except AttributeError:
            self._signature_cache: dict[Callable, Signature] = {}
            sig_ = None

        if sig_ is None:
            sig = signature(func)
            self._signature_cache[func] = sig

            return sig
        else:
            return sig_

    def _set_active_dims(self, operate_on: np.ndarray | None = None) -> None:
        """Sets dimensions this kernel should work on."""
        if operate_on is not None and isinstance(operate_on, (list, np.ndarray)):
            if not isinstance(operate_on, np.ndarray):
                raise TypeError(f"The argument `operate_on` needs to be of type np.ndarray but is {type(operate_on)}")

            if not np.issubdtype(operate_on.dtype, np.integer):
                raise ValueError(f"The dtype of `operate_on` needs to be np.integer, but is {operate_on.dtype}")

            self.operate_on = operate_on
            self._len_active = len(operate_on)
        else:
            self.operate_on = None
            self._len_active = None


class SumKernel(AbstractKernel, kernels.Sum):
    """Sum kernel implementation."""

    def __init__(
        self,
        k1: kernels.Kernel,
        k2: kernels.Kernel,
        operate_on: np.ndarray | None = None,
        has_conditions: bool = False,
    ) -> None:
        super().__init__(
            operate_on=operate_on,
            has_conditions=has_conditions,
            k1=k1,
            k2=k2,
        )

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
        active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).

        Y : np.ndarray, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : np.ndarray, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y).

        K_gradient : np.ndarray (opt.), shape (n_samples_X, n_samples_X, n_dims)
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


class ProductKernel(AbstractKernel, kernels.Product):
    """Product kernel implementation."""

    def __init__(
        self,
        k1: kernels.Kernel,
        k2: kernels.Kernel,
        operate_on: np.ndarray | None = None,
        has_conditions: bool = False,
    ) -> None:
        super().__init__(
            operate_on=operate_on,
            has_conditions=has_conditions,
            k1=k1,
            k2=k2,
        )

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
        active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).

        Y : np.ndarray, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : np.ndarray, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y).

        K_gradient : np.ndarray (opt.), shape (n_samples_X, n_samples_X, n_dims)
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


class ConstantKernel(AbstractKernel, kernels.ConstantKernel):
    def __init__(
        self,
        constant_value: float = 1.0,
        constant_value_bounds: tuple[float, float] = (1e-5, 1e5),
        operate_on: np.ndarray | None = None,
        has_conditions: bool = False,
        prior: AbstractPrior | None = None,
    ) -> None:
        super().__init__(
            operate_on=operate_on,
            has_conditions=has_conditions,
            prior=prior,
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
        )

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
        active: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).

        Y : np.ndarray, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            is evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        active : np.ndarray (n_samples_X, n_features) (optional)
            Boolean array specifying which hyperparameters are active.

        Returns
        -------
        K : np.ndarray, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y).

        K_gradient : np.ndarray (opt.), shape (n_samples_X, n_samples_X, n_dims)
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
