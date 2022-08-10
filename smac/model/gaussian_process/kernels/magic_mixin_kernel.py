from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math
from inspect import Signature, signature

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import scipy.special
import sklearn.gaussian_process.kernels as kernels

from smac.model.gaussian_process.priors.prior import Prior

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class MagicMixin:
    # This is a mixin for a kernel to override functions of the kernel.
    # Because it overrides functions of the kernel, it needs to be placed first in the inheritance
    # hierarchy. For this reason it is not possible to subclass the
    # Mixin from the kernel class because this will prevent it from being instantiatable.
    # Therefore, mypy won't know about anything related to the superclass and I had
    # to add a few type:ignore statements when accessing a member that is declared in the
    # superclass such as self.has_conditions, self._call, super().get_params etc.

    prior = None  # type: Optional[Prior]

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
