from inspect import signature

import sklearn.gaussian_process.kernels
import skopt.learning.gaussian_process.kernels
import numpy as np


class MagicMixin:

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.operate_on is None:
            return super(MagicMixin, self).__call__(X, Y, eval_gradient)
        else:
            if Y is None:
                return super(MagicMixin, self).__call__(X=X[:, self.operate_on].reshape([-1, self.len_active]), Y=None,
                                                        eval_gradient=eval_gradient)
            else:
                return super(MagicMixin, self).__call__(X=X[:, self.operate_on].reshape([-1, self.len_active]),
                                                        Y=Y[:, self.operate_on].reshape([-1, self.len_active]),
                                                        eval_gradient=eval_gradient)

    def __add__(self, b):
        if not isinstance(b, skopt.learning.gaussian_process.kernels.Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, skopt.learning.gaussian_process.kernels.Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, skopt.learning.gaussian_process.kernels.Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, skopt.learning.gaussian_process.kernels.Kernel):
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
            # introspect the constructor arguments to find the model parameters
            # to represent
            cls = self.__class__
            init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
            init_sign = signature(init)
            args, varargs = [], []
            for parameter in init_sign.parameters.values():
                if (parameter.kind != parameter.VAR_KEYWORD and
                        parameter.name != 'self'):
                    args.append(parameter.name)
                if parameter.kind == parameter.VAR_POSITIONAL:
                    varargs.append(parameter.name)

            if len(varargs) != 0:
                raise RuntimeError("scikit-learn kernels should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s doesn't follow this convention."
                                   % (cls,))

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

        r = []
        for attr in dir(self):
            if attr.startswith("hyperparameter_"):
                r.append(getattr(self, attr))

        self._hyperparameters_cache = r

        return r

    @property
    def n_dims(self):
        """Returns the number of non-fixed hyperparameters of the kernel."""

        try:
            return self._n_dims_cache
        except AttributeError:
            pass

        self._n_dims_cache = self.theta.shape[0]
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


class Sum(MagicMixin, skopt.learning.gaussian_process.kernels.Sum):

    def __init__(self, k1, k2, operate_on=None):
        super(Sum, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)


class Product(MagicMixin, skopt.learning.gaussian_process.kernels.Product):

    def __init__(self, k1, k2, operate_on=None):
        super(Product, self).__init__(k1=k1, k2=k2)
        self.set_active_dims(operate_on)


class ConstantKernel(MagicMixin, skopt.learning.gaussian_process.kernels.ConstantKernel):

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5), operate_on=None):
        super(ConstantKernel, self).__init__(constant_value=constant_value, constant_value_bounds=constant_value_bounds)
        self.set_active_dims(operate_on)


class Matern(MagicMixin, skopt.learning.gaussian_process.kernels.Matern):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5, operate_on=None):
        super(Matern, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.set_active_dims(operate_on)


class RBF(MagicMixin, skopt.learning.gaussian_process.kernels.RBF):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), operate_on=None):
        super(RBF, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.set_active_dims(operate_on)


class WhiteKernel(MagicMixin, skopt.learning.gaussian_process.kernels.WhiteKernel):

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5), operate_on=None):
        super(WhiteKernel, self).__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        self.set_active_dims(operate_on)


class HammingKernel(MagicMixin, skopt.learning.gaussian_process.kernels.HammingKernel):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), operate_on=None):
        super(HammingKernel, self).__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.set_active_dims(operate_on)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        * `X` [array-like, shape=(n_samples_X, n_features)]
            Left argument of the returned kernel k(X, Y)

        * `Y` [array-like, shape=(n_samples_Y, n_features) or None(default)]
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        * `eval_gradient` [bool, False(default)]
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        * `K` [array-like, shape=(n_samples_X, n_samples_Y)]
            Kernel k(X, Y)

        * `K_gradient` [array-like, shape=(n_samples_X, n_samples_X, n_dims)]
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.

        Note
        ----
        Code partially copied from skopt (https://github.com/scikit-optimize).
        Made small changes to only compute necessary values and use scikit-learn helper functions.
        """
        if self.operate_on is None:
            pass
        else:
            if Y is None:
                X = X[:, self.operate_on].reshape([-1, self.len_active])
            else:
                X = X[:, self.operate_on].reshape([-1, self.len_active])
                Y = Y[:, self.operate_on].reshape([-1, self.len_active])

        X = np.atleast_2d(X)
        length_scale = sklearn.gaussian_process.kernels._check_length_scale(X, self.length_scale)

        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("gradient can be evaluated only when Y != X")
        else:
            Y = np.atleast_2d(Y)

        indicator = np.expand_dims(X, axis=1) != Y
        kernel_prod = -(length_scale * indicator).sum(axis=2)
        kernel_prod = np.exp(kernel_prod)

        if eval_gradient:
            # dK / d theta = (dK / dl) * (dl / d theta)
            # theta = log(l) => dl / d (theta) = e^theta = l
            # dK / d theta = l * dK / dl

            # dK / dL computation
            if np.iterable(length_scale) and length_scale.shape[0] > 1:
                grad = (-np.expand_dims(kernel_prod, axis=-1) * np.array(indicator, dtype=np.float32))
            else:
                grad = -np.expand_dims(kernel_prod * np.sum(indicator, axis=2), axis=-1)

            grad *= length_scale

            return kernel_prod, grad
        return kernel_prod
