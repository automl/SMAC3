from inspect import signature

import skopt.learning.gaussian_process.kernels
import numpy as np


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
        self.operate_on = np.array(operate_on, dtype=np.int)
        self.len_active = len(operate_on)
    else:
        self.operate_on = None
        self.len_active = None


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
        print('Sum')
        if not isinstance(b, skopt.learning.gaussian_process.kernels.Kernel):
            return Sum(self, ConstantKernel(b))
        return Sum(self, b)

    def __radd__(self, b):
        print('Sum')
        if not isinstance(b, skopt.learning.gaussian_process.kernels.Kernel):
            return Sum(ConstantKernel(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        print('Mul')
        if not isinstance(b, skopt.learning.gaussian_process.kernels.Kernel):
            return Product(self, ConstantKernel(b))
        return Product(self, b)

    def __rmul__(self, b):
        print('Mul')
        if not isinstance(b, skopt.learning.gaussian_process.kernels.Kernel):
            return Product(ConstantKernel(b), self)
        return Product(b, self)


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


for kernel in (
    ConstantKernel, Matern, RBF, WhiteKernel, HammingKernel,  # kernels
    Product, Sum,  # Operations
):
    kernel.get_params = get_params
    kernel._signature = _signature
    kernel.hyperparameters = hyperparameters
    kernel.n_dims = n_dims
    kernel.clone_with_theta = clone_with_theta
    kernel.set_active_dims = set_active_dims
