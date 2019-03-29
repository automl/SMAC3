from inspect import signature

import skopt.learning.gaussian_process.kernels


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


class MagicMixin:
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
    pass


class Product(MagicMixin, skopt.learning.gaussian_process.kernels.Product):
    pass


class ConstantKernel(MagicMixin, skopt.learning.gaussian_process.kernels.ConstantKernel):
    pass


class Matern(MagicMixin, skopt.learning.gaussian_process.kernels.Matern):
    pass


class RBF(MagicMixin, skopt.learning.gaussian_process.kernels.RBF):
    pass


class WhiteKernel(MagicMixin, skopt.learning.gaussian_process.kernels.WhiteKernel):
    pass


for kernel in (
    ConstantKernel, Matern, RBF, WhiteKernel,  # kernels
    Product, Sum,  # Operations
):
    kernel.get_params = get_params
    kernel._signature = _signature
    kernel.hyperparameters = hyperparameters
    kernel.n_dims = n_dims
    kernel.clone_with_theta = clone_with_theta
