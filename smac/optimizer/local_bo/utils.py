from typing import Any, Dict, Union

import numpy as np
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from sklearn.gaussian_process.kernels import Kernel as SKLKernels
from smac.epm.gp_kernels import ConstantKernel, WhiteKernel
from smac.epm.boing_kernels import MixedKernel


def construct_gp_kernel(kernel_kwargs: Dict[str, Any],
                        cont_dims: np.ndarray, cat_dims: np.ndarray) -> Union[Kernel, SKLKernels]:
    """
    construct a GP kernel with the given kernel init kwargs and the cont_dims and cat_dims
    Parameters
    ----------
    kernel_kwargs: Dict[str, Any]
        kernel kwargs. It needs to contain the type of each individual kernels and their initial arguments, including
        constraints and priors. It needs to contain the following items:
            cont_kernel: type of continuous kernels
            cont_kernel_kwargs: additional arguments for continuous kernels, for instance, length constraints and prior
            cat_kernel: type of categorical kernels
            cat_kernel_kwargs: additional arguments for categorical kernels, for instance, length constraints and prior
            scale_kernel: type of scale kernels
            scale_kernel_kwargs: additional arguments for scale kernels,  for instance, length constraints and prior
    cont_dims: np.ndarray
        dimensions of continuous hyperparameters
    cat_dims: np.ndarray
        dimensions of categorical hyperparameters
    Returns
    -------
    kernel: Union[Kernel, SKLKernels]
        constructed kernels

    """
    if len(cont_dims) > 0:
        cont_kernel_class = kernel_kwargs.get('cont_kernel', MaternKernel)
        cont_kernel_kwargs = kernel_kwargs.get('cont_kernel_kwargs', {})
        cont_kernel = cont_kernel_class(ard_num_dims=cont_dims.shape[-1],
                                        active_dims=tuple(cont_dims), **cont_kernel_kwargs).double()

    if len(cat_dims) > 0:
        cat_kernel_class = kernel_kwargs.get('cat_kernel', MaternKernel)
        cat_kernel_kwargs = kernel_kwargs.get('cat_kernel_kwargs', {})
        cat_kernel = cat_kernel_class(ard_num_dims=cat_dims.shape[-1],
                                      active_dims=tuple(cat_dims), **cat_kernel_kwargs).double()

    if len(cont_dims) > 0 and len(cat_dims) > 0:
        if isinstance(cont_kernel, SKLKernels):
            base_kernel = cont_kernel * cat_kernel
        else:
            base_kernel = MixedKernel(cont_kernel=cont_kernel, cat_kernel=cat_kernel)
    elif len(cont_dims) > 0 and len(cat_dims) == 0:
        base_kernel = cont_kernel
    elif len(cont_dims) == 0 and len(cat_dims) > 0:
        base_kernel = cat_kernel
    else:
        raise ValueError('Either cont_dims or cat_dims must exist!')
    if isinstance(base_kernel, SKLKernels):
        scale_kernel_class = kernel_kwargs.get('scale_kernel', ConstantKernel)
        scale_kernel_kwargs = kernel_kwargs.get('scale_kernel_kwargs', {})
        scale_kernel = scale_kernel_class(**scale_kernel_kwargs)

        noise_kernel_class = kernel_kwargs.get('noise_kernel', WhiteKernel)
        noise_kernel_kwargs = kernel_kwargs.get('noise_kernel_kwargs', {})
        noise_kernel = noise_kernel_class(**noise_kernel_kwargs)

        gp_kernel = scale_kernel * base_kernel + noise_kernel
    else:
        scale_kernel_class = kernel_kwargs.get('scale_kernel', ScaleKernel)
        scale_kernel_kwargs = kernel_kwargs.get('scale_kernel_kwargs', {})
        gp_kernel = scale_kernel_class(base_kernel=base_kernel,
                                       **scale_kernel_kwargs)
    return gp_kernel
