"""
All the functions implemented here come from botorch under MIT license:
https://github.com/pytorch/botorch

The idea is to have a vendored version of BoTorch functions that does not lead to consistent issues with dependencies
related issue: https://github.com/automl/SMAC3/issues/924
"""
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

from collections import OrderedDict
from math import inf

import numpy as np
import torch
from gpytorch import module
from gpytorch.kernels import Kernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.utils.errors import NanError
from torch.nn import Module
from torch.optim import Optimizer

ParameterBounds = Dict[str, Tuple[Optional[Union[float, torch.Tensor]], Optional[Union[float, torch.Tensor]]]]


class CategoricalKernel(Kernel):
    r"""A Kernel for categorical features.

    Computes `exp(-dist(x1, x2) / lengthscale)`, where
    `dist(x1, x2)` is zero if `x1 == x2` and one if `x1 != x2`.
    If the last dimension is not a batch dimension, then the
    mean is considered.

    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the covariance between x1 and x2.

        Parameters
        ----------
        x1 : torch.Tensor
            First set of data, `n x d` or `b x n x d`
        x2 : torch.Tensor
            Second set of data, `m x d` or `b x m x d`
        diag : bool, optional
            Should the Kernel compute the whole kernel, or just the diag?, by default False
        last_dim_is_batch : bool, optional
            If this is true, it treats the last dimension of the data as another batch dimension.
            (Useful for additive structure over the dimensions), by default False

        Returns
        -------
        torch.Tensor | :class:`~linear_operator.operators.LinearOperator`
            The exact size depends on the kernel's evaluation mode:

            * `full_covar`: `n x m` or `b x n x m`
            * `full_covar` with `last_dim_is_batch=True`: `k x n x m` or `b x k x n x m`
            * `diag`: `n` or `b x n`
        """
        delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        dists = delta / self.lengthscale.unsqueeze(-2)
        if last_dim_is_batch:
            dists = dists.transpose(-3, -1)
        else:
            dists = dists.mean(-1)
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


class TorchAttr(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


def module_to_array(
    module: Module,
    bounds: Optional[ParameterBounds] = None,
    exclude: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]]:
    r"""Extract named parameters from a module into a numpy array.

    Only extracts parameters with requires_grad, since it is meant for optimizing.

    Parameters
    ----------
        module: A module with parameters. May specify parameter constraints in
            a `named_parameters_and_constraints` method.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        exclude: A list of parameter names that are to be excluded from extraction.

    Returns
    -------
        3-element tuple containing
        - The parameter values as a numpy array.
        - An ordered dictionary with the name and tensor attributes of each
        parameter.
        - A `2 x n_params` numpy array with lower and upper bounds if at least
        one constraint is finite, and None otherwise.

    Example
    -------
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> parameter_array, property_dict, bounds_out = module_to_array(mll)
    """
    x: List[np.ndarray] = []
    lower: List[np.ndarray] = []
    upper: List[np.ndarray] = []
    property_dict = OrderedDict()
    exclude = set() if exclude is None else exclude

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(module, "named_parameters_and_constraints"):
        for param_name, _, constraint in module.named_parameters_and_constraints():  # type: ignore[operator]
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    for p_name, t in module.named_parameters():
        if p_name not in exclude and t.requires_grad:
            property_dict[p_name] = TorchAttr(shape=t.shape, dtype=t.dtype, device=t.device)
            x.append(t.detach().view(-1).cpu().double().clone().numpy())
            # construct bounds
            if bounds_:
                l_, u_ = bounds_.get(p_name, (-inf, inf))
                if torch.is_tensor(l_):
                    l_ = l_.cpu().detach()  # type: ignore[union-attr]
                if torch.is_tensor(u_):
                    u_ = u_.cpu().detach()  # type: ignore[union-attr]
                # check for Nones here b/c it may be passed in manually in bounds
                lower.append(np.full(t.nelement(), l_ if l_ is not None else -inf))
                upper.append(np.full(t.nelement(), u_ if u_ is not None else inf))

    x_out = np.concatenate(x)
    bounds_out = None
    if bounds_:
        if not all(np.isinf(b).all() for lu in (lower, upper) for b in lu):
            bounds_out = np.stack([np.concatenate(lower), np.concatenate(upper)])
    return x_out, property_dict, bounds_out


def set_params_with_array(module: Module, x: np.ndarray, property_dict: Dict[str, TorchAttr]) -> Module:
    r"""Set module parameters with values from numpy array.

    Parameters
    ----------
        module: Module with parameters to be set
        x: Numpy array with parameter values
        property_dict: Dictionary of parameter names and torch attributes as
            returned by module_to_array.

    Returns
    -------
        Module: module with parameters updated in-place.

    Example
    -------
        >>> mll = ExactMarginalLogLikelihood(model.likelihood, model)
        >>> parameter_array, property_dict, bounds_out = module_to_array(mll)
        >>> parameter_array += 0.1  # perturb parameters (for example only)
        >>> mll = set_params_with_array(mll, parameter_array,  property_dict)
    """
    param_dict = OrderedDict(module.named_parameters())
    start_idx = 0
    for p_name, attrs in property_dict.items():
        # Construct the new tensor
        if len(attrs.shape) == 0:  # deal with scalar tensors
            end_idx = start_idx + 1
            new_data = torch.tensor(x[start_idx], dtype=attrs.dtype, device=attrs.device)
        else:
            end_idx = start_idx + np.prod(attrs.shape)
            new_data = torch.tensor(x[start_idx:end_idx], dtype=attrs.dtype, device=attrs.device).view(*attrs.shape)
        start_idx = end_idx
        # Update corresponding parameter in-place. Disable autograd to update.
        param_dict[p_name].requires_grad_(False)
        param_dict[p_name].copy_(new_data)
        param_dict[p_name].requires_grad_(True)
    return module


def _get_extra_mll_args(
    mll: MarginalLogLikelihood,
) -> Union[List[torch.Tensor], List[List[torch.Tensor]], List]:
    r"""Obtain extra arguments for MarginalLogLikelihood objects.

    Get extra arguments (beyond the model output and training targets) required
    for the particular type of MarginalLogLikelihood for a forward pass.

    Parameters
    ----------
        mll: The MarginalLogLikelihood module.

    Returns
    -------
        Extra arguments for the MarginalLogLikelihood.
        Returns an empty list if the mll type is unknown.
    """
    if isinstance(mll, ExactMarginalLogLikelihood):
        return list(mll.model.train_inputs)
    elif isinstance(mll, SumMarginalLogLikelihood):
        return [list(x) for x in mll.model.train_inputs]
    return []


def _scipy_objective_and_grad(
    x: np.ndarray,
    mll: module,
    property_dict: Dict,
    train_inputs: Optional[torch.Tensor] = None,
    train_targets: Optional[torch.Tensor] = None,
    variational_optimizer: Optional[Optimizer] = None,
) -> Tuple[float, np.ndarray]:
    """
    A modification of from botorch.optim.utils._scipy_objective_and_grad, the key difference is that
    we do an additional natural gradient update before computing the gradient values
    Parameters
    ----------
    x: np.ndarray
        optimizer input
    mll: module
        a gpytorch module whose hyperparameters are defined by x
    property_dict: Dict
        a dict describing how x is mapped to initialize mll
    train_inputs: torch.Tensor (N_input, D)
        input points of the GP model
    train_targets: torch.Tensor (N_input, 1)
        target value of the GP model
    variational_optimizer: Optional[Optimizer]
        an optional variational optimizer that optimize the variational parameter during the optimizing processes
    Returns
    ----------
    loss: np.ndarray
        loss value
    grad: np.ndarray
        gradient w.r.t. the inputs
    ----------
    """
    # A modification of from botorch.optim.utils._scipy_objective_and_grad:
    # https://botorch.org/api/_modules/botorch/optim/utils.html
    # The key difference is that we do an additional natural gradient update here
    if variational_optimizer is not None:
        variational_optimizer.zero_grad()

    mll = set_params_with_array(mll, x, property_dict)

    if train_inputs is None or train_targets is None:
        train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets

    mll.zero_grad()

    try:  # catch linear algebra errors in gpytorch
        output = mll.model(*train_inputs)
        args = [output, train_targets] + _get_extra_mll_args(mll)
        loss = -mll(*args).sum()
    except RuntimeError as e:
        if isinstance(e, NanError):
            return float("nan"), np.full_like(x, "nan")
        else:
            raise e  # pragma: nocover
    loss.backward()
    if variational_optimizer is not None:
        variational_optimizer.step()
    param_dict = OrderedDict(mll.named_parameters())
    grad = []
    for p_name in property_dict:
        t = param_dict[p_name].grad
        if t is None:
            # this deals with parameters that do not affect the loss
            grad.append(np.zeros(property_dict[p_name].shape.numel()))
        else:
            grad.append(t.detach().view(-1).cpu().double().clone().numpy())
    mll.zero_grad()
    return loss.item(), np.concatenate(grad)
