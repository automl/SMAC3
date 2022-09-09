'''

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from typing import Any, TypeVar


import gpytorch
import numpy as np
import torch
from botorch.optim.numpy_converter import module_to_array, set_params_with_array
from botorch.optim.utils import _get_extra_mll_args, _scipy_objective_and_grad
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import HorseshoePrior
from gpytorch.utils.errors import NanError, NotPSDError
from scipy import optimize
from scipy.stats.qmc import LatinHypercube

from ConfigSpace import ConfigurationSpace
from smac.constants import VERY_SMALL_NUMBER
from smac.model.gaussian_process.abstract_gaussian_process import AbstractGaussianProcess
from smac.model.utils import check_subspace_points
from smac.model.gaussian_process.kernels.boing import FITCKernel, FITCMean

warnings.filterwarnings("ignore", module="gpytorch")

logger = logging.getLogger(__name__)


class ExactGaussianProcessModel(ExactGP):
    """Exact GP model that serves as a backbone for `GPyTorchGaussianProcess`."

    Parameters
    ----------
    train_X: torch.tenor
        input feature
    train_y: torch.tensor
        input observations
    base_covar_kernel: Kernel
        covariance kernel used to compute covariance matrix
    likelihood: GaussianLikelihood
        GP likelihood
    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_y: torch.Tensor,
        base_covar_kernel: Kernel,
        likelihood: GaussianLikelihood,
    ):
        super(ExactGaussianProcessModel, self).__init__(train_X, train_y, likelihood)

        # In our experiments we find that ZeroMean more robust than ConstantMean when y is normalized
        self._mean_module = ZeroMean()
        self._covar_module = base_covar_kernel

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Computes the posterior mean and variance."""
        mean_x = self._mean_module(x)
        covar_x = self._covar_module(x)

        return MultivariateNormal(mean_x, covar_x)


class AugmentedLocalGaussianProcess(ExactGP):
    def __init__(
        self,
        X_in: torch.Tensor,
        y_in: torch.Tensor,
        X_out: torch.Tensor,
        y_out: torch.Tensor,
        likelihood: GaussianLikelihood,
        base_covar_kernel: Kernel,
    ):
        """
        An Augmented Local GP, it is trained with the points inside a subregion while its prior is augemented by the
        points outside the subregion (global configurations)

        Parameters
        ----------
        X_in: torch.Tensor (N_in, D),
            feature vector of the points inside the subregion
        y_in: torch.Tensor (N_in, 1),
            observation inside the subregion
        X_out: torch.Tensor (N_out, D),
            feature vector  of the points outside the subregion
        y_out:torch.Tensor (N_out, 1),
            observation inside the subregion
        likelihood: GaussianLikelihood,
            likelihood of the GP (noise)
        base_covar_kernel: Kernel,
            Covariance Kernel
        """
        X_in = X_in.unsqueeze(-1) if X_in.ndimension() == 1 else X_in
        X_out = X_out.unsqueeze(-1) if X_out.ndimension() == 1 else X_out
        assert X_in.shape[-1] == X_out.shape[-1]

        super(AugmentedLocalGaussianProcess, self).__init__(X_in, y_in, likelihood)

        self._mean_module = ZeroMean()
        self.base_covar = base_covar_kernel

        self.X_out = X_out
        self.y_out = y_out
        self.augmented = False

    def set_augment_module(self, X_inducing: torch.Tensor) -> None:
        """
        Set an augmentation module, which will be used later for inference

        Parameters
        ----------
        X_inducing: torch.Tensor(N_inducing, D)
           inducing points, it needs to have the same number of dimensions as X_in
        """
        X_inducing = X_inducing.unsqueeze(-1) if X_inducing.ndimension() == 1 else X_inducing
        # assert X_inducing.shape[-1] == self.X_out.shape[-1]
        self.covar_module = FITCKernel(
            self.base_covar, X_inducing=X_inducing, X_out=self.X_out, y_out=self.y_out, likelihood=self._likelihood
        )
        self.mean_module = FITCMean(covar_module=self.covar_module)
        self.augmented = True

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Compute the prior values. If optimize_kernel_hps is set True in the training phases, this model degenerates to
        a vanilla GP model with ZeroMean and base_covar as covariance matrix. Otherwise, we apply partial sparse GP
        mean and kernels here.
        """
        if not self.augmented:
            # we only optimize for kernel hyperparameters
            covar_x = self.base_covar(x)
            mean_x = self._mean_module(x)
        else:
            covar_x = self.covar_module(x)
            mean_x = self.mean_module(x)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGaussianProcess(gpytorch.models.ApproximateGP):
    """
    A variational GP to compute the position of the inducing points.
    We only optimize for the position of the continuous dimensions and keep the categorical dimensions constant.
    """

    def __init__(self, kernel: Kernel, X_inducing: torch.Tensor):
        """
        Initialize a Variational GP
        we set the lower bound and upper bounds of inducing points for numerical hyperparameters between 0 and 1,
        that is, we constrain the inducing points to lay inside the subregion.

        Parameters
        ----------
        kernel: Kernel
            kernel of the variational GP, its hyperparameter needs to be fixed when it is by LGPGA
        X_inducing: torch.tensor (N_inducing, D)
            inducing points
        """
        variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(X_inducing.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, X_inducing, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGaussianProcess, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

        shape_X_inducing = X_inducing.shape
        lower_X_inducing = torch.zeros([shape_X_inducing[-1]]).repeat(shape_X_inducing[0])
        upper_X_inducing = torch.ones([shape_X_inducing[-1]]).repeat(shape_X_inducing[0])

        self.variational_strategy.register_constraint(
            param_name="inducing_points",
            constraint=Interval(lower_X_inducing, upper_X_inducing, transform=None),
        )
        self.double()

        for p_name, t in self.named_hyperparameters():
            if p_name != "variational_strategy.inducing_points":
                t.requires_grad = False

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """
        Pass the posterior mean and variance given input X

        Parameters
        ----------
        x: torch.Tensor
            Input data
        Returns
        -------
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, cont_only=True)
        return MultivariateNormal(mean_x, covar_x)
'''
