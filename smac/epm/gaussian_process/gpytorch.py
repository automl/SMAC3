from typing import List, Optional, Tuple

import warnings
from collections import OrderedDict

import gpytorch
import numpy as np
import torch

from gpytorch.constraints.constraints import Interval
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import HorseshoePrior
from gpytorch.utils.errors import NotPSDError
from scipy import optimize

from smac.configspace import ConfigurationSpace
from smac.epm.gaussian_process import BaseModel
from smac.epm.gaussian_process.utils.botorch_utils import (
    module_to_array,
    set_params_with_array,
    _scipy_objective_and_grad
)
from smac.utils.constants import VERY_SMALL_NUMBER

warnings.filterwarnings("ignore", module="gpytorch")


class ExactGPModel(ExactGP):
    """Exact GP model serves as a backbone of the class GaussianProcessGPyTorch"""

    def __init__(
        self, train_X: torch.Tensor, train_y: torch.Tensor, base_covar_kernel: Kernel, likelihood: GaussianLikelihood
    ):
        """
        Initialization function

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
        super(ExactGPModel, self).__init__(train_X, train_y, likelihood)
        # in our experiments we find that ZeroMean more robust than ConstantMean when y is normalized
        self.mean_module = ZeroMean()
        self.covar_module = base_covar_kernel

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Compute the posterior mean and variance"""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPyTorchGaussianProcess(BaseModel):
    def __init__(
        self,
        configspace: ConfigurationSpace,
        types: List[int],
        bounds: List[Tuple[float, float]],
        seed: int,
        kernel: Kernel,
        normalize_y: bool = True,
        n_opt_restarts: int = 10,
        likelihood: Optional[FixedNoiseGaussianLikelihood] = None,
        instance_features: Optional[np.ndarray] = None,
        pca_components: Optional[int] = None,
    ):
        """
        A Gaussian Process written with GPyTorch, its interface is written to be compatible with partial sparse gaussian
        process

        Parameters
        ----------
        configspace: ConfigurationSpace
            Configuration space
        types : List[int]
             Specifies the number of categorical values of an input dimension where
             the i-th entry corresponds to the i-th input dimension. Let's say we
             have 2 dimensions where the first dimension consists of 3 different
             categorical choices, and the second dimension is continuous than we
             have to pass [3, 0]. Note that we count starting from 0.
        bounds : List[Tuple[float, float]]
            bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
        seed : int
            Model seed.
        kernel : Kernel
            Specifies the kernel that is used for all Gaussian Process
        normalize_y : bool
            Zero mean unit variance normalization of the output values
        n_opt_restarts : int
            Number of restarts for GP hyperparameter optimization
        likelihood: Optional[FixedNoiseGaussianLikelihood] = None,
            Gaussian Likelihood (or noise)
        instance_features : np.ndarray (I, K)
            Contains the K dimensional instance features of the I different instances
        pca_components : float
            The number of components to keep when using PCA to reduce dimensionality of instance features. Requires to
            set n_feats (> pca_dims).
        """
        super(GPyTorchGaussianProcess, self).__init__(
            configspace,
            types,
            bounds,
            seed,
            kernel,
            instance_features,
            pca_components,
        )
        if likelihood is None:
            noise_prior = HorseshoePrior(0.1)
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior, noise_constraint=Interval(np.exp(-25), np.exp(2), transform=None)
            ).double()
        self.likelihood = likelihood

        self.normalize_y = normalize_y

        n_opt_restarts = int(n_opt_restarts)
        if n_opt_restarts <= 0:
            raise ValueError(f"n_opt_restarts needs to be positive, however, it get {n_opt_restarts}")
        self.n_opt_restarts = n_opt_restarts

        self.hypers = np.empty((0,))
        self.property_dict = OrderedDict()  # type: OrderedDict
        self.is_trained = False

    def _train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True) -> "GPyTorchGaussianProcess":
        """
        Computes the Cholesky decomposition of the covariance of X and
        estimates the GP hyperparameters by optimizing the marginal
        loglikelihood. The prior mean of the GP is set to the empirical
        mean of X.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values, N as the number of points
        do_optimize: boolean
            If set to true, the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        """
        X = self._impute_inactive(X)
        if len(y.shape) == 1:
            self.n_objectives_ = 1
        else:
            self.n_objectives_ = y.shape[1]
        if self.n_objectives_ == 1:
            y = y.flatten()

        if self.normalize_y:
            y = self._normalize_y(y)

        n_tries = 10
        for i in range(n_tries):
            try:
                self.gp = self._get_gp(X, y)
                break
            except Exception as e:
                if i == n_tries - 1:
                    # To avoid Endless loop, we need to stop it when we have n_tries unsuccessful tries.
                    raise e

        if do_optimize:
            self.hypers = self._optimize()
            self.gp = set_params_with_array(self.gp, self.hypers, self.property_dict)
        else:
            self.hypers, self.property_dict, _ = module_to_array(module=self.gp)
        self.is_trained = True
        return self

    def _get_gp(
        self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None
    ) -> Optional[ExactMarginalLogLikelihood]:
        """
        Get the GP model with the given X and y values. As GPyTorch requires the input data to initialize a new
        model, we also pass X and y here. X and y are set optional to ensure compatibility.

        Parameters
        ----------
        X: Optional[np.ndarray(N, D)]
            input feature vectors, N is number of data points, and D is number of feature dimensions
        y: Optional[np.ndarray(N,)]
            input observations, N is number of data points
        Returns
        -------
        mll : Optional[ExactMarginalLogLikelihood]
            a GPyTorch model with Zero Mean and user specified covariance
        """
        if X is None:
            # To be compatible with the base model
            return None

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        self.gp_model = ExactGPModel(X, y, likelihood=self.likelihood, base_covar_kernel=self.kernel).double()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        mll.double()
        return mll

    def _optimize(self) -> np.ndarray:
        """
        Optimizes the marginal log likelihood and returns the best found
        hyperparameter configuration theta.

        Returns
        -------
        theta : np.ndarray(H)
            Hyperparameter vector that maximizes the marginal log likelihood
        """
        x0, property_dict, bounds = module_to_array(module=self.gp)

        bounds = np.asarray(bounds).transpose().tolist()

        self.property_dict = property_dict

        p0 = [x0]

        # Avoid infinite sampling
        n_tries = 5000
        for i in range(n_tries):
            try:
                gp_model = self.gp.pyro_sample_from_prior()
                x_out = []
                for key in property_dict.keys():
                    param = gp_model
                    param_names = key.split(".")
                    for name in param_names:
                        param = getattr(param, name)
                    x_out.append(param.detach().view(-1).cpu().double().clone().numpy())
                sample = np.concatenate(x_out)
                p0.append(sample.astype(np.float64))
            except Exception as e:
                if i == n_tries - 1:
                    self.logger.debug(f"Fails to sample new hyperparameters because of {e}")
                    raise e
                continue
            if len(p0) == self.n_opt_restarts:
                break

        self.gp_model.train()
        self.likelihood.train()

        theta_star = x0
        f_opt_star = np.inf
        for i, start_point in enumerate(p0):
            try:
                theta, f_opt, _ = optimize.fmin_l_bfgs_b(
                    _scipy_objective_and_grad,
                    start_point,
                    args=(self.gp, property_dict),
                    bounds=bounds,
                )
            except NotPSDError as e:
                self.logger.warning(f"Fail to optimize the GP hyperparameters as an Error occurs: {e}")
                f_opt = np.inf
                theta = start_point
            if f_opt < f_opt_star:
                f_opt_star = f_opt
                theta_star = theta
        return theta_star

    def _predict(
        self, X_test: np.ndarray, cov_return_type: Optional[str] = "diagonal_cov"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        cov_return_type: Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        -------
        np.array(N,)
            predictive mean
        np.array(N,) or np.array(N, N) or None
            predictive variance or standard deviation

        """
        if not self.is_trained:
            raise Exception("Model has to be trained first!")

        X_test = torch.from_numpy(self._impute_inactive(X_test))
        self.likelihood.eval()
        self.gp_model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            observed_pred = self.likelihood(self.gp_model(X_test))

            mu = observed_pred.mean.numpy()
            if cov_return_type is None:
                var = None

                if self.normalize_y:
                    mu = self._untransform_y(mu)

            else:
                if cov_return_type != "full_cov":
                    var = observed_pred.stddev.numpy()
                    var = var**2  # since we get standard deviation for faster computation
                else:
                    # output full covariance
                    var = observed_pred.covariance_matrix.numpy()

                # Clip negative variances and set them to the smallest
                # positive float value
                var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

                if self.normalize_y:
                    mu, var = self._untransform_y(mu, var)

                if cov_return_type == "diagonal_std":
                    var = np.sqrt(var)  # converting variance to std deviation if specified

        return mu, var

    def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
        """
        Samples F function values from the current posterior at the N
        specified test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points
        n_funcs: int
            The number of function values that are drawn at each test point.

        Returns
        -------
        function_samples: np.array(F, N)
            The F function values drawn at the N test points.
        """
        if not self.is_trained:
            raise Exception("Model has to be trained first!")
        self.likelihood.eval()
        self.gp_model.eval()

        X_test = torch.from_numpy(self._impute_inactive(X_test))
        with torch.no_grad():
            funcs = self.likelihood(self.gp_model(X_test)).sample(torch.Size([n_funcs])).t().cpu().numpy()

        if self.normalize_y:
            funcs = self._untransform_y(funcs)

        if len(funcs.shape) == 1:
            return funcs[None, :]
        else:
            return funcs
