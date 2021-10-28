import copy
import math
import typing
from collections import OrderedDict
import pyDOE
import typing

import numpy as np
from scipy import optimize

import torch
import gpytorch
from gpytorch import settings
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.means.mean import Mean
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, PsdSumLazyTensor, RootLazyTensor, delazify
from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import Kernel
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from gpytorch.utils.errors import NanError

from botorch.optim.numpy_converter import module_to_array, set_params_with_array
from botorch.optim.utils import _scipy_objective_and_grad, _get_extra_mll_args

from smac.configspace import ConfigurationSpace
from smac.utils.constants import VERY_SMALL_NUMBER
from smac.epm.base_gp import BaseModel
from smac.epm.gaussian_process_gpytorch import ExactGPModel, GaussianProcessGPyTorch

gpytorch.settings.debug.off()

class PartialSparseKernel(Kernel):
    def __init__(self,
                 base_kernel: Kernel,
                 inducing_points: torch.tensor,
                 likelihood: GaussianLikelihood,
                 outer_points: torch.tensor,
                 outer_y: torch.tensor,
                 active_dims: typing.Optional[typing.Tuple[int]] = None):
        """
        A kernel for partial sparse gaussian process
        Parameters
        ----------
        base_kernel: Kernel
            base kernel function
        inducing_points: torch.tensor
            inducing points
        likelihood: GaussianLikelihood
            GP likelihood
        outer_points: torch.tensor:
            datapoints outside the subregion
        outer_y: torch.tensor
            data observations outside the subregion
        active_dims: typing.Optional[typing.Tuple[int]] = None
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        """
        super(PartialSparseKernel, self).__init__(active_dims=active_dims)
        self.has_lengthscale = base_kernel.has_lengthscale
        self.base_kernel = base_kernel
        self.likelihood = likelihood

        if inducing_points.ndimension() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        self.outer_points = outer_points
        self.outer_y = outer_y
        self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        self.register_added_loss_term("inducing_point_loss_term")

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        if hasattr(self, "_cached_inducing_sigma"):
            del self._cached_inducing_sigma
        if hasattr(self, "_cached_poster_mean_mat"):
            del self._cached_poster_mean_mat
        if hasattr(self, "_train_cached_k_u1"):
            del self._train_cached_k_u1
        if hasattr(self, "_train_cached_inducing_sigma_inv_root"):
            del self._train_cached_inducing_sigma_inv_root
        if hasattr(self, "_train_cached_lambda_diag_inv"):
            del self._train_cached_lambda_diag_inv
        if hasattr(self, "_cached_posterior_mean"):
            del self._cached_posterior_mean
        return super(PartialSparseKernel, self).train(mode)

    @property
    def _inducing_mat(self):
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = delazify(self.base_kernel(self.inducing_points, self.inducing_points))
            if not self.training:
                self._cached_kernel_mat = res
            return res

    @property
    def _inducing_inv_root(self):
        if not self.training and hasattr(self, "_cached_kernel_inv_root"):
            return self._cached_kernel_inv_root
        else:
            chol = psd_safe_cholesky(self._inducing_mat, upper=True, jitter=settings.cholesky_jitter.value())
            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]

            res = inv_root
            if not self.training:
                self._cached_kernel_inv_root = res
            return res

    @property
    def _k_u1(self):
        if not self.training and hasattr(self, "_cached_k_u1"):
            return self._cached_k_u1
        else:
            res = delazify(self.base_kernel(self.inducing_points, self.outer_points))
            if not self.training:
                self._cached_k_u1 = res
            else:
                self._train_cached_k_u1 = res.clone()
            return res

    @property
    def _lambda_diag_inv(self):
        if not self.training and hasattr(self, "_cached_lambda_diag_inv"):
            return self._cached_lambda_diag_inv
        else:
            diag_k11 = delazify(self.base_kernel(self.outer_points, diag=True))

            diag_q11 = delazify(RootLazyTensor(self._k_u1.transpose(-1, -2).matmul(self._inducing_inv_root))).diag()

            # Diagonal correction for predictive posterior
            correction = (diag_k11 - diag_q11).clamp(0, math.inf)

            sigma = self.likelihood._shaped_noise_covar(correction.shape).diag()

            res = delazify(DiagLazyTensor((correction + sigma).reciprocal()))

            if not self.training:
                self._cached_lambda_diag_inv = res
            else:
                self._train_cached_lambda_diag_inv = res.clone()
            return res

    @property
    def _inducing_sigma(self):
        if not self.training and hasattr(self, "_cached_inducing_sigma"):
            return self._cached_inducing_sigma
        else:
            k_u1 = self._k_u1
            res = PsdSumLazyTensor(self._inducing_mat, MatmulLazyTensor(k_u1, MatmulLazyTensor(self._lambda_diag_inv,
                                                                                               k_u1.transpose(-1, -2))))
            res = delazify(res)
            if not self.training:
                self._cached_inducing_sigma = res

            return res

    @property
    def _inducing_sigma_inv_root(self):
        if not self.training and hasattr(self, "_cached_inducing_sigma_inv_root"):
            return self._cached_inducing_sigma_inv_root
        else:
            chol = psd_safe_cholesky(self._inducing_sigma, upper=True, jitter=settings.cholesky_jitter.value())

            eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
            inv_root = torch.triangular_solve(eye, chol)[0]
            res = inv_root
            if not self.training:
                self._cached_inducing_sigma_inv_root = res
            else:
                self._train_cached_inducing_sigma_inv_root = res.clone()
            return res

    @property
    def _poster_mean_mat(self):
        if not self.training and hasattr(self, "_cached_poster_mean_mat"):
            return self._cached_poster_mean_mat
        else:
            inducing_sigma_inv_root = self._inducing_sigma_inv_root
            sigma = RootLazyTensor(inducing_sigma_inv_root)

            k_u1 = self._k_u1
            lambda_diag_inv = self._lambda_diag_inv

            res_mat = delazify(MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))

            res = torch.matmul(res_mat, self.outer_y)

            if not self.training:
                self._cached_poster_mean_mat = res
            return res

    def _get_covariance(self, x1, x2):
        k_x1x2 = self.base_kernel(x1, x2)
        k_x1u = delazify(self.base_kernel(x1, self.inducing_points))
        inducing_inv_root = self._inducing_inv_root
        inducing_sigma_inv_root = self._inducing_sigma_inv_root
        if torch.equal(x1, x2):
            q_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_inv_root))

            s_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_sigma_inv_root))
        else:
            k_x2u = delazify(self.base_kernel(x2, self.inducing_points))
            q_x1x2 = MatmulLazyTensor(
                k_x1u.matmul(inducing_inv_root), k_x2u.matmul(inducing_inv_root).transpose(-1, -2)
            )
            s_x1x2 = MatmulLazyTensor(
                k_x1u.matmul(inducing_sigma_inv_root), k_x2u.matmul(inducing_sigma_inv_root).transpose(-1, -2)
            )
        covar = PsdSumLazyTensor(k_x1x2, -1. * q_x1x2, s_x1x2)

        if self.training:
            k_iu = self.base_kernel(x1, self.inducing_points)
            sigma = RootLazyTensor(inducing_sigma_inv_root)

            k_u1 = self._train_cached_k_u1 if hasattr(self, "_train_cached_k_u1") else self._k_u1
            lambda_diag_inv = self._train_cached_lambda_diag_inv \
                if hasattr(self, "_train_cached_lambda_diag_inv") else self._lambda_diag_inv

            mean = torch.matmul(
                delazify(MatmulLazyTensor(k_iu, MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))),
                self.outer_y)

            self._cached_posterior_mean = mean
        return covar

    def _covar_diag(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        # Get diagonal of covar
        covar_diag = delazify(self.base_kernel(inputs, diag=True))
        return DiagLazyTensor(covar_diag)

    def posterior_mean(self, inputs):
        if self.training and hasattr(self, "_cached_posterior_mean"):
            return self._cached_posterior_mean
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        k_iu = delazify(self.base_kernel(inputs, self.inducing_points))
        poster_mean = self._poster_mean_mat
        res = torch.matmul(k_iu, poster_mean)
        return res

    def forward(self, x1, x2, diag=False, **kwargs):
        covar = self._get_covariance(x1, x2)
        if self.training:
            if not torch.equal(x1, x2):
                raise RuntimeError("x1 should equal x2 in training mode")

        if diag:
            return covar.diag()
        else:
            return covar

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def __deepcopy__(self, memo):
        replace_inv_root = False
        replace_kernel_mat = False
        replace_k_u1 = False
        replace_lambda_diag_inv = False
        replace_inducing_sigma = False
        replace_inducing_sigma_inv_root = False
        replace_poster_mean = False

        if hasattr(self, "_cached_kernel_inv_root"):
            replace_inv_root = True
            kernel_inv_root = self._cached_kernel_inv_root
        if hasattr(self, "_cached_kernel_mat"):
            replace_kernel_mat = True
            kernel_mat = self._cached_kernel_mat
        if hasattr(self, "_cached_k_u1"):
            replace_k_u1 = True
            k_u1 = self._cached_k_u1
        if hasattr(self, "_cached_lambda_diag_inv"):
            replace_lambda_diag_inv = True
            lambda_diag_inv = self._cached_lambda_diag_inv
        if hasattr(self, "_cached_inducing_sigma"):
            replace_inducing_sigma = True
            inducing_sigma = self._cached_inducing_sigma
        if hasattr(self, "_cached_inducing_sigma_inv_root"):
            replace_inducing_sigma_inv_root = True
            inducing_sigma_inv_root = self._cached_inducing_sigma_inv_root
        if hasattr(self, "_cached_poster_mean_mat"):
            replace_poster_mean = True
            poster_mean_mat = self._cached_poster_mean_mat

        cp = self.__class__(
            base_kernel=copy.deepcopy(self.base_kernel),
            inducing_points=copy.deepcopy(self.inducing_points),
            outer_points=self.outer_points,
            outer_y=self.outer_y,
            likelihood=self.likelihood,
            active_dims=self.active_dims,
        )

        if replace_inv_root:
            cp._cached_kernel_inv_root = kernel_inv_root

        if replace_kernel_mat:
            cp._cached_kernel_mat = kernel_mat

        if replace_k_u1:
            cp._cached_k_u1 = k_u1

        if replace_lambda_diag_inv:
            cp._cached_lambda_diag_inv = lambda_diag_inv

        if replace_inducing_sigma:
            cp._inducing_sigma = inducing_sigma

        if replace_inducing_sigma_inv_root:
            cp._cached_inducing_sigma_inv_root = inducing_sigma_inv_root

        if replace_poster_mean:
            cp._cached_poster_mean_mat = poster_mean_mat

        return cp


class PartialSparseMean(Mean):
    def __init__(self, covar_module: PartialSparseKernel, prior=None, batch_shape=torch.Size(), **kwargs):
        super(PartialSparseMean, self).__init__()
        self.covar_module = covar_module
        self.batch_shape = batch_shape
        self.covar_module = covar_module

    def forward(self, input):
        res = self.covar_module.posterior_mean(input).detach()
        return res


class PartailSparseGPModel(ExactGP):
    def __init__(self,
                 in_x: torch.tensor,
                 in_y: torch.tensor,
                 out_x: torch.tensor,
                 out_y:torch.tensor,
                 likelihood: GaussianLikelihood,
                 base_covar_kernel: Kernel,
                 inducing_points: torch.tensor,
                 batch_shape=torch.Size(), ):
        """
        A Partial Sparse Gaussian Process (PSGP), it is dense inside a given subregion and the impact of all other
        points are approximated by a sparse GP
        Parameters:
        ----------
        in_x: torch.tensor,
            feature vector of the points inside the subregion
        in_y: torch.tensor,
            observation inside the subregion
        out_x: torch.tensor,
            feature vector  of the points outside the subregion
        out_y:torch.tensor,
            observation inside the subregion
        likelihood: GaussianLikelihood,
            likelihood of the GP (noise)
        base_covar_kernel: Kernel,
            Covariance Kernel
        inducing_points: torch.tensor,
            position of the inducing points
        """
        in_x = in_x.unsqueeze(-1) if in_x.ndimension() == 1 else in_x
        out_x = out_x.unsqueeze(-1) if out_x.ndimension() == 1 else out_x
        inducing_points = inducing_points.unsqueeze(-1) if inducing_points.ndimension() == 1 else inducing_points
        assert inducing_points.shape[-1] == in_x.shape[-1] == out_x.shape[-1]
        super(PartailSparseGPModel, self).__init__(in_x, in_y, likelihood)

        self.base_covar = base_covar_kernel
        self.covar_module = PartialSparseKernel(self.base_covar, inducing_points=inducing_points,
                                                outer_points=out_x, outer_y=out_y, likelihood=likelihood)
        self.mean_module = PartialSparseMean(covar_module=self.covar_module)
        self._mean_module = ZeroMean()

        self.optimize_kernel_hps = True

    def deactivate_kernel_grad(self):
        """
        We deactive kernel grad to only optimize the position of the inducing points
        """
        self.optimize_kernel_hps = False
        for p_name, t in self.named_parameters():
            if p_name == 'covar_module.inducing_points':
                t.requires_grad = True
            else:
                t.requires_grad = False

    def deactivate_inducing_points_grad(self):
        """
        We deactive inducing points grad to only optimize kernel hyperparameters
        """
        if not self.optimize_kernel_hps:
            raise ValueError("inducing_points will only be inactivate if self.optimize_kernel_hps is set True")
        for p_name, t in self.named_parameters():
            if p_name == 'covar_module.inducing_points':
                t.requires_grad = False
            else:
                t.requires_grad = True

    def forward(self, x):
        if self.training:
            if self.optimize_kernel_hps:
                covar_x = self.base_covar(x)
                mean_x = self._mean_module(x)
            else:
                covar_x = self.covar_module(x)
                mean_x = self.mean_module(x)
        else:
            covar_x = self.covar_module(x)
            mean_x = self.mean_module(x)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGPModel(gpytorch.models.ApproximateGP):
    """
    A variational GP to compute the position of the inducing points
    """
    def __init__(self, kernel: Kernel, inducing_points: torch.tensor):
        """
        Initialize a Variational GP
        Parameters:
        ----------
        kernel: Kernel
            kernel of the variational GP, its hyperparameter needs to be fixed when it is used for initializing a PSGP
        inducing_points: torch.tensor
            inducing points
        """
        variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

        shape_inducing_points = inducing_points.shape
        lower_inducing_points = torch.zeros([shape_inducing_points[-1]]).repeat(shape_inducing_points[0])
        upper_inducing_points = torch.ones([shape_inducing_points[-1]]).repeat(shape_inducing_points[0])

        self.variational_strategy.register_constraint(param_name="inducing_points",
                                                      constraint=Interval(lower_inducing_points,
                                                                          upper_inducing_points,
                                                                          transform=None),
                                                      )
        self.double()

        for p_name, t in self.named_hyperparameters():
            if p_name != "variational_strategy.inducing_points":
                t.requires_grad = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PartialSparseGaussianProcess(GaussianProcessGPyTorch):
    def __init__(self,
                 configspace: ConfigurationSpace,
                 types: typing.List[int],
                 bounds: typing.List[typing.Tuple[float, float]],
                 bounds_cont: np.ndarray,
                 bounds_cat: typing.List[typing.List[typing.Tuple]],
                 seed: int,
                 kernel: Kernel,
                 num_inducing_points: int = 2,
                 likelihood: typing.Optional[GaussianLikelihood] = None,
                 normalize_y: bool = True,
                 n_opt_restarts: int = 10,
                 instance_features: typing.Optional[np.ndarray] = None,
                 pca_components: typing.Optional[int] = None,
                 ):
        """
        Partial Sparse Gaussian process model. It is composed of two models: an Exact GP to descirbe the data
        distribution inside a subregion and an Approximate GP to approxiamte the data distribution outside a
        subregion.

        The GP hyperparameterŝ are obtained by optimizing the marginal log likelihood and optimize with botorch

        Parameters
        ----------
        types : List[int]
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass [3, 0]. Note that we count starting from 0.
        bounds : List[Tuple[float, float]]
            bounds of input dimensions: (lower, uppper) for continuous dims; (n_cat, np.nan) for categorical dims
        seed : int
            Model seed.
        kernel : george kernel object
            Specifies the kernel that is used for all Gaussian Process
        num_inducing_points: int
            Number of inducing points
        likelihood: Optional[GaussianLikelihood]
            Likelihood values
        normalize_y : bool
            Zero mean unit variance normalization of the output values, when the model is a partial sparse GP model,
        n_opt_restart : int
            Number of restarts for GP hyperparameter optimization
        instance_features : np.ndarray (I, K)
            Contains the K dimensional instance features of the I different instances
        pca_components : float
            Number of components to keep when using PCA to reduce dimensionality of instance features. Requires to
            set n_feats (> pca_dims).
        """
        super(PartialSparseGaussianProcess, self).__init__(configspace=configspace,
                                                           types=types,
                                                           bounds=bounds,
                                                           seed=seed,
                                                           kernel=kernel,
                                                           likelihood=likelihood,
                                                           normalize_y=normalize_y,
                                                           n_opt_restarts=n_opt_restarts,
                                                           instance_features=instance_features,
                                                           pca_components=pca_components,
                                                           )
        self.bounds_cont = bounds_cont,
        self.bounds_cat = bounds_cat,
        self.num_inducing_points = num_inducing_points

    def update_attribute(self, **kwargs: typing.Any):
        for key in kwargs:
            if not hasattr(self, key):
                raise ValueError(f"{self.__name__} has no attribute named {key}")
            setattr(self, key, kwargs[key])

    def _train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True) -> 'GaussianProcess':
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
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        """
        X = self._impute_inactive(X)
        if len(y.shape) == 1:
            self.n_objectives_ = 1
        else:
            self.n_objectives_ = y.shape[1]
        if self.n_objectives_ == 1:
            y = y.flatten()

        ss_data_indices = check_points_in_ss(X,
                                             cont_dims=self.cont_dims,
                                             cat_dims=self.cat_dims,
                                             bounds_cont=self.bound_cont,
                                             bounds_cat=self.bound_cat)
        if np.sum(ss_data_indices) > np.shape(y)[0] - self.num_inducing_points:
            if self.normalize_y:
                y = self._normalize_y(y)
            self.num_points = np.shape(y)[0]
            get_gp_kwargs = {'in_x': X, 'in_y': y, 'out_x': None, 'out_y': None}
        else:
            in_x = X[ss_data_indices]
            in_y = y[ss_data_indices]
            out_x = X[~ss_data_indices]
            out_y = y[~ss_data_indices]
            self.num_points = np.shape(in_y)[0]
            if self.normalize_y:
                in_y = self._normalize_y(in_y)
            out_y = (out_y - self.mean_y_) / self.std_y_
            get_gp_kwargs = {'in_x': in_x, 'in_y': in_y, 'out_x': out_x, 'out_y': out_y}

        n_tries = 10

        for i in range(n_tries):
            try:
                self.gp = self._get_gp(**get_gp_kwargs)
                break
            except np.linalg.LinAlgError as e:
                if i == n_tries:
                    raise e

        if do_optimize:
            # self._all_priors = self._get_all_priors(add_bound_priors=False)
            self.hypers = self._optimize()
            self.gp = set_params_with_array(self.gp, self.hypers, self.property_dict)
            if isinstance(self.gp.model, PartailSparseGPModel):

                self.gp.model.deactivate_kernel_grad()

                inducing_points = torch.from_numpy(pyDOE.lhs(n=out_x.shape[-1], samples=self.num_inducing_points))

                kernel = self.gp.model.base_covar
                var_gp = VariationalGPModel(kernel, inducing_points=inducing_points)

                out_x_ = torch.from_numpy(out_x)
                out_y_ = torch.from_numpy(out_y)

                variational_ngd_optimizer = gpytorch.optim.NGD(var_gp.variational_parameters(), num_data=out_y_.size(0),
                                                               lr=0.1)

                var_gp.train()
                likelihood = GaussianLikelihood().double()
                likelihood.train()

                mll_func = gpytorch.mlls.PredictiveLogLikelihood

                var_mll = mll_func(likelihood, var_gp, num_data=out_y_.size(0))

                for t in var_gp.variational_parameters():
                    t.requires_grad = False

                x0, property_dict, bounds = module_to_array(module=var_mll)
                for t in var_gp.variational_parameters():
                    t.requires_grad = True
                bounds = np.asarray(bounds).transpose().tolist()

                start_points = [x0]

                inducing_idx = 0
                inducing_size = out_x.shape[-1] * self.num_inducing_points
                for p_name, attrs in property_dict.items():
                    if p_name != "model.variational_strategy.inducing_points":
                        # Construct the new tensor
                        if len(attrs.shape) == 0:  # deal with scalar tensors
                            inducing_idx = inducing_idx + 1
                        else:
                            inducing_idx = inducing_idx + np.prod(attrs.shape)
                    else:
                        break
                while len(start_points) < 3:
                    new_start_point = np.random.rand(*x0.shape)
                    new_inducing_points = torch.from_numpy(
                        pyDOE.lhs(n=out_x.shape[-1], samples=self.num_inducing_points)).flatten()
                    new_start_point[inducing_idx: inducing_idx + inducing_size] = new_inducing_points
                    start_points.append(new_start_point)

                def sci_opi_wrapper(x, mll, property_dict, train_inputs, train_targets):
                    # A modification of from botorch.optim.utils._scipy_objective_and_grad,
                    # THe key difference is that we do an additional nature gradient update here
                    variational_ngd_optimizer.zero_grad()

                    mll = set_params_with_array(mll, x, property_dict)
                    mll.zero_grad()
                    try:  # catch linear algebra errors in gpytorch
                        output = mll.model(train_inputs)
                        args = [output, train_targets] + _get_extra_mll_args(mll)
                        loss = -mll(*args).sum()
                    except RuntimeError as e:
                        if isinstance(e, NanError) or "singular" in e.args[0]:
                            return float("nan"), np.full_like(x, "nan")
                        else:
                            raise e  # pragma: nocover
                    loss.backward()
                    variational_ngd_optimizer.step()
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

                theta_star = x0
                f_opt_star = np.inf
                for start_point in start_points:
                    try:
                        theta, f_opt, res_dict = optimize.fmin_l_bfgs_b(sci_opi_wrapper,
                                                                        start_point,
                                                                        args=(var_mll, property_dict, out_x_, out_y_),
                                                                        bounds=bounds,
                                                                        maxiter=50,
                                                                        )
                        if f_opt < f_opt_star:
                            f_opt_star = f_opt
                            theta_star = theta
                    except Exception as e:
                        self.logger.warning(f"An exception {e} occurs during the optimizaiton")

                start_idx = 0
                # modification on botorch.optim.numpy_converter.set_params_with_array
                for p_name, attrs in property_dict.items():
                    if p_name != "model.variational_strategy.inducing_points":
                        # Construct the new tensor
                        if len(attrs.shape) == 0:  # deal with scalar tensors
                            start_idx = start_idx + 1
                        else:
                            start_idx = start_idx + np.prod(attrs.shape)
                    else:
                        end_idx = start_idx + np.prod(attrs.shape)
                        inducing_points = torch.tensor(
                            theta_star[start_idx:end_idx], dtype=attrs.dtype, device=attrs.device
                        ).view(*attrs.shape)
                        break
                # set inducing points for covariance module here
                self.gp_model.initialize(**{'covar_module.inducing_points': inducing_points})
        else:
            self.hypers, self.property_dict, _ = module_to_array(module=self.gp)

        self.is_trained = True
        return self

    def _get_gp(self,
                in_x: typing.Optional[np.ndarray] = None,
                in_y: typing.Optional[np.ndarray] = None,
                out_x: typing.Optional[np.ndarray] = None,
                out_y: typing.Optional[np.ndarray] = None) -> typing.Optional[ExactMarginalLogLikelihood]:
        """
        Construction a new GP model based on the inputs
        If both in and out are None: return an empty models
        If only in_x and in_y are given: return a vanilla GP model
        If in_x, in_y, out_x, out_y are given: return a partial sparse GP model.

        Parameters
        ----------
        in_x: Optional[np.ndarray (N_in, D)]
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        in_y: Optional[np.ndarray (N,)]
            The corresponding target values.
        out_x: Optional[np.ndarray (N_out, D)]
            If set to true the hyperparameters are optimized otherwise
            the default hyperparameters of the kernel are used.
        out_y: typing.Optional[np.ndarray (N_out)] = None
        """
        if in_x is None:
            return None

        in_x = torch.from_numpy(in_x)
        in_y = torch.from_numpy(in_y)
        if out_x is None:
            self.gp_model = ExactGPModel(in_x, in_y, likelihood=self.likelihood, base_covar_kernel=self.kernel).double()
        else:
            out_x = torch.from_numpy(out_x)
            out_y = torch.from_numpy(out_y)

            if self.num_inducing_points <= in_y.shape[0]:
                weights = torch.ones(in_y.shape[0]) / in_y.shape[0]
                inducing_points = in_x[torch.multinomial(weights, self.num_inducing_points)]
            else:
                weights = torch.ones(out_y.shape[0]) / out_y.shape[0]
                inducing_points = out_x[torch.multinomial(weights, self.num_inducing_points - in_y.shape[0])]
                inducing_points = torch.cat([inducing_points, in_x])
            self.gp_model = PartailSparseGPModel(in_x, in_y, out_x, out_y,
                                                 likelihood=self.likelihood,
                                                 base_covar_kernel=self.kernel,
                                                 inducing_points=inducing_points).double()

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
        if isinstance(self.gp_model, PartailSparseGPModel):
            self.gp_model.deactivate_inducing_points_grad()

        x0, property_dict, bounds = module_to_array(module=self.gp)

        self.property_dict = property_dict

        bounds = np.asarray(bounds).transpose().tolist()
        p0 = [x0]
        n_tries = 5000
        for i in range(n_tries):
            try:
                self.gp.pyro_sample_from_prior()
                sample, _, _ = module_to_array(module=self.gp)
                p0.append(sample.astype(np.float64))
            except Exception as e:
                if i == n_tries - 1:
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
                theta, f_opt, res_dict = optimize.fmin_l_bfgs_b(
                    _scipy_objective_and_grad,
                    start_point,
                    args=(self.gp, property_dict),
                    bounds=bounds,
                )

            except RuntimeError as e:
                self.logger.warning(f"Fail to optimize as an Error occurs: {e}")
                continue
            if f_opt < f_opt_star:
                f_opt_star = f_opt
                theta_star = theta

        return theta_star


def check_points_in_ss(X: np.ndarray,
                       cont_dims: np.ndarray,
                       cat_dims: np.ndarray,
                       bounds_cont: np.ndarray,
                       bounds_cat: typing.List[typing.List[typing.Tuple]],
                       ):
    """
    check which points will be included in the subspace
    Parameters
    ----------
    X: np.ndarray(N,D),
        points to be checked
    cont_dims: np.ndarray(D_cont)
        dimensions of the continuous hyperparameters
    cat_dims: np.ndarray(D_cat)
        dimensions of the categorical hyperparameters
    bounds_cont: typing.List[typing.Tuple]
        subspaces bounds of categorical hyperparameters, its length is the number of categorical hyperparameters
    bounds_cat: np.ndarray(D_cont, 2)
        subspaces bounds of continuous hyperparameters, its length is the number of categorical hyperparameters
    Return
    ----------
    indices_in_ss:np.ndarray(N)
        indices of data that included in subspaces
    """
    if len(X.shape) == 1:
        X = X[np.newaxis, :]

    if cont_dims.size != 0:
        data_in_ss = np.all(X[:, cont_dims] <= bounds_cont[:, 1], axis=1) & \
                     np.all(X[:, cont_dims] >= bounds_cont[:, 0], axis=1)

        bound_left = bounds_cont[:, 0] - np.min(X[data_in_ss][:, cont_dims] - bounds_cont[:, 0], axis=0)
        bound_right = bounds_cont[:, 1] + np.min(bounds_cont[:, 1] - X[data_in_ss][:, cont_dims], axis=0)
        data_in_ss = np.all(X[:, cont_dims] <= bound_right, axis=1) & \
                     np.all(X[:, cont_dims] >= bound_left, axis=1)
    else:
        data_in_ss = np.ones(X.shape[-1], dtype=bool)

    # TODO find out where cause the None value of  bounds_cat
    if bounds_cat == None:
        bounds_cat = [()]

    for bound_cat, cat_dim in zip(bounds_cat, cat_dims):
        data_in_ss &= np.in1d(X[:, cat_dim], bound_cat)


    # indices_in_ss = np.where(in_ss_dims)[0]
    return data_in_ss
