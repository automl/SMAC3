# from __future__ import annotations

# import logging
# import warnings
# from collections import OrderedDict
# from typing import Any, TypeVar


# import gpytorch
# import numpy as np
# from smac.model.gaussian_process.utils.gpytorch import ExactGaussianProcessModel
# import torch
# from botorch.optim.numpy_converter import module_to_array, set_params_with_array  # noqa
# from botorch.optim.utils import _get_extra_mll_args, _scipy_objective_and_grad  # noqa
# from gpytorch.constraints.constraints import Interval  # noqa
# from gpytorch.distributions.multivariate_normal import MultivariateNormal  # noqa
# from gpytorch.kernels import Kernel  # noqa
# from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood  # noqa
# from gpytorch.means import ZeroMean  # noqa
# from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa
# from gpytorch.models import ExactGP  # noqa
# from gpytorch.priors import HorseshoePrior  # noqa
# from gpytorch.utils.errors import NanError, NotPSDError  # noqa
# from scipy import optimize
# from scipy.stats.qmc import LatinHypercube  # noqa
# import logging
# import warnings
# from collections import OrderedDict
# from typing import Any, TypeVar

# """
# import gpytorch
# import numpy as np
# import torch
# from botorch.optim.numpy_converter import module_to_array, set_params_with_array
# from botorch.optim.utils import _get_extra_mll_args, _scipy_objective_and_grad
# from gpytorch.constraints.constraints import Interval
# from gpytorch.distributions.multivariate_normal import MultivariateNormal
# from gpytorch.kernels import Kernel
# from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
# from gpytorch.means import ZeroMean
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from gpytorch.models import ExactGP
# from gpytorch.priors import HorseshoePrior
# from gpytorch.utils.errors import NanError, NotPSDError
# from scipy import optimize
# from scipy.stats.qmc import LatinHypercube
# """

# from ConfigSpace import ConfigurationSpace
# from smac.constants import VERY_SMALL_NUMBER
# from smac.model.gaussian_process.abstract_gaussian_process import AbstractGaussianProcess
# from smac.model.utils import check_subspace_points
# from smac.model.gaussian_process.kernels._boing import FITCKernel, FITCMean


# # from smac.model.utils import check_subspace_points  # noqa
# from smac.model.gaussian_process.kernels._boing import FITCKernel, FITCMean  # noqa

# warnings.filterwarnings("ignore", module="gpytorch")

# logger = logging.getLogger(__name__)


# Self = TypeVar("Self", bound="GPyTorchGaussianProcess")


# class GPyTorchGaussianProcess(AbstractGaussianProcess):
#     """A Gaussian process written with GPyTorch. The interface is written to be compatible with partial sparse
# gaussian
#     process.

#     Parameters
#     ----------
#     configspace : ConfigurationSpace
#     kernel : Kernel
#         Kernel which is used for the Gaussian process.
#     n_restarts : int, defaults to 10
#         Number of restarts for the Gaussian process hyperparameter optimization.
#     normalize_y : bool, defaults to True
#         Zero mean unit variance normalization of the output values.
#     likelihood : FixedNoiseGaussianLikelihood | None, defaults to None
#         The Gaussian likelihood (or noise).
#     instance_features : dict[str, list[int | float]] | None, defaults to None
#         Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
#         on which the model is trained on.
#     pca_components : float, defaults to 7
#         Number of components to keep when using PCA to reduce dimensionality of instance features.
#     seed : int
#     """

#     def __init__(
#         self,
#         configspace: ConfigurationSpace,
#         kernel: Kernel,
#         n_restarts: int = 10,
#         normalize_y: bool = True,
#         likelihood: FixedNoiseGaussianLikelihood | None = None,
#         instance_features: dict[str, list[int | float]] | None = None,
#         pca_components: int | None = 7,
#         seed: int = 0,
#     ):
#         if n_restarts <= 0:
#             raise ValueError("The argument `n_restarts` needs to be positive.")

#         super(GPyTorchGaussianProcess, self).__init__(
#             configspace=configspace,
#             kernel=kernel,
#             instance_features=instance_features,
#             pca_components=pca_components,
#             seed=seed,
#         )

#         if likelihood is None:
#             noise_prior = HorseshoePrior(0.1)
#             likelihood = GaussianLikelihood(
#                 noise_prior=noise_prior,
#                 noise_constraint=Interval(
#                     np.exp(-25),
#                     np.exp(2),
#                     transform=None,
#                 ),
#             ).double()

#         self._likelihood = likelihood
#         self._normalize_y = normalize_y
#         self._n_restarts = n_restarts
#         self._hypers = np.empty((0,))
#         self._property_dict: OrderedDict = OrderedDict()
#         self._is_trained = False

#     @property
#     def meta(self) -> dict[str, Any]:  # noqa: D102
#         meta = super().meta
#         meta.update(
#             {
#                 "n_restarts": self._n_restarts,
#                 "normalize_y": self._normalize_y,
#                 "likelihood": self._likelihood.__data__,
#             }
#         )

#         return meta

#     def _train(
#         self: Self,
#         X: np.ndarray,
#         y: np.ndarray,
#         optimize_hyperparameters: bool = True,
#     ) -> Self:
#         """Computes the Cholesky decomposition of the covariance of X and estimates the GP
#         hyperparameters by optimizing the marginal log likelihood. The prior mean of the GP is set to
#         the empirical mean of X.

#         Parameters
#         ----------
#         X : np.ndarray [#samples, #hyperparameter + #features]
#             Input data points.
#         Y : np.ndarray [#samples, #objectives]
#             The corresponding target values.
#         optimize_hyperparameters: boolean
#             If set to true the hyperparameters are optimized otherwise the default hyperparameters of the kernel are
#             used.
#         """
#         X = self._impute_inactive(X)

#         if len(y.shape) == 1:
#             self._n_objectives = 1
#         else:
#             self._n_objectives = y.shape[1]

#         if self._n_objectives == 1:
#             y = y.flatten()

#         if self._normalize_y:
#             y = self._normalize(y)

#         n_tries = 10
#         for i in range(n_tries):
#             try:
#                 self._gp = self._get_gaussian_process(X, y)
#                 break
#             except Exception as e:
#                 if i == n_tries - 1:
#                     # To avoid Endless loop, we need to stop it when we have n_tries unsuccessful tries.
#                     raise e

#         if optimize_hyperparameters:
#             self._hypers = self._optimize()
#             self._gp = set_params_with_array(self._gp, self._hypers, self._property_dict)
#         else:
#             self._hypers, self._property_dict, _ = module_to_array(module=self._gp)

#         self._is_trained = True
#         return self

#     def _get_gaussian_process(
#         self, X: np.ndarray | None = None, y: np.ndarray | None = None
#     ) -> ExactMarginalLogLikelihood | None:
#         """
#         Get the GP model with the given X and y values. As GPyTorch requires the input data to initialize a new
#         model, we also pass X and y here. X and y are set optional to ensure compatibility.

#         Parameters
#         ----------
#         X : np.ndarray [#samples, #hyperparameter + #features]
#             Input data points.
#         Y : np.ndarray [#samples, #objectives]
#             The corresponding target values.

#         Returns
#         -------
#         mll : ExactMarginalLogLikelihood | None
#             A GPyTorch model with zero mean and user specified covariance.
#         """
#         if X is None:
#             # To be compatible with the base model
#             return None

#         X = torch.from_numpy(X)
#         y = torch.from_numpy(y)
#         self._gp_model = ExactGaussianProcessModel(
#             X, y, likelihood=self._likelihood, base_covar_kernel=self._kernel
#         ).double()

#         mll = ExactMarginalLogLikelihood(self._likelihood, self._gp_model)
#         mll.double()

#         return mll

#     def _optimize(self) -> np.ndarray:
#         """Optimizes the marginal log likelihood and returns the best found hyperparameter
#         configuration theta.

#         Returns
#         -------
#         theta : np.ndarray
#             Hyperparameter vector that maximizes the marginal log likelihood.
#         """
#         x0, property_dict, bounds = module_to_array(module=self._gp)

#         bounds = np.asarray(bounds).transpose().tolist()

#         self._property_dict = property_dict

#         p0 = [x0]

#         # Avoid infinite sampling
#         n_tries = 5000
#         for i in range(n_tries):
#             try:
#                 gp_model = self._gp.pyro_sample_from_prior()
#                 x_out = []
#                 for key in property_dict.keys():
#                     param = gp_model
#                     param_names = key.split(".")
#                     for name in param_names:
#                         param = getattr(param, name)
#                     x_out.append(param.detach().view(-1).cpu().double().clone().numpy())
#                 sample = np.concatenate(x_out)
#                 p0.append(sample.astype(np.float64))
#             except Exception as e:
#                 if i == n_tries - 1:
#                     logger.debug(f"Fails to sample new hyperparameters because of {e}.")
#                     raise e
#                 continue
#             if len(p0) == self._n_restarts:
#                 break

#         self._gp_model.train()
#         self._likelihood.train()

#         theta_star = x0
#         f_opt_star = np.inf
#         for i, start_point in enumerate(p0):
#             try:
#                 theta, f_opt, _ = optimize.fmin_l_bfgs_b(
#                     _scipy_objective_and_grad,
#                     start_point,
#                     args=(self._gp, property_dict),
#                     bounds=bounds,
#                 )
#             except NotPSDError as e:
#                 logger.warning(f"Fail to optimize the GP hyperparameters as an Error occurs: {e}.")
#                 f_opt = np.inf
#                 theta = start_point

#             if f_opt < f_opt_star:
#                 f_opt_star = f_opt
#                 theta_star = theta

#         return theta_star

#     def _predict(
#         self,
#         X: np.ndarray,
#         covariance_type: str | None = "diagonal",
#     ) -> tuple[np.ndarray, np.ndarray | None]:
#         if not self._is_trained:
#             raise Exception("Model has to be trained first.")

#         X_test = torch.from_numpy(self._impute_inactive(X))
#         self._likelihood.eval()
#         self._gp_model.eval()

#         with torch.no_grad(), gpytorch.settings.fast_pred_var():

#             observed_pred = self._likelihood(self._gp_model(X_test))

#             mu = observed_pred.mean.numpy()
#             if covariance_type is None:
#                 var = None

#                 if self._normalize_y:
#                     mu = self._untransform_y(mu)

#             else:
#                 if covariance_type != "full":
#                     var = observed_pred.stddev.numpy()
#                     var = var**2  # since we get standard deviation for faster computation
#                 else:
#                     # output full covariance
#                     var = observed_pred.covariance_matrix.numpy()

#                 # Clip negative variances and set them to the smallest
#                 # positive float value
#                 var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

#                 if self._normalize_y:
#                     mu, var = self._untransform_y(mu, var)

#                 if covariance_type == "diagonal":
#                     var = np.sqrt(var)  # converting variance to std deviation if specified

#         return mu, var

#     def sample_functions(self, X_test: np.ndarray, n_funcs: int = 1) -> np.ndarray:
#         """Samples F function values from the current posterior at the N specified test points.

#         Parameters
#         ----------
#         X : np.ndarray [#samples, #hyperparameter + #features]
#             Input data points.
#         n_funcs: int
#             Number of function values that are drawn at each test point.

#         Returns
#         -------
#         function_samples : np.ndarray
#             The F function values drawn at the N test points.
#         """
#         if not self._is_trained:
#             raise Exception("Model has to be trained first!")
#         self._likelihood.eval()
#         self._gp_model.eval()

#         X_test = torch.from_numpy(self._impute_inactive(X_test))
#         with torch.no_grad():
#             funcs = self._likelihood(self._gp_model(X_test)).sample(torch.Size([n_funcs])).t().cpu().numpy()

#         if self._normalize_y:
#             funcs = self._untransform_y(funcs)

#         if len(funcs.shape) == 1:
#             return funcs[None, :]
#         else:
#             return funcs


# class GloballyAugmentedLocalGaussianProcess(GPyTorchGaussianProcess):
#     def __init__(
#         self,
#         configspace: ConfigurationSpace,
#         bounds_cont: np.ndarray,
#         bounds_cat: list[tuple],
#         kernel: Kernel,
#         num_inducing_points: int = 2,
#         likelihood: GaussianLikelihood | None = None,
#         normalize_y: bool = True,
#         n_restarts: int = 10,
#         instance_features: dict[str, list[int | float]] | None = None,
#         pca_components: int | None = 7,
#         seed: int = 0,
#     ):
#         """
#         The GP hyperparameters are obtained by optimizing the marginal log-likelihood and optimized with botorch
#         We train an LGPGA in two stages:
#         In the first stage, we only train the kernel hyperparameter and thus deactivate the gradient w.r.t the
# position
#         of the inducing points.
#         In the second stage, we use the kernel hyperparameter acquired in the first stage to initialize a new
#         variational Gaussian process and only optimize its inducing points' position with natural gradients.
#         Finally, we update the position of the inducing points and use it for evaluation.


#         Parameters
#         ----------
#         bounds_cont: np.ndarray(N_cont, 2),
#            bounds of the continuous hyperparameters, store as [[0,1] * N_cont]
#         bounds_cat: List[Tuple],
#            bounds of categorical hyperparameters
#         kernel : gpytorch kernel object
#            Specifies the kernel that is used for all Gaussian Process
#         num_inducing_points: int
#            Number of inducing points
#         likelihood: Optional[GaussianLikelihood]
#            Likelihood values
#         normalize_y : bool
#            Zero mean unit variance normalization of the output values when the model is a partial sparse GP model.
#         """
#         super(GloballyAugmentedLocalGaussianProcess, self).__init__(
#             configspace=configspace,
#             kernel=kernel,
#             likelihood=likelihood,
#             normalize_y=normalize_y,
#             n_restarts=n_restarts,
#             instance_features=instance_features,
#             pca_components=pca_components,
#             seed=seed,
#         )
#         self.cont_dims = np.where(np.array(self.types) == 0)[0]
#         self.cat_dims = np.where(np.array(self.types) != 0)[0]
#         self.bounds_cont = bounds_cont
#         self.bounds_cat = bounds_cat
#         self.num_inducing_points = num_inducing_points

#     def update_attribute(self, **kwargs: Any) -> None:
#         """We update the class attribute (for instance, number of inducing points)"""
#         for key in kwargs:
#             if not hasattr(self, key):
#                 raise AttributeError(f"{self.__class__.__name__} has no attribute named {key}")
#             setattr(self, key, kwargs[key])

#     def _train(
#         self, X: np.ndarray, y: np.ndarray, do_optimize: bool = True
#     ) -> AugmentedLocalGaussianProcess | GPyTorchGaussianProcess:
#         """
#         Update the hyperparameters of the partial sparse kernel. Depending on the number of inputs inside and
#         outside the subregion, we initialize a  PartialSparseGaussianProcess or a GaussianProcessGPyTorch

#         Parameters
#         ----------
#         X: np.ndarray (N, D)
#             Input data points. The dimensionality of X is (N, D),
#             with N as the number of points and D is the number of features., N = N_in + N_out
#         y: np.ndarray (N,)
#             The corresponding target values.
#         do_optimize: boolean
#                 If set to true, the hyperparameters are optimized otherwise,
#                 the default hyperparameters of the kernel are used.
#         """
#         X = self._impute_inactive(X)
#         if len(y.shape) == 1:
#             self._n_objectives = 1
#         else:
#             self._n_objectives = y.shape[1]
#         if self._n_objectives == 1:
#             y = y.flatten()

#         ss_data_indices = check_subspace_points(
#             X,
#             cont_dims=self.cont_dims,
#             cat_dims=self.cat_dims,
#             bounds_cont=self.bounds_cont,
#             bounds_cat=self.bounds_cat,
#             expand_bound=True,
#         )

#         if np.sum(ss_data_indices) > np.shape(y)[0] - self.num_inducing_points:
#             # we initialize a vanilla GaussianProcessGPyTorch
#             if self._normalize_y:
#                 y = self._normalize_y(y)
#             self.num_points = np.shape(y)[0]
#             get_gp_kwargs = {"X_in": X, "y_in": y, "X_out": None, "y_out": None}
#         else:
#             # we initialize a PartialSparseGaussianProcess object
#             X_in = X[ss_data_indices]
#             y_in = y[ss_data_indices]
#             X_out = X[~ss_data_indices]
#             y_out = y[~ss_data_indices]
#             self.num_points = np.shape(y_in)[0]
#             if self._normalize_y:
#                 y_in = self._normalize_y(y_in)
#                 y_out = (y_out - self.mean_y_) / self.std_y_
#             get_gp_kwargs = {"X_in": X_in, "y_in": y_in, "X_out": X_out, "y_out": y_out}

#         n_tries = 10

#         for i in range(n_tries):
#             try:
#                 self._gp = self._get_gaussian_process(**get_gp_kwargs)
#                 break
#             except Exception as e:
#                 if i == n_tries - 1:
#                     raise RuntimeError(f"Fails to initialize a GP model, {e}")

#         if do_optimize:
#             self._hypers = self._optimize()
#             self._gp = set_params_with_array(self._gp, self._hypers, self._property_dict)
#             if isinstance(self._gp.model, AugmentedLocalGaussianProcess):
#                 # we optimize the position of the inducing points and thus needs to deactivate the gradient of kernel
#                 # hyperparameters
#                 lhd = LatinHypercube(d=X.shape[-1], seed=self.rng.randint(0, 1000000))

#                 inducing_points = torch.from_numpy(lhd.random(n=self.num_inducing_points))

#                 kernel = self._gp.model.base_covar
#                 var_gp = VariationalGaussianProcess(kernel, X_inducing=inducing_points)

#                 X_out_ = torch.from_numpy(X_out)
#                 y_out_ = torch.from_numpy(y_out)

#                 variational_ngd_optimizer = gpytorch.optim.NGD(
#                     var_gp.variational_parameters(), num_data=y_out_.size(0), lr=0.1
#                 )

#                 var_gp.train()
#                 likelihood = GaussianLikelihood().double()
#                 likelihood.train()

#                 mll_func = gpytorch.mlls.PredictiveLogLikelihood

#                 var_mll = mll_func(likelihood, var_gp, num_data=y_out_.size(0))

#                 for t in var_gp.variational_parameters():
#                     t.requires_grad = False

#                 x0, property_dict, bounds = module_to_array(module=var_mll)
#                 for t in var_gp.variational_parameters():
#                     t.requires_grad = True
#                 bounds = np.asarray(bounds).transpose().tolist()

#                 start_points = [x0]

#                 inducing_idx = 0

#                 inducing_size = X_out.shape[-1] * self.num_inducing_points
#                 for p_name, attrs in property_dict.items():
#                     if p_name != "model.variational_strategy.inducing_points":
#                         # Construct the new tensor
#                         if len(attrs.shape) == 0:  # deal with scalar tensors
#                             inducing_idx = inducing_idx + 1
#                         else:
#                             inducing_idx = inducing_idx + np.prod(attrs.shape)
#                     else:
#                         break
#                 while len(start_points) < 3:
#                     new_start_point = np.random.rand(*x0.shape)
#                     new_inducing_points = torch.from_numpy(lhd.random(n=self.num_inducing_points)).flatten()
#                     new_start_point[inducing_idx : inducing_idx + inducing_size] = new_inducing_points
#                     start_points.append(new_start_point)

#                 def sci_opi_wrapper(
#                     x: np.ndarray,
#                     mll: gpytorch.module,
#                     property_dict: dict,
#                     train_inputs: torch.Tensor,
#                     train_targets: torch.Tensor,
#                 ) -> tuple[float, np.ndarray]:
#                     """
#                     A modification of from botorch.optim.utils._scipy_objective_and_grad, the key difference is that
#                     we do an additional natural gradient update before computing the gradient values
#                     Parameters
#                     ----------
#                     x: np.ndarray
#                         optimizer input
#                     mll: gpytorch.module
#                         a gpytorch module whose hyperparameters are defined by x
#                     property_dict: Dict
#                         a dict describing how x is mapped to initialize mll
#                     train_inputs: torch.Tensor (N_input, D)
#                         input points of the GP model
#                     train_targets: torch.Tensor (N_input, 1)
#                         target value of the GP model
#                     Returns
#                     ----------
#                     loss: np.ndarray
#                         loss value
#                     grad: np.ndarray
#                         gradient w.r.t. the inputs
#                     ----------
#                     """
#                     # A modification of from botorch.optim.utils._scipy_objective_and_grad:
#                     # https://botorch.org/api/_modules/botorch/optim/utils.html
#                     # The key difference is that we do an additional natural gradient update here
#                     variational_ngd_optimizer.zero_grad()

#                     mll = set_params_with_array(mll, x, property_dict)
#                     mll.zero_grad()
#                     try:  # catch linear algebra errors in gpytorch
#                         output = mll.model(train_inputs)
#                         args = [output, train_targets] + _get_extra_mll_args(mll)
#                         loss = -mll(*args).sum()
#                     except RuntimeError as e:
#                         if isinstance(e, NanError) or "singular" in e.args[0]:
#                             return float("nan"), np.full_like(x, "nan")
#                         else:
#                             raise e  # pragma: nocover
#                     loss.backward()
#                     variational_ngd_optimizer.step()
#                     param_dict = OrderedDict(mll.named_parameters())
#                     grad = []
#                     for p_name in property_dict:
#                         t = param_dict[p_name].grad
#                         if t is None:
#                             # this deals with parameters that do not affect the loss
#                             grad.append(np.zeros(property_dict[p_name].shape.numel()))
#                         else:
#                             grad.append(t.detach().view(-1).cpu().double().clone().numpy())
#                     mll.zero_grad()
#                     return loss.item(), np.concatenate(grad)

#                 theta_star = x0
#                 f_opt_star = np.inf
#                 for start_point in start_points:
#                     try:
#                         theta, f_opt, res_dict = optimize.fmin_l_bfgs_b(
#                             sci_opi_wrapper,
#                             start_point,
#                             args=(var_mll, property_dict, X_out_, y_out_),
#                             bounds=bounds,
#                             maxiter=50,
#                         )
#                         if f_opt < f_opt_star:
#                             f_opt_star = f_opt
#                             theta_star = theta
#                     except Exception as e:
#                         logger.warning(f"An exception {e} occurs during the optimizaiton")

#                 start_idx = 0
#                 # modification on botorch.optim.numpy_converter.set_params_with_array as we only need to extract the
#                 # positions of inducing points
#                 for p_name, attrs in property_dict.items():
#                     if p_name != "model.variational_strategy.inducing_points":
#                         # Construct the new tensor
#                         if len(attrs.shape) == 0:  # deal with scalar tensors
#                             start_idx = start_idx + 1
#                         else:
#                             start_idx = start_idx + np.prod(attrs.shape)
#                     else:
#                         end_idx = start_idx + np.prod(attrs.shape)
#                         X_inducing = torch.tensor(
#                             theta_star[start_idx:end_idx], dtype=attrs.dtype, device=attrs.device
#                         ).view(*attrs.shape)
#                         break
#                 # set inducing points for covariance module here
#                 self._gp_model.set_augment_module(X_inducing)
#         else:
#             self._hypers, self._property_dict, _ = module_to_array(module=self._gp)

#         self._is_trained = True
#         return self

#     def _get_gaussian_process(
#         self,
#         X_in: np.ndarray | None = None,
#         y_in: np.ndarray | None = None,
#         X_out: np.ndarray | None = None,
#         y_out: np.ndarray | None = None,
#     ) -> ExactMarginalLogLikelihood | None:
#         """
#         Construct a new GP model based on the inputs
#         If both in and out are None: return an empty model
#         If only in_x and in_y are given: return a vanilla GP model
#         If in_x, in_y, out_x, out_y are given: return a partial sparse GP model.

#         Parameters
#         ----------
#         X_in: np.ndarray (N_in, D) | None
#             Input data points inside the subregion. The dimensionality of X_in is (N_in, D),
#             with N_in as the number of points inside the subregion and D is the number of features. If it is not
# given,
#             this function will return None to be compatible with the implementation of its parent class
#         y_in: np.ndarray (N_in,) | None
#             The corresponding target values inside the subregion.
#         X_out: np.ndarray (N_out, D) | None
#             Input data points outside the subregion. The dimensionality of X_out is (N_out, D). If it is not given,
# this
#         function will return a vanilla Gaussian Process
#         y_out: np.ndarray (N_out) | None
#             The corresponding target values outside the subregion.

#         Returns
#         -------
#         mll: ExactMarginalLogLikelihood
#             a gp module
#         """
#         if X_in is None:
#             return None

#         X_in = torch.from_numpy(X_in)
#         y_in = torch.from_numpy(y_in)
#         if X_out is None:
#             self._gp_model = ExactGaussianProcessModel(
#                 X_in, y_in, likelihood=self._likelihood, base_covar_kernel=self.kernel
#             ).double()
#         else:
#             X_out = torch.from_numpy(X_out)
#             y_out = torch.from_numpy(y_out)

#             self._gp_model = AugmentedLocalGaussianProcess(
#                 X_in, y_in, X_out, y_out, likelihood=self._likelihood, base_covar_kernel=self.kernel  # type:ignore
#             ).double()
#         mll = ExactMarginalLogLikelihood(self._likelihood, self._gp_model)
#         mll.double()
#         return mll


# class ExactGaussianProcessModel(ExactGP):
#     """Exact GP model that serves as a backbone for `GPyTorchGaussianProcess`."

#     Parameters
#     ----------
#     train_X: torch.tenor
#         input feature
#     train_y: torch.tensor
#         input observations
#     base_covar_kernel: Kernel
#         covariance kernel used to compute covariance matrix
#     likelihood: GaussianLikelihood
#         GP likelihood
#     """

#     def __init__(
#         self,
#         train_X: torch.Tensor,
#         train_y: torch.Tensor,
#         base_covar_kernel: Kernel,
#         likelihood: GaussianLikelihood,
#     ):
#         super(ExactGaussianProcessModel, self).__init__(train_X, train_y, likelihood)

#         # In our experiments we find that ZeroMean more robust than ConstantMean when y is normalized
#         self._mean_module = ZeroMean()
#         self._covar_module = base_covar_kernel

#     def forward(self, x: torch.Tensor) -> MultivariateNormal:
#         """Computes the posterior mean and variance."""
#         mean_x = self._mean_module(x)
#         covar_x = self._covar_module(x)

#         return MultivariateNormal(mean_x, covar_x)


# class AugmentedLocalGaussianProcess(ExactGP):
#     def __init__(
#         self,
#         X_in: torch.Tensor,
#         y_in: torch.Tensor,
#         X_out: torch.Tensor,
#         y_out: torch.Tensor,
#         likelihood: GaussianLikelihood,
#         base_covar_kernel: Kernel,
#     ):
#         """
#         An Augmented Local GP, it is trained with the points inside a subregion while its prior is augemented by the
#         points outside the subregion (global configurations)

#         Parameters
#         ----------
#         X_in: torch.Tensor (N_in, D),
#             feature vector of the points inside the subregion
#         y_in: torch.Tensor (N_in, 1),
#             observation inside the subregion
#         X_out: torch.Tensor (N_out, D),
#             feature vector  of the points outside the subregion
#         y_out:torch.Tensor (N_out, 1),
#             observation inside the subregion
#         likelihood: GaussianLikelihood,
#             likelihood of the GP (noise)
#         base_covar_kernel: Kernel,
#             Covariance Kernel
#         """
#         X_in = X_in.unsqueeze(-1) if X_in.ndimension() == 1 else X_in
#         X_out = X_out.unsqueeze(-1) if X_out.ndimension() == 1 else X_out
#         assert X_in.shape[-1] == X_out.shape[-1]

#         super(AugmentedLocalGaussianProcess, self).__init__(X_in, y_in, likelihood)

#         self._mean_module = ZeroMean()
#         self.base_covar = base_covar_kernel

#         self.X_out = X_out
#         self.y_out = y_out
#         self.augmented = False

#     def set_augment_module(self, X_inducing: torch.Tensor) -> None:
#         """
#         Set an augmentation module, which will be used later for inference

#         Parameters
#         ----------
#         X_inducing: torch.Tensor(N_inducing, D)
#            inducing points, it needs to have the same number of dimensions as X_in
#         """
#         X_inducing = X_inducing.unsqueeze(-1) if X_inducing.ndimension() == 1 else X_inducing
#         # assert X_inducing.shape[-1] == self.X_out.shape[-1]
#         self.covar_module = FITCKernel(
#             self.base_covar, X_inducing=X_inducing, X_out=self.X_out, y_out=self.y_out, likelihood=self._likelihood
#         )
#         self.mean_module = FITCMean(covar_module=self.covar_module)
#         self.augmented = True

#     def forward(self, x: torch.Tensor) -> MultivariateNormal:
#         """
#         Compute the prior values. If optimize_kernel_hps is set True in the training phases, this model degenerates to
#         a vanilla GP model with ZeroMean and base_covar as covariance matrix. Otherwise, we apply partial sparse GP
#         mean and kernels here.
#         """
#         if not self.augmented:
#             # we only optimize for kernel hyperparameters
#             covar_x = self.base_covar(x)
#             mean_x = self._mean_module(x)
#         else:
#             covar_x = self.covar_module(x)
#             mean_x = self.mean_module(x)
#         return MultivariateNormal(mean_x, covar_x)


# class VariationalGaussianProcess(gpytorch.models.ApproximateGP):
#     """
#     A variational GP to compute the position of the inducing points.
#     We only optimize for the position of the continuous dimensions and keep the categorical dimensions constant.
#     """

#     def __init__(self, kernel: Kernel, X_inducing: torch.Tensor):
#         """
#         Initialize a Variational GP
#         we set the lower bound and upper bounds of inducing points for numerical hyperparameters between 0 and 1,
#         that is, we constrain the inducing points to lay inside the subregion.

#         Parameters
#         ----------
#         kernel: Kernel
#             kernel of the variational GP, its hyperparameter needs to be fixed when it is by LGPGA
#         X_inducing: torch.tensor (N_inducing, D)
#             inducing points
#         """
#         variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(X_inducing.size(0))
#         variational_strategy = gpytorch.variational.VariationalStrategy(
#             self, X_inducing, variational_distribution, learn_inducing_locations=True
#         )
#         super(VariationalGaussianProcess, self).__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ZeroMean()
#         self.covar_module = kernel

#         shape_X_inducing = X_inducing.shape
#         lower_X_inducing = torch.zeros([shape_X_inducing[-1]]).repeat(shape_X_inducing[0])
#         upper_X_inducing = torch.ones([shape_X_inducing[-1]]).repeat(shape_X_inducing[0])

#         self.variational_strategy.register_constraint(
#             param_name="inducing_points",
#             constraint=Interval(lower_X_inducing, upper_X_inducing, transform=None),
#         )
#         self.double()

#         for p_name, t in self.named_hyperparameters():
#             if p_name != "variational_strategy.inducing_points":
#                 t.requires_grad = False

#     def forward(self, x: torch.Tensor) -> MultivariateNormal:
#         """
#         Pass the posterior mean and variance given input X

#         Parameters
#         ----------
#         x: torch.Tensor
#             Input data
#         Returns
#         -------
#         """
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x, cont_only=True)
#         return MultivariateNormal(mean_x, covar_x)
