# from __future__ import annotations

# from typing import Any, Dict, Tuple

# import copy
# import math

# import gpytorch
# import numpy as np
# import torch
# from gpytorch import settings
# from gpytorch.kernels import Kernel, MaternKernel, ProductKernel, ScaleKernel
# from gpytorch.lazy import (
#     DiagLazyTensor,
#     MatmulLazyTensor,
#     PsdSumLazyTensor,
#     RootLazyTensor,
#     delazify,
# )
# from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.means.mean import Mean
# from gpytorch.utils.cholesky import psd_safe_cholesky
# from sklearn.gaussian_process.kernels import Kernel as SKLKernels

# from smac.model.gaussian_process.kernels import ConstantKernel, WhiteKernel


# class MixedKernel(ProductKernel):
#     """
#     A special form of ProductKernel. It is composed of a cont_kernel and a cat_kernel that work with continuous and
#     categorical parameters, respectively. Its forward pass allows an additional parameter to determine if only
#     cont_kernel is applied to the input.
#     """

#     def __init__(self, cont_kernel: Kernel, cat_kernel: Kernel):
#         kernels = cont_kernel.kernels if isinstance(cont_kernel, ProductKernel) else [cont_kernel]
#         kernels += cat_kernel.kernels if isinstance(cat_kernel, ProductKernel) else [cat_kernel]
#         super().__init__(*kernels)
#         self.cont_kernel = cont_kernel
#         self.cat_kernel = cat_kernel

#     def forward(
#         self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, cont_only: bool = False, **params: Any
#     ) -> gpytorch.lazy.LazyTensor:
#         """Compute kernel values, if cont_only is True, then the categorical kernel is omitted"""
#         if not cont_only:
#             return super().forward(x1, x2, diag, **params)
#         else:
#             return self.cont_kernel(x1, x2, diag, **params)


# def construct_gp_kernel(
#     kernel_kwargs: Dict[str, Any], cont_dims: np.ndarray, cat_dims: np.ndarray
# ) -> Kernel | SKLKernels:
#     """
#     Construct a GP kernel with the given kernel init argument, the cont_dims, and cat_dims of the problem. Since the
#     subspace might not have the same number of dimensions as the global search space.
#     We need to reconstruct the kernel every time when a new subspace is generated.

#     Parameters
#     ----------
#     kernel_kwargs: Dict[str, Any]
#         kernel kwargs. Arguments to initialize the kernels. It needs to contain the following items:
#             cont_kernel: type of continuous kernels
#             cont_kernel_kwargs: additional arguments for continuous kernels, for instance, length constraints and
# prior
#             cat_kernel: type of categorical kernels
#             cat_kernel_kwargs: additional arguments for categorical kernels, for instance, length constraints and
# prior
#             scale_kernel: type of scale kernels
#             scale_kernel_kwargs: additional arguments for scale kernels,  for instance, length constraints and prior
#     cont_dims: np.ndarray
#         dimensions of continuous hyperparameters
#     cat_dims: np.ndarray
#         dimensions of categorical hyperparameters
#     Returns
#     -------
#     kernel: Kernel | SKLKernels
#         constructed kernels

#     """
#     if len(cont_dims) > 0:
#         cont_kernel_class = kernel_kwargs.get("cont_kernel", MaternKernel)
#         cont_kernel_kwargs = kernel_kwargs.get("cont_kernel_kwargs", {})
#         cont_kernel = cont_kernel_class(
#             ard_num_dims=cont_dims.shape[-1], active_dims=tuple(cont_dims), **cont_kernel_kwargs
#         ).double()

#     if len(cat_dims) > 0:
#         cat_kernel_class = kernel_kwargs.get("cat_kernel", MaternKernel)
#         cat_kernel_kwargs = kernel_kwargs.get("cat_kernel_kwargs", {})
#         cat_kernel = cat_kernel_class(
#             ard_num_dims=cat_dims.shape[-1], active_dims=tuple(cat_dims), **cat_kernel_kwargs
#         ).double()

#     if len(cont_dims) > 0 and len(cat_dims) > 0:
#         if isinstance(cont_kernel, SKLKernels):
#             base_kernel = cont_kernel * cat_kernel
#         else:
#             base_kernel = MixedKernel(cont_kernel=cont_kernel, cat_kernel=cat_kernel)
#     elif len(cont_dims) > 0 and len(cat_dims) == 0:
#         base_kernel = cont_kernel
#     elif len(cont_dims) == 0 and len(cat_dims) > 0:
#         base_kernel = cat_kernel
#     else:
#         raise ValueError("Either cont_dims or cat_dims must exist!")

#     if isinstance(base_kernel, SKLKernels):
#         scale_kernel_class = kernel_kwargs.get("scale_kernel", ConstantKernel)
#         scale_kernel_kwargs = kernel_kwargs.get("scale_kernel_kwargs", {})
#         scale_kernel = scale_kernel_class(**scale_kernel_kwargs)

#         noise_kernel_class = kernel_kwargs.get("noise_kernel", WhiteKernel)
#         noise_kernel_kwargs = kernel_kwargs.get("noise_kernel_kwargs", {})
#         noise_kernel = noise_kernel_class(**noise_kernel_kwargs)

#         gp_kernel = scale_kernel * base_kernel + noise_kernel
#     else:
#         scale_kernel_class = kernel_kwargs.get("scale_kernel", ScaleKernel)
#         scale_kernel_kwargs = kernel_kwargs.get("scale_kernel_kwargs", {})
#         gp_kernel = scale_kernel_class(base_kernel=base_kernel, **scale_kernel_kwargs)

#     return gp_kernel


# class FITCKernel(Kernel):
#     def __init__(
#         self,
#         base_kernel: Kernel,
#         X_inducing: torch.Tensor,
#         likelihood: GaussianLikelihood,
#         X_out: torch.Tensor,
#         y_out: torch.Tensor,
#         active_dims: Tuple[int] | None = None,
#     ):
#         r"""A reimplementation of FITC Kernel that computes the posterior explicitly for globally augmented local GP.
#         This should work exactly the same as a gpytorch.kernel.InducingPointKernel.
#          However, it takes much less time when combined with LGPGA.
#          References: Edward Snelson and Zoubin Ghahramani. Sparse Gaussian processes using pseudo-inputs. Advances in
#          Neural Information Processing Systems 18, Cambridge, Massachusetts, 2006. The MIT Press.
#          https://papers.nips.cc/paper/2005/hash/4491777b1aa8b5b32c2e8666dbe1a495-Abstract.html

#         Mean value is computed with:
#         \mathbf{\mu_{l'}}  = \mathbf{K_{l',u} \Sigma K_{u,1} \Lambda}^{-1}\mathbf{y_g} \label{eq:mean_sgp}
#         and variance value:
#         \mathbf{\sigma}^2_{l'} = \mathbf{K_{l',l'}} - \mathbf{Q_{l', l'} + \mathbf{K_{l', u}\Sigma K_{u, l'}}}
#         \mathbf{\Sigma} = (\mathbf{K_{u,u}} + \mathbf{K_{u, g} \Lambda}^{-1}\mathbf{K_{g,u}})^{-1}
#         \mathbf{\Lambda} = diag[\mathbf{K_{g,g}-Q_{g,g}} + \sigma^2_{noise}\idenmat]
#         ----------
#         base_kernel: Kernel
#             base kernel function
#         X_inducing: torch.Tensor (N_inducing, D)
#             inducing points, a torch tensor with shape (N_inducing, D), N_inducing is the number of the inducing
# points
#         likelihood: GaussianLikelihood
#             GP likelihood
#         X_out: torch.Tensor (N_out,D)
#             data features outside the subregion, it needs to be of size (N_out, D), N_out is the number of points
#             outside the subspace
#         y_out: torch.Tensor
#             data observations outside the subregion
#         active_dims: Tuple[int] | None = None
#             Set this if you want to compute the covariance of only a few input dimensions. The ints
#             corresponds to the indices of the dimensions. Default: `None`.
#         """
#         super(FITCKernel, self).__init__(active_dims=active_dims)
#         self.has_lengthscale = base_kernel.has_lengthscale
#         self.base_kernel = base_kernel
#         self.likelihood = likelihood

#         if X_inducing.ndimension() == 1:
#             X_inducing = X_inducing.unsqueeze(-1)

#         self.X_out = X_out
#         self.y_out = y_out
#         self.register_parameter(name="X_inducing", parameter=torch.nn.Parameter(X_inducing))

#     def _clear_cache(self) -> None:
#         if hasattr(self, "_cached_kernel_mat"):
#             del self._cached_kernel_mat
#         if hasattr(self, "_cached_inducing_sigma"):
#             del self._cached_inducing_sigma
#         if hasattr(self, "_cached_poster_mean_mat"):
#             del self._cached_poster_mean_mat
#         if hasattr(self, "_train_cached_k_u1"):
#             del self._train_cached_k_u1
#         if hasattr(self, "_train_cached_lambda_diag_inv"):
#             del self._train_cached_lambda_diag_inv
#         if hasattr(self, "_train_cached_posterior_mean"):
#             del self._train_cached_posterior_mean
#         if hasattr(self, "_cached_kernel_inv_root"):
#             del self._cached_kernel_inv_root

#     @property
#     def _inducing_mat(self) -> torch.Tensor:
#         """
#         Computes inducing matrix, K(X_inducing, X_inducing)

#         Returns
#         -------
#         res: torch.Tensor (N_inducing, N_inducing)
#             K(X_inducing, X_inducing)
#         """
#         if not self.training and hasattr(self, "_cached_kernel_mat"):
#             return self._cached_kernel_mat
#         else:
#             res = delazify(self.base_kernel(self.X_inducing, self.X_inducing))
#             if not self.training:
#                 self._cached_kernel_mat = res  # type: torch.Tensor
#             return res

#     @property
#     def _inducing_inv_root(self) -> torch.Tensor:
#         """
#         Computes the inverse of the inducing matrix: K_inv(X_inducing, X_inducing) = K(X_inducing, X_inducing)^(-1)

#         Returns
#         -------
#         res: torch.Tensor (N_inducing, N_inducing)
#             K_inv(X_inducing, X_inducing)
#         """
#         if not self.training and hasattr(self, "_cached_kernel_inv_root"):
#             return self._cached_kernel_inv_root
#         else:
#             chol = psd_safe_cholesky(self._inducing_mat, upper=True, jitter=settings.cholesky_jitter.value())
#             eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
#             inv_root = torch.triangular_solve(eye, chol)[0]

#             res = inv_root
#             if not self.training:
#                 self._cached_kernel_inv_root = res  # type: torch.Tensor
#             return res

#     @property
#     def _k_u1(self) -> torch.Tensor:
#         """
#         Computes the covariance matrix between the X_inducing and X_out : K(X_inducing, X_out)

#         Returns
#         -------
#         res: torch.Tensor (N_inducing, N_out)
#             K(X_inducing, X_out)
#         """
#         if not self.training and hasattr(self, "_cached_k_u1"):
#             return self._cached_k_u1
#         else:
#             res = delazify(self.base_kernel(self.X_inducing, self.X_out))
#             if not self.training:
#                 self._cached_k_u1 = res  # type: torch.Tensor
#             else:
#                 self._train_cached_k_u1 = res  # type: torch.Tensor
#             return res

#     @property
#     def _lambda_diag_inv(self) -> torch.Tensor:
#         r"""Computes the inverse of lambda matrix, it is computed by
#         \Lambda = diag[\mathbf{K_{X_out,X_out}-Q_{X_out,X_out}} + \sigma^2_{noise}\idenmat] and
#         Q{X_out, X_out} = K(X_out, X_inducing) K^{-1}(X_inducing,X_inducing) K(X_inducing, X_out)

#         Returns
#         -------
#         res: torch.Tensor (N_out, N_out)
#             inverse of the diagonal matrix lambda
#         """
#         if not self.training and hasattr(self, "_cached_lambda_diag_inv"):
#             return self._cached_lambda_diag_inv
#         else:
#             diag_k11 = delazify(self.base_kernel(self.X_out, diag=True))

#             diag_q11 = delazify(RootLazyTensor(self._k_u1.transpose(-1, -2).matmul(self._inducing_inv_root))).diag()

#             # Diagonal correction for predictive posterior
#             correction = (diag_k11 - diag_q11).clamp(0, math.inf)

#             sigma = self.likelihood._shaped_noise_covar(correction.shape).diag()

#             res = delazify(DiagLazyTensor((correction + sigma).reciprocal()))

#             if not self.training:
#                 self._cached_lambda_diag_inv = res  # type: torch.Tensor
#             else:
#                 self._train_cached_lambda_diag_inv = res  # type: torch.Tensor
#             return res

#     @property
#     def _inducing_sigma(self) -> torch.Tensor:
#         r"""Computes the inverse of lambda matrix, it is computed by
#         \mathbf{\Sigma} = (\mathbf{K_{X_inducing,X_inducing}} +
#          \mathbf{K_{X_inducing, X_out} \Lambda}^{-1}\mathbf{K_{X_out,X_inducing}})

#         Returns
#         -------
#         res: torch.Tensor (N_inducing, N_inducing)
#             \Sigma
#         """
#         if not self.training and hasattr(self, "_cached_inducing_sigma"):
#             return self._cached_inducing_sigma
#         else:
#             k_u1 = self._k_u1
#             res = PsdSumLazyTensor(
#                 self._inducing_mat,
#                 MatmulLazyTensor(k_u1, MatmulLazyTensor(self._lambda_diag_inv, k_u1.transpose(-1, -2))),
#             )
#             res = delazify(res)
#             if not self.training:
#                 self._cached_inducing_sigma = res  # type: torch.Tensor

#             return res

#     @property
#     def _inducing_sigma_inv_root(self) -> torch.Tensor:
#         r"""Inverse of Sigma matrix:

#         Returns
#         -------
#         res: torch.Tensor (N_inducing, N_inducing)
#             \Sigma ^{-1}
#         """
#         if not self.training and hasattr(self, "_cached_inducing_sigma_inv_root"):
#             return self._cached_inducing_sigma_inv_root
#         else:
#             chol = psd_safe_cholesky(self._inducing_sigma, upper=True, jitter=settings.cholesky_jitter.value())

#             eye = torch.eye(chol.size(-1), device=chol.device, dtype=chol.dtype)
#             inv_root = torch.triangular_solve(eye, chol)[0]
#             res = inv_root
#             if not self.training:
#                 self._cached_inducing_sigma_inv_root = res  # type: torch.Tensor
#             return res

#     @property
#     def _poster_mean_mat(self) -> torch.Tensor:
#         r"""A cached value for computing the posterior mean of a sparse kernel it is defined by
#         \Sigma K_{u, 1} \Lambda}^{-1}\mathbf{y_out}

#         Returns
#         -------
#         res: torch.Tensor (N_inducing, 1)
#             cached posterior mean
#         """
#         if not self.training and hasattr(self, "_cached_poster_mean_mat"):
#             return self._cached_poster_mean_mat
#         else:
#             inducing_sigma_inv_root = self._inducing_sigma_inv_root
#             sigma = RootLazyTensor(inducing_sigma_inv_root)

#             k_u1 = self._k_u1
#             lambda_diag_inv = self._lambda_diag_inv

#             res_mat = delazify(MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))

#             res = torch.matmul(res_mat, self.y_out)

#             if not self.training:
#                 self._cached_poster_mean_mat = res  # type: torch.Tensor
#             return res

#     def _get_covariance(self, x1: torch.Tensor, x2: torch.Tensor) -> gpytorch.lazy.LazyTensor:
#         r"""Compute the posterior covariance matrix of a sparse kernel explicitly

#         Parameters
#         ----------
#         x1: torch.Tensor(N_x1, D)
#             first input of the FITC kernel
#         x2: torch.Tensor(N_x2, D)
#             second input of the FITC kernel

#         Returns
#         -------
#         res: Optional[torch.Tensor (N_x1, 1), PsdSumLazyTensor]
#             a cached value for computing the posterior mean, it
#             is defined by  \Sigma K_{u, 1} \Lambda}^{-1}\mathbf{y_out}
#         """
#         k_x1x2 = self.base_kernel(x1, x2)
#         k_x1u = delazify(self.base_kernel(x1, self.X_inducing))
#         inducing_inv_root = self._inducing_inv_root
#         inducing_sigma_inv_root = self._inducing_sigma_inv_root
#         if torch.equal(x1, x2):
#             q_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_inv_root))

#             s_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_sigma_inv_root))
#         else:
#             k_x2u = delazify(self.base_kernel(x2, self.X_inducing))
#             q_x1x2 = MatmulLazyTensor(
#                 k_x1u.matmul(inducing_inv_root), k_x2u.matmul(inducing_inv_root).transpose(-1, -2)
#             )
#             s_x1x2 = MatmulLazyTensor(
#                 k_x1u.matmul(inducing_sigma_inv_root), k_x2u.matmul(inducing_sigma_inv_root).transpose(-1, -2)
#             )
#         covar = PsdSumLazyTensor(k_x1x2, -1.0 * q_x1x2, s_x1x2)

#         if self.training:
#             k_iu = self.base_kernel(x1, self.X_inducing)
#             sigma = RootLazyTensor(inducing_sigma_inv_root)

#             k_u1 = self._train_cached_k_u1 if hasattr(self, "_train_cached_k_u1") else self._k_u1
#             lambda_diag_inv = (
#                 self._train_cached_lambda_diag_inv
#                 if hasattr(self, "_train_cached_lambda_diag_inv")
#                 else self._lambda_diag_inv
#             )

#             mean = torch.matmul(
#                 delazify(MatmulLazyTensor(k_iu, MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))),
#                 self.y_out,
#             )

#             self._train_cached_posterior_mean = mean  # type: torch.Tensor
#         return covar

#     def posterior_mean(self, inputs: torch.Tensor) -> torch.Tensor:
#         """
#         The posterior mean of the FITC kernel, will serve as the prior mean of the dense kernel.

#         Parameters
#         ----------
#         inputs: torch.Tensor(N_inputs, D)
#             input of the FITC kernel

#         Returns
#         -------
#         res: Torch.Tensor (N_inputs, 1)
#             The posterior mean of the FITC Kernel
#         """
#         if self.training and hasattr(self, "_train_cached_posterior_mean"):
#             return self._train_cached_posterior_mean
#         if inputs.ndimension() == 1:
#             inputs = inputs.unsqueeze(1)

#         k_iu = delazify(self.base_kernel(inputs, self.X_inducing, cont_only=True))
#         poster_mean = self._poster_mean_mat
#         res = torch.matmul(k_iu, poster_mean)
#         return res

#     def forward(
#         self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **kwargs: Any
#     ) -> gpytorch.lazy.LazyTensor:
#         """Compute the kernel function"""
#         covar = self._get_covariance(x1, x2)
#         if self.training:
#             if not torch.equal(x1, x2):
#                 raise RuntimeError("x1 should equal x2 in training mode")

#         if diag:
#             return covar.diag()
#         else:
#             return covar

#     def num_outputs_per_input(self, x1: torch.Tensor, x2: torch.Tensor) -> int:
#         """
#         Number of outputs given the inputs
#         if x1 is of size `n x d` and x2 is size `m x d`, then the size of the kernel
#         will be `(n * num_outputs_per_input) x (m * num_outputs_per_input)`

#         Parameters
#         ----------
#         x1: torch.Tensor
#             the first input of the kernel
#         x2: torch.Tensor
#             the second input of the kernel
#         Returns
#         -------
#         res: int
#             for base kernels such as matern or RBF kernels, this value needs to be 1.
#         """
#         return self.base_kernel.num_outputs_per_input(x1, x2)

#     def __deepcopy__(self, memo: Dict) -> "FITCKernel":
#         replace_inv_root = False
#         replace_kernel_mat = False
#         replace_k_u1 = False
#         replace_lambda_diag_inv = False
#         replace_inducing_sigma = False
#         replace_inducing_sigma_inv_root = False
#         replace_poster_mean = False

#         if hasattr(self, "_cached_kernel_inv_root"):
#             replace_inv_root = True
#             kernel_inv_root = self._cached_kernel_inv_root
#         if hasattr(self, "_cached_kernel_mat"):
#             replace_kernel_mat = True
#             kernel_mat = self._cached_kernel_mat
#         if hasattr(self, "_cached_k_u1"):
#             replace_k_u1 = True
#             k_u1 = self._cached_k_u1
#         if hasattr(self, "_cached_lambda_diag_inv"):
#             replace_lambda_diag_inv = True
#             lambda_diag_inv = self._cached_lambda_diag_inv
#         if hasattr(self, "_cached_inducing_sigma"):
#             replace_inducing_sigma = True
#             inducing_sigma = self._cached_inducing_sigma
#         if hasattr(self, "_cached_inducing_sigma_inv_root"):
#             replace_inducing_sigma_inv_root = True
#             inducing_sigma_inv_root = self._cached_inducing_sigma_inv_root
#         if hasattr(self, "_cached_poster_mean_mat"):
#             replace_poster_mean = True
#             poster_mean_mat = self._cached_poster_mean_mat

#         cp = self.__class__(
#             base_kernel=copy.deepcopy(self.base_kernel),
#             X_inducing=copy.deepcopy(self.X_inducing),
#             X_out=self.X_out,
#             y_out=self.y_out,
#             likelihood=copy.deepcopy(self.likelihood),
#             active_dims=self.active_dims,
#         )

#         if replace_inv_root:
#             cp._cached_kernel_inv_root = kernel_inv_root

#         if replace_kernel_mat:
#             cp._cached_kernel_mat = kernel_mat

#         if replace_k_u1:
#             cp._cached_k_u1 = k_u1

#         if replace_lambda_diag_inv:
#             cp._cached_lambda_diag_inv = lambda_diag_inv

#         if replace_inducing_sigma:
#             cp._cached_inducing_sigma = inducing_sigma

#         if replace_inducing_sigma_inv_root:
#             cp._cached_inducing_sigma_inv_root = inducing_sigma_inv_root

#         if replace_poster_mean:
#             cp._cached_poster_mean_mat = poster_mean_mat

#         return cp


# class FITCMean(Mean):
#     def __init__(self, covar_module: FITCKernel, batch_shape: torch.Size = torch.Size(), **kwargs: Any):
#         """
#         Read the posterior mean value of the given fitc kernel and serve as a prior mean value for the
#         second stage

#         Parameters
#         ----------
#         covar_module: FITCKernel
#             a FITC  kernel
#         batch_shape: torch.size
#             batch size
#         """
#         super(FITCMean, self).__init__()
#         self.covar_module = covar_module
#         self.batch_shape = batch_shape
#         self.covar_module = covar_module

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the posterior mean from the cached value of FITC kernels

#         Parameters
#         ----------
#         input: torch.Tensor(N_xin, D)
#             input torch Tensor

#         Returns
#         -------
#         res: torch.Tensor(N_xin)
#             posterior mean value of FITC GP model
#         """
#         # detach is applied here to avoid updating the same parameter twice in the same iteration
#         # which might result in an error
#         res = self.covar_module.posterior_mean(input).detach()
#         return res
