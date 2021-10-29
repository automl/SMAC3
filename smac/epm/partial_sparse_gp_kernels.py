import math
from typing import Optional, Tuple
import copy

import torch
from gpytorch import settings
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.means.mean import Mean
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, PsdSumLazyTensor, RootLazyTensor, delazify
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood


class PartialSparseKernel(Kernel):
    def __init__(self,
                 base_kernel: Kernel,
                 X_inducing: torch.tensor,
                 likelihood: GaussianLikelihood,
                 X_out: torch.tensor,
                 y_out: torch.tensor,
                 active_dims: Optional[Tuple[int]] = None):
        """
        A kernel for partial sparse gaussian process. When doing forward, it needs to pass two GP kernel where the
        two kernels share the same hyperparameters (kernel length, kernel scale and noises), the first one is a sparse
        GP kernel and has inducing points as its hyperparameter. When computing the posterior of the partial sparse
        kernel, we first compute the posterior w.r.t. outer_X and outer_y. Then we consider this posterior as a prior
        of the second stage where we compute the posterior distribution of X_in (the input of forward function)
        Parameters
        Mean value is computed with:
        \mathbf{\mu_{l'}}  = \mathbf{K_{l',u} \Sigma K_{u,1} \Lambda}^{-1}\mathbf{y_g} \label{eq:mean_sgp}
        and variance value:
        \mathbf{\sigma}^2_{l'} = \mathbf{K_{l',l'}} - \mathbf{Q_{l', l'} + \mathbf{K_{l', u}\Sigma K_{u, l'}}}
        \mathbf{\Sigma} = (\mathbf{K_{u,u}} + \mathbf{K_{u, g} \Lambda}^{-1}\mathbf{K_{g,u}})^{-1}
        \mathbf{\Lambda} = diag[\mathbf{K_{g,g}-Q_{g,g}} + \sigma^2_{noise}\idenmat]
        ----------
        base_kernel: Kernel
            base kernel function
        X_inducing: torch.tensor (N_inducing, D)
            inducing points, should be of size (N_inducing, D), N_inducing is the number of the inducing points
        likelihood: GaussianLikelihood
            GP likelihood
        X_out: torch.tensor (N_out,D)
            data features outside the subregion, needs to be of size (N_out, D), N_out is the number of points outside
            the subspace
        y_out: torch.tensor
            data observations outside the subregion
        active_dims: typing.Optional[typing.Tuple[int]] = None
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        """
        super(PartialSparseKernel, self).__init__(active_dims=active_dims)
        self.has_lengthscale = base_kernel.has_lengthscale
        self.base_kernel = base_kernel
        self.likelihood = likelihood

        if X_inducing.ndimension() == 1:
            X_inducing = X_inducing.unsqueeze(-1)

        self.X_out = X_out
        self.y_out = y_out
        self.register_parameter(name="X_inducing", parameter=torch.nn.Parameter(X_inducing))

    def train(self, mode: bool = True) -> None:
        """
        turn the model into training mode, needs to clear all the cached value as they are not required when doing
        training
        Parameters
        ----------
        mode: bool
        if the model is under training mode
        """
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
    def _inducing_mat(self) -> torch.tensor:
        """
        computes inducing matrix, K(X_inducing, X_inducing)
        Returns
        -------
        res: torch.tensor (N_inducing, N_inducing)
        K(X_inducing, X_inducing)
        """
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat
        else:
            res = delazify(self.base_kernel(self.X_inducing, self.X_inducing))
            if not self.training:
                self._cached_kernel_mat = res
            return res

    @property
    def _inducing_inv_root(self) -> torch.tensor:
        """
        computes the inverse of the inducing matrix: K_inv(X_inducing, X_inducing) = K(X_inducing, X_inducing)^(-1)
        Returns
        -------
        res: torch.tensor (N_inducing, N_inducing)
        K_inv(X_inducing, X_inducing)
        """
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
    def _k_u1(self) -> torch.tensor:
        """
        computes the covariance matrix between the X_inducing and X_out : K(X_inducing, X_out)
        Returns
        -------
        res: torch.tensor (N_inducing, N_out)
         K(X_inducing, X_out)
        """
        if not self.training and hasattr(self, "_cached_k_u1"):
            return self._cached_k_u1
        else:
            res = delazify(self.base_kernel(self.X_inducing, self.X_out))
            if not self.training:
                self._cached_k_u1 = res
            else:
                self._train_cached_k_u1 = res.clone()
            return res

    @property
    def _lambda_diag_inv(self):
        """
        computes the inverse of lambda matrix, is computed by
        \Lambda = diag[\mathbf{K_{X_out,X_out}-Q_{X_out,X_out}} + \sigma^2_{noise}\idenmat] and
        Q{X_out, X_out} = K(X_out, X_inducing) K^{-1}(X_inducing,X_inducing) K(X_inducing, X_out)
        Returns
        -------
        res: torch.tensor (N_out, N_out)
        inverse of the diagonal matrix lambda
        """
        if not self.training and hasattr(self, "_cached_lambda_diag_inv"):
            return self._cached_lambda_diag_inv
        else:
            diag_k11 = delazify(self.base_kernel(self.X_out, diag=True))

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
        """
        computes the inverse of lambda matrix, is computed by
        \mathbf{\Sigma} = (\mathbf{K_{X_inducing,X_inducing}} +
         \mathbf{K_{X_inducing, X_out} \Lambda}^{-1}\mathbf{K_{X_out,X_inducing}})
        Returns
        -------
        res: torch.tensor (N_inducing, N_inducing)
        \Sigma
        """
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
        """
        inverse of Sigma matrix:
        Returns
        -------
        res: torch.tensor (N_inducing, N_inducing)
        \Sigma ^{-1}
        """
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
        """
        A cached value for computing posterior mean of a sparse kernel:
        Returns
        -------
        res: torch.tensor (N_inducing, 1)
        a cached value for computing the posterior mean,
        is defined by  \Sigma K_{u, 1} \Lambda}^{-1}\mathbf{y_out}
        """
        if not self.training and hasattr(self, "_cached_poster_mean_mat"):
            return self._cached_poster_mean_mat
        else:
            inducing_sigma_inv_root = self._inducing_sigma_inv_root
            sigma = RootLazyTensor(inducing_sigma_inv_root)

            k_u1 = self._k_u1
            lambda_diag_inv = self._lambda_diag_inv

            res_mat = delazify(MatmulLazyTensor(sigma, MatmulLazyTensor(k_u1, lambda_diag_inv)))

            res = torch.matmul(res_mat, self.y_out)

            if not self.training:
                self._cached_poster_mean_mat = res
            return res

    def _get_covariance(self, x1: torch.tensor, x2: torch.tensor):
        """
        Compute the posterior covariance matrix of the sparse kernel (will serve as the prior for the GP
        kernel in the second stage)
        Parameters
        ----------
        x1: torch.tensor(N_x1, D)
        first input of the partial sparse kernel
        x2: torch.tensor(N_x2, D)
        second input of the partial sparse kernel
        Returns
        -------
        res: torch.tensor (N_x1, 1) or PsdSumLazyTensor
        a cached value for computing the posterior mean,
        is defined by  \Sigma K_{u, 1} \Lambda}^{-1}\mathbf{y_out}
        """
        k_x1x2 = self.base_kernel(x1, x2)
        k_x1u = delazify(self.base_kernel(x1, self.X_inducing))
        inducing_inv_root = self._inducing_inv_root
        inducing_sigma_inv_root = self._inducing_sigma_inv_root
        if torch.equal(x1, x2):
            q_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_inv_root))

            s_x1x2 = RootLazyTensor(k_x1u.matmul(inducing_sigma_inv_root))
        else:
            k_x2u = delazify(self.base_kernel(x2, self.X_inducing))
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
        """
        covar matrix diagonal
        Parameters
        ----------
        inputs: torch.tensor(N_inputs, D)
        input of the partial sparse kernel
        Returns
        -------
        res: DiagLazyTensor (N_inputs, 1)
        a diagional matrix
        """
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        # Get diagonal of covar
        covar_diag = delazify(self.base_kernel(inputs, diag=True))
        return DiagLazyTensor(covar_diag)

    def posterior_mean(self, inputs):
        """
        posterior mean of the sparse kernel, will serve as the prior mean of the dense kernel
        Parameters
        ----------
        inputs: torch.tensor(N_inputs, D)
        input of the partial sparse kernel
        Returns
        -------
        res: Torch.tensor (N_inputs, 1)
        posterior mean of sparse Kernel
        """
        if self.training and hasattr(self, "_cached_posterior_mean"):
            return self._cached_posterior_mean
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)

        k_iu = delazify(self.base_kernel(inputs, self.X_inducing))
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

    def num_outputs_per_input(self, x1, x2) -> int:
        """
        Number of outputs given the inputs
        Parameters
        ----------
        x1: torch.tensor(N_x1, D)
        input of the partial sparse kernel
        Returns
        -------
        res: int
        for base kernels such as matern or RBF kernels, this value needs to be 1.
        """
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
            X_inducing=copy.deepcopy(self.X_inducing),
            X_out=self.X_out,
            y_out=self.y_out,
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
    def __init__(self, covar_module: PartialSparseKernel, batch_shape=torch.Size(), **kwargs):
        """
        Read the posterior mean value of the given partial sparse kernel and serve as a prior mean value for the
        second stage
        Parameters
        ----------
        covar_module: PartialSparseKernel
        a partial sparse kernel
        batch_shape: torch.size
        batch size
        """
        super(PartialSparseMean, self).__init__()
        self.covar_module = covar_module
        self.batch_shape = batch_shape
        self.covar_module = covar_module

    def forward(self, input: torch.tensor):
        """
        Compute the posterior mean from the cached value of partial sparse kernel
        Parameters
        ----------
        input: torch.tensor(N_xin, D)
        input torch tensor
        Returns
        -------
        res: torch.tensor(N_xin)
        posterior mean value of sparse GP model
        """
        # detach is applied here to avoid updating the same parameter twice in the same iteration
        # which might result in an error
        res = self.covar_module.posterior_mean(input).detach()
        return res
