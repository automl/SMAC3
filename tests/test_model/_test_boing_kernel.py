import copy
import unittest.mock

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.inducing_point_kernel import InducingPointKernel
from gpytorch.lazy import LazyEvaluatedKernelTensor, delazify
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means.zero_mean import ZeroMean
from gpytorch.models.exact_gp import ExactGP

from smac.model.gaussian_process.kernels._boing import FITCKernel, FITCMean
from smac.model.utils import check_subspace_points


class FITC(ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_kernel, inducing_points):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = InducingPointKernel(base_kernel, inducing_points, likelihood)
        self.prediction_strategy = self.covar_module.prediction_strategy

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def generate_test_data(
    rs: np.random.RandomState, num_inducing=4, num_points_in=8, num_points_out=10, num_dims=5, expand_bound=True
):
    X_out = rs.rand(num_points_out, num_dims)
    Y_out = rs.rand(num_points_out, 1)
    # X \in [-0.6, 1.4] and the bound is [0, 1]
    X_out = (X_out - 0.3) * 2

    X_in = rs.rand(num_points_in, num_dims)
    Y_in = rs.rand(num_points_in, 1)

    X_inducing = rs.rand(num_inducing, num_dims)

    X = np.vstack([X_in, X_out])
    Y = np.vstack([Y_in, Y_out])

    Y = np.squeeze(Y)

    data_indices = np.arange((num_points_in + num_points_out))

    rs.shuffle(data_indices)

    X = X[data_indices]
    Y = Y[data_indices]

    ss_in = check_subspace_points(
        X, cont_dims=np.arange(num_dims), bounds_cont=np.tile([0.0, 1.0], [num_dims, 1]), expand_bound=expand_bound
    )

    X_in = X[ss_in]
    Y_in = Y[ss_in]
    X_out = X[~ss_in]
    Y_out = Y[~ss_in]

    return X_in, Y_in, X_out, Y_out, X_inducing


def generate_kernel(n_dimensions):
    exp_kernel = MaternKernel(
        2.5,
        lengthscale_constraint=Interval(
            torch.tensor(np.exp(-6.754111155189306).repeat(n_dimensions)),
            torch.tensor(np.exp(0.0858637988771976).repeat(n_dimensions)),
            transform=None,
            initial_value=1.0,
        ),
        ard_num_dims=n_dimensions,
        active_dims=torch.arange(n_dimensions),
    ).double()

    kernel = ScaleKernel(
        exp_kernel, outputscale_constraint=Interval(np.exp(-10.0), np.exp(2.0), transform=None, initial_value=2.0)
    ).double()
    return kernel


class TestFITCKernel(unittest.TestCase):
    def setUp(self) -> None:
        rs = np.random.RandomState(1)
        num_dims = 5
        self.likelihood = GaussianLikelihood().double()
        X_in, Y_in, X_out, Y_out, X_inducing = generate_test_data(rs, num_dims=num_dims)
        self.kernel = generate_kernel(num_dims)
        self.X_in = torch.from_numpy(X_in)
        self.Y_in = torch.from_numpy(Y_in)
        self.X_out = torch.from_numpy(X_out)
        self.Y_out = torch.from_numpy(Y_out)
        self.X_inducing = torch.from_numpy(X_inducing)
        self.ga_kernel = FITCKernel(
            base_kernel=self.kernel,
            X_inducing=self.X_inducing,
            likelihood=self.likelihood,
            X_out=self.X_out,
            y_out=self.Y_out,
        ).double()
        self.ga_mean = FITCMean(covar_module=self.ga_kernel)
        self.fitc = FITC(
            train_x=self.X_out,
            train_y=self.Y_out,
            likelihood=self.likelihood,
            base_kernel=self.kernel,
            inducing_points=self.X_inducing,
        )

        self.fitc_eval_cache = {
            "_cached_kernel_mat": "_inducing_mat",
            "_cached_inducing_sigma": "_inducing_sigma",
            "_cached_poster_mean_mat": "_poster_mean_mat",
            "_cached_kernel_inv_root": "_inducing_inv_root",
        }
        self.fitc_train_cache = {
            "_train_cached_k_u1": "_k_u1",
            "_train_cached_lambda_diag_inv": "_lambda_diag_inv",
            "_train_cached_posterior_mean": "posterior_mean",
        }

    def test_init(self):
        ga_kernel = FITCKernel(
            base_kernel=self.kernel, X_inducing=torch.from_numpy(np.empty(2)), likelihood=None, X_out=None, y_out=None
        )
        self.assertTrue(hasattr(ga_kernel, "X_inducing"))
        self.assertEqual(len(ga_kernel.X_inducing.shape), 2)
        self.assertTrue("X_inducing" in dict(ga_kernel.named_parameters()))
        self.assertTrue(self.ga_mean.covar_module is self.ga_kernel)

    def test_forward(self):
        ga_covar = self.ga_kernel(self.X_in)
        ga_mean = delazify(self.ga_mean(self.X_in))
        ga_covar_diag = self.ga_kernel(self.X_in).diag()

        self.assertIsInstance(ga_covar, LazyEvaluatedKernelTensor)
        ga_covar = delazify(ga_covar)
        self.fitc.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            fitc_output = self.fitc(self.X_in)
            fitc_mean = fitc_output.mean
            fitc_covar = fitc_output.covariance_matrix

        torch.testing.assert_allclose(fitc_mean, ga_mean)
        torch.testing.assert_allclose(ga_covar, fitc_covar)
        torch.testing.assert_allclose(ga_covar_diag, fitc_output.variance)

        with self.assertRaises(RuntimeError):
            delazify(self.ga_kernel(self.X_in, self.X_out))

        self.ga_kernel.eval()
        ga_covar_train_test = delazify(self.ga_kernel(self.X_in, self.X_out))

        self.assertTrue(ga_covar_train_test.shape == (len(self.X_in), len(self.X_out)))

    def test_cache(self):
        self.ga_kernel.train()
        delazify(self.ga_kernel(self.X_in))
        for cache in self.fitc_eval_cache.keys():
            self.assertFalse(hasattr(self.ga_kernel, cache))
        # Make sure that all the cached values are successfully stored
        for cache, value in self.fitc_train_cache.items():
            self.assertTrue(hasattr(self.ga_kernel, cache))
            if cache == "_train_cached_posterior_mean":
                torch.testing.assert_allclose(getattr(self.ga_kernel, cache), getattr(self.ga_kernel, value)(self.X_in))
            else:
                torch.testing.assert_allclose(getattr(self.ga_kernel, cache), getattr(self.ga_kernel, value))

        self.ga_kernel.eval()
        delazify(self.ga_kernel(self.X_in))
        delazify(self.ga_mean(self.X_in))
        for cache, value in self.fitc_eval_cache.items():
            self.assertTrue(hasattr(self.ga_kernel, cache))
            self.assertTrue(getattr(self.ga_kernel, cache) is getattr(self.ga_kernel, value))
        for cache in self.fitc_train_cache:
            self.assertFalse(hasattr(self.ga_kernel, cache))

        self.ga_kernel.train()
        for cache in self.fitc_eval_cache:
            self.assertFalse(hasattr(self.ga_kernel, cache))
        for cache in self.fitc_train_cache:
            self.assertFalse(hasattr(self.ga_kernel, cache))

    def test_copy(self):
        # clear all the cache
        self.ga_kernel.train()
        gp_kernel_copy_1 = copy.deepcopy(self.ga_kernel)
        # ga_kernel does not have the cached values thus they should all be empty
        for cache in self.fitc_train_cache.keys():
            self.assertFalse(hasattr(gp_kernel_copy_1, cache))
        for cache in self.fitc_eval_cache.keys():
            self.assertFalse(hasattr(gp_kernel_copy_1, cache))

        delazify(self.ga_kernel(self.X_in))
        gp_kernel_copy_2 = copy.deepcopy(self.ga_kernel)
        for cache in self.fitc_train_cache.keys():
            self.assertFalse(hasattr(gp_kernel_copy_2, cache))
        for cache in self.fitc_eval_cache.keys():
            self.assertFalse(hasattr(gp_kernel_copy_2, cache))

        self.ga_kernel.eval()
        delazify(self.ga_kernel(self.X_in))
        delazify(self.ga_mean(self.X_in))

        gp_kernel_copy_3 = copy.deepcopy(self.ga_kernel)
        for cache, value in self.fitc_eval_cache.items():
            self.assertTrue(hasattr(gp_kernel_copy_3, cache))
        for cache in self.fitc_train_cache:
            self.assertFalse(hasattr(gp_kernel_copy_3, cache))
