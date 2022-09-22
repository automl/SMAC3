from typing import Tuple

import unittest.mock

import gpytorch
import numpy as np
import pyro
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors import HorseshoePrior

from smac.configspace import ConfigurationSpace, UniformFloatHyperparameter
from smac.model.gaussian_process.augmented_local_gaussian_process import (
    AugmentedLocalGaussianProcess,
    GloballyAugmentedLocalGaussianProcess,
)
from smac.model.gaussian_process.gpytorch_gaussian_process import (
    ExactGaussianProcessModel,
)

from ._test_boing_kernel import generate_kernel, generate_test_data
from ._test_gp_gpytorch import TestGPGPyTorch

torch.manual_seed(0)
pyro.set_rng_seed(0)


def generate_lgpga(
    kernel, n_dimensions, rs, noise=None, num_inducing=2, normalize_y=True
) -> Tuple[GloballyAugmentedLocalGaussianProcess, ConfigurationSpace]:
    if noise is None:
        likelihood = None
    else:
        noise_prior = HorseshoePrior(0.1)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior, noise_constraint=Interval(np.exp(-25), np.exp(2), transform=None)
        ).double()
        likelihood.initialize(noise=noise)

    bounds = [(0.0, 1.0) for _ in range(n_dimensions)]
    types = np.zeros(n_dimensions)

    configspace = ConfigurationSpace()
    for i in range(n_dimensions):
        configspace.add_hyperparameter(UniformFloatHyperparameter("x%d" % i, 0, 1))

    model = GloballyAugmentedLocalGaussianProcess(
        configspace=configspace,
        bounds=bounds,
        types=types,
        bounds_cont=np.array(bounds),
        bounds_cat=[],
        kernel=kernel,
        seed=rs.randint(low=1, high=10000),
        normalize_y=normalize_y,
        likelihood=likelihood,
        n_opt_restarts=2,
        num_inducing_points=num_inducing,
    )
    return model, configspace


class TestLGPGA(TestGPGPyTorch):
    def setUp(self) -> None:
        rs = np.random.RandomState(1)
        num_dims = 5
        X_in, Y_in, X_out, Y_out, _ = generate_test_data(rs, num_dims=num_dims, expand_bound=True)
        self.kernel = generate_kernel(num_dims)
        self.num_dims = num_dims
        self.X_in = X_in
        self.Y_in = Y_in
        self.X_out = X_out
        self.Y_out = Y_out

        self.X_all = np.vstack([self.X_in, self.X_out])
        self.Y_all = np.hstack([self.Y_in, self.Y_out])
        self.gp_model, self.cs = generate_lgpga(self.kernel, n_dimensions=num_dims, rs=rs)

    def test_init(self):
        np.testing.assert_equal(self.gp_model.cont_dims, np.arange(len(self.cs.get_hyperparameters())))
        np.testing.assert_equal(self.gp_model.cat_dims, np.array([]))

    def test_update_attribute(self):
        rs = np.random.RandomState(1)
        num_dims = 5
        num_inducing = 2
        gp_model, _ = generate_lgpga(self.kernel, n_dimensions=num_dims, num_inducing=2, rs=rs)
        self.assertTrue(gp_model.num_inducing_points, num_inducing)
        num_inducing = 4

        gp_model.update_attribute(num_inducing_points=num_inducing)
        self.assertTrue(gp_model.num_inducing_points, num_inducing)

        with self.assertRaises(AttributeError):
            gp_model.update_attribute(unknown_param=1)

    def test_get_gp(self):
        self.assertIsNone(self.gp_model.gp)
        self.gp_model._get_gp(self.X_in, self.Y_in)
        self.assertIsInstance(self.gp_model.gp_model, ExactGaussianProcessModel)

        self.gp_model._get_gp(self.X_in, self.Y_in, self.X_out, self.Y_out)
        self.assertIsInstance(self.gp_model.gp_model, AugmentedLocalGaussianProcess)

        # num_outer is not enough, we return to a vanilla GP model
        self.gp_model._train(self.X_in, self.Y_in, do_optimize=False)
        self.assertIsInstance(self.gp_model.gp_model, ExactGaussianProcessModel)

        self.gp_model._train(self.X_all, self.Y_all, do_optimize=False)
        self.assertIsInstance(self.gp_model.gp_model, AugmentedLocalGaussianProcess)
        self.assertFalse(self.gp_model.gp_model.augmented)
        self.assertFalse(hasattr(self.gp_model.gp_model, "covar_module"))

    def test_normalize(self):
        self.gp_model._train(self.X_all, self.Y_all, do_optimize=False)
        y_in_mean = np.mean(self.Y_in)
        y_in_std = np.std(self.Y_in)
        np.testing.assert_allclose(self.gp_model.gp_model.y_out.numpy(), (self.Y_out - y_in_mean) / y_in_std)

        rs = np.random.RandomState(1)
        model_unnormalize, configspace = generate_lgpga(self.kernel, self.num_dims, rs, normalize_y=False)
        model_unnormalize._train(self.X_all, self.Y_all, do_optimize=False)
        np.testing.assert_allclose(model_unnormalize.gp_model.y_out.numpy(), self.Y_out)

    def test_augmented_gp(self):
        X_in = torch.from_numpy(self.X_in)
        Y_in = torch.from_numpy(self.Y_in)
        X_out = torch.from_numpy(self.X_out)
        Y_out = torch.from_numpy(self.Y_out)

        augmented_gp = AugmentedLocalGaussianProcess(
            X_in, Y_in, X_out, Y_out, self.gp_model.likelihood, self.kernel
        ).double()
        exact_gp = ExactGaussianProcessModel(X_in, Y_in, self.kernel, self.gp_model.likelihood).double()

        # if augmented_gp.augmented is false, it should behave the same as an exact gp
        output_agp = augmented_gp(X_in)
        output_exact_gp = exact_gp(X_in)
        torch.testing.assert_allclose(output_agp.mean, output_exact_gp.mean)
        torch.testing.assert_allclose(output_agp.covariance_matrix, output_exact_gp.covariance_matrix)

        augmented_gp.eval()
        exact_gp.eval()
        output_agp = augmented_gp(X_out)
        output_exact_gp = exact_gp(X_out)

        torch.testing.assert_allclose(output_agp.mean, output_exact_gp.mean)
        torch.testing.assert_allclose(output_agp.covariance_matrix, output_exact_gp.covariance_matrix)

        # now augmentd_gp is augmented with inducing points, it no longer provides the same output as exact gp
        augmented_gp.set_augment_module(X_inducing=torch.ones([1, self.num_dims]))
        augmented_gp.eval()
        output_agp = augmented_gp(X_out)
        self.assertFalse(torch.equal(output_agp.mean, output_exact_gp.mean))
        self.assertFalse(torch.equal(output_agp.covariance_matrix, output_exact_gp.covariance_matrix))

    @unittest.mock.patch("gpytorch.models.exact_gp.ExactGP.__init__")
    def test_exception(self, fit_mock):
        # Check that training will not continue sampling if pyro raises an error
        class Dummy:
            counter = 0

            def __call__(self):
                self.counter += 1
                raise RuntimeError("Unable to sample new cfgs")

        fit_mock.side_effect = Dummy()

        with self.assertRaises(RuntimeError):
            self.gp_model._train(self.X_all, self.Y_all, do_optimize=True)

    def test_predict_with_actual_values(self):
        self.gp_model._train(self.X_all, self.Y_all, do_optimize=False)
        self.assertFalse(hasattr(self.gp_model.gp_model, "covar_module"))

        self.gp_model._train(self.X_all, self.Y_all, do_optimize=True)
        self.assertTrue(hasattr(self.gp_model.gp_model, "covar_module"))

        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 2.0],
                [-1.0, 2.0, -1.0],
                [-1.0, 2.0, 2.0],
                [2.0, -1.0, -1.0],
                [2.0, -1.0, 2.0],
                [2.0, 2.0, -1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float64,
        )
        y = np.array(
            [
                [0.1],
                [0.2],
                [9],
                [9.2],
                [100.0],
                [100.2],
                [109.0],
                [109.2],
                [1.0],
                [1.2],
                [14.0],
                [14.2],
                [110.0],
                [111.2],
                [129.0],
                [129.2],
            ],
            dtype=np.float64,
        )
        rs = np.random.RandomState(1)
        num_inducing = 4
        model, _ = generate_lgpga(kernel=generate_kernel(3), n_dimensions=3, rs=rs, num_inducing=num_inducing)
        model.train(np.vstack((X, X, X)), np.vstack((y, y, y)))

        self.assertEqual(model.is_trained, True)

        self.assertTrue(hasattr(self.gp_model.gp_model, "covar_module"))
        mu_hat, var_hat = model.predict(np.array([[0.5, 0.5, 0.5]]))

        self.assertAlmostEqual(mu_hat[0][0], 54.612500000000004)
        # There's a slight difference between my local installation and travis
        self.assertLess(abs(var_hat[0][0] - 1026.149240121437), 15)

    def test_varitional_inference(self):
        # test taht variational inference is actually called
        # https://github.com/cornellius-gp/gpytorch/blob/master/test/kernels/test_inducing_point_kernel.py#L45
        _wrapped_ps = unittest.mock.MagicMock(wraps=gpytorch.variational.TrilNaturalVariationalDistribution)
        with unittest.mock.patch("gpytorch.variational.TrilNaturalVariationalDistribution", new=_wrapped_ps) as rf_mock:
            self.gp_model._train(self.X_all, self.Y_all, do_optimize=True)
            self.assertTrue(rf_mock.called)
