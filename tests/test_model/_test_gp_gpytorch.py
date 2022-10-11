import unittest.mock

import numpy as np
import pyro
import torch
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.kernels import MaternKernel, ProductKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.models.exact_gp import ExactGP
from gpytorch.priors import HorseshoePrior, LogNormalPrior, UniformPrior
from gpytorch.utils.errors import NotPSDError

from smac.configspace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    UniformFloatHyperparameter,
)
from smac.model.gaussian_process.gpytorch_gaussian_process import (
    GPyTorchGaussianProcess,
)

from ._test_gp import TestGP, get_cat_data, get_cont_data

torch.manual_seed(0)
pyro.set_rng_seed(0)


def get_gp(n_dimensions, rs, noise=None, normalize_y=True) -> GPyTorchGaussianProcess:
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
        lengthscale_prior=UniformPrior(np.exp(-6.754111155189306), np.exp(0.858637988771976)),
    ).double()

    kernel = ScaleKernel(
        exp_kernel,
        outputscale_constraint=Interval(np.exp(-10.0), np.exp(2.0), transform=None, initial_value=2.0),
        outputscale_prior=LogNormalPrior(0.0, 1.0),
    ).double()
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

    model = GPyTorchGaussianProcess(
        configspace=configspace,
        bounds=bounds,
        types=types,
        kernel=kernel,
        seed=rs.randint(low=1, high=10000),
        normalize_y=normalize_y,
        likelihood=likelihood,
        n_opt_restarts=2,
    )
    return model


def get_mixed_gp(cat_dims, cont_dims, rs, normalize_y=True):
    cat_dims = np.array(cat_dims, dtype=np.int)
    cont_dims = np.array(cont_dims, dtype=np.int)
    n_dimensions = len(cat_dims) + len(cont_dims)
    exp_kernel = MaternKernel(
        2.5,
        lengthscale_constraint=Interval(
            torch.tensor(np.exp(-6.754111155189306).repeat(cont_dims.shape[-1])),
            torch.tensor(np.exp(0.0858637988771976).repeat(cont_dims.shape[-1])),
            transform=None,
            initial_value=1.0,
        ),
        ard_num_dims=cont_dims.shape[-1],
        active_dims=tuple(cont_dims),
        lengthscale_prior=UniformPrior(np.exp(-6.754111155189306), np.exp(0.858637988771976)),
    ).double()

    ham_kernel = CategoricalKernel(
        lengthscale_constraint=Interval(
            torch.tensor(np.exp(-6.754111155189306).repeat(cat_dims.shape[-1])),
            torch.tensor(np.exp(0.0858637988771976).repeat(cat_dims.shape[-1])),
            transform=None,
            initial_value=1.0,
        ),
        ard_num_dims=cat_dims.shape[-1],
        active_dims=tuple(cat_dims),
        lengthscale_prior=UniformPrior(np.exp(-6.754111155189306), np.exp(0.858637988771976)),
    ).double()

    kernel = ProductKernel(exp_kernel, ham_kernel)

    kernel = ScaleKernel(
        kernel,
        outputscale_constraint=Interval(np.exp(-10.0), np.exp(2.0), transform=None, initial_value=2.0),
        outputscale_prior=LogNormalPrior(0.0, 1.0),
    )

    bounds = [0] * n_dimensions
    types = np.zeros(n_dimensions)
    for c in cont_dims:
        bounds[c] = (0.0, 1.0)
    for c in cat_dims:
        types[c] = 3
        bounds[c] = (3, np.nan)

    cs = ConfigurationSpace()
    for c in cont_dims:
        cs.add_hyperparameter(UniformFloatHyperparameter("X%d" % c, 0, 1))
    for c in cat_dims:
        cs.add_hyperparameter(CategoricalHyperparameter("X%d" % c, [0, 1, 2, 3]))

    model = GPyTorchGaussianProcess(
        configspace=cs,
        bounds=bounds,
        types=types,
        kernel=kernel,
        seed=rs.randint(low=1, high=10000),
        normalize_y=normalize_y,
    )
    return model


class TestGPGPyTorch(TestGP):
    def test_gp_model(self):
        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)
        model = get_gp(n_dims, rs, normalize_y=True)
        self.assertTrue(model.normalize_y)
        self.assertIsNone(model.gp)
        self.assertEqual(np.shape(model.hypers), (0,))
        self.assertEqual(model.is_trained, False)
        self.assertEqual(bool(model.property_dict), False)

        mll = model._get_gaussian_process(X, Y)
        self.assertIsInstance(mll, ExactMarginalLogLikelihood)
        self.assertIsInstance(mll.model, ExactGP)

    def test_likelihood(self):
        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)
        model = get_gp(n_dims, rs)
        self.assertIsInstance(model.likelihood, GaussianLikelihood)
        for prior in model.likelihood.named_priors():
            self.assertIsInstance(prior[1].noise_prior, HorseshoePrior)

        for constraint_name, constraint in model.likelihood.named_constraints():
            self.assertIsInstance(constraint, Interval)
            np.testing.assert_almost_equal(constraint.lower_bound.numpy(), torch.tensor(np.exp(-25)).numpy())
            np.testing.assert_almost_equal(constraint.upper_bound.numpy(), torch.tensor(np.exp(2)).numpy())

        self.assertEqual(torch.tensor([0.0]), model.likelihood.raw_noise.data)
        noise_level = 1e-3
        model = get_gp(n_dims, rs, noise=1e-3)
        self.assertEqual(torch.tensor([noise_level]), model.likelihood.raw_noise.data)

    def test_predict(self):
        rs = np.random.RandomState(1)
        # cont
        X, Y, n_dims = get_cont_data(rs)
        # cat
        X, Y, cat_dims, cont_dims = get_cat_data(rs)

        for model in (get_gp(n_dims, rs), get_mixed_gp(cat_dims, cont_dims, rs)):
            model.train(X[:10], Y[:10])
            m_hat, v_hat = model.predict(X[10:])
            self.assertEqual(m_hat.shape, (10, 1))
            self.assertEqual(v_hat.shape, (10, 1))

    def test_train_do_optimize(self):
        # Check that do_optimize does not mess with the kernel hyperparameters given to the Gaussian process!
        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, rs)
        model._train(X[:10], Y[:10], do_optimize=False)

        fixture = np.array([0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        hypers = model.hypers

        np.testing.assert_array_almost_equal(hypers, fixture)

        model._train(X[:10], Y[:10], do_optimize=True)
        hypers = model.hypers
        self.assertFalse(np.any(hypers == fixture))

    @unittest.mock.patch("gpytorch.module.Module.pyro_sample_from_prior")
    def test_exception_1(self, fit_mock):
        # Check that training will not continue sampling if pyro raises an error
        class Dummy1:
            counter = 0

            def __call__(self):
                self.counter += 1
                raise RuntimeError("Unable to sample new cfgs")

        fit_mock.side_effect = Dummy1()

        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, rs)
        with self.assertRaises(RuntimeError):
            model._train(X[:10], Y[:10], do_optimize=True)

    @unittest.mock.patch("gpytorch.models.exact_gp.ExactGP.__init__")
    def test_exception_2(self, fit_mock):
        class Dummy2:
            counter = 0

            def __call__(self, train_inputs, train_targets, likelihood):
                self.counter += 1
                raise RuntimeError("Unable to initialize a new GP")

        fit_mock.side_effect = Dummy2()
        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, rs)
        with self.assertRaises(RuntimeError):
            model._train(X[:10], Y[:10], do_optimize=False)
        with self.assertRaises(RuntimeError):
            model._get_gaussian_process(X[:10], Y[:10])

    @unittest.mock.patch("gpytorch.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood.forward")
    def test_train_continue_on_linalg_error(self, fit_mock):
        # Check that training does not stop on a NotPSDError error, but that uncertainty is increased!
        class Dummy:
            counter = 0

            def __call__(self, function_dist, target, *params):
                if self.counter >= 13:
                    return None
                else:
                    self.counter += 1
                    raise NotPSDError

        fit_mock.side_effect = Dummy()

        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)

        model = get_gp(n_dims, rs)
        model._train(X[:10], Y[:10], do_optimize=True)

        fixture = np.array([0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        hypers = model.hypers

        np.testing.assert_array_almost_equal(hypers, fixture)

    def test_predict_with_actual_values(self):
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
            ],
            dtype=np.float64,
        )
        y = np.array([[0.1], [0.2], [9], [9.2], [100.0], [100.2], [109.0], [109.2]], dtype=np.float64)
        rs = np.random.RandomState(1)
        model = get_gp(3, rs)
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

        self.assertEqual(model.is_trained, True)

        mu_hat, var_hat = model.predict(X)
        for y_i, y_hat_i, mu_hat_i in zip(
            y.reshape((1, -1)).flatten(),
            mu_hat.reshape((1, -1)).flatten(),
            var_hat.reshape((1, -1)).flatten(),
        ):
            self.assertAlmostEqual(y_hat_i, y_i, delta=2)
            self.assertAlmostEqual(mu_hat_i, 0, delta=2)

        # Regression test that performance does not drastically decrease in the near future
        mu_hat, var_hat = model.predict(np.array([[10.0, 10.0, 10.0]]))

        self.assertAlmostEqual(mu_hat[0][0], 54.612500000000004)
        # There's a slight difference between my local installation and travis
        self.assertLess(abs(var_hat[0][0] - 1017.1374468449195), 15)

        # test other covariance results
        _, var_fc = model.predict(X, cov_return_type="full")
        self.assertEqual(var_fc.shape, (8, 8))
        _, var_sd = model.predict(X, cov_return_type="std")
        self.assertEqual(var_sd.shape, (8, 1))
        _, var_no = model.predict(np.array([[10.0, 10.0, 10.0]]), cov_return_type=None)
        self.assertIsNone(var_no)
        # check values
        _, var_fc = model.predict(np.array([[10.0, 10.0, 10.0]]), cov_return_type="full")
        self.assertAlmostEqual(var_fc[0][0], var_hat[0][0])
        _, var_sd = model.predict(np.array([[10.0, 10.0, 10.0]]), cov_return_type="std")
        self.assertAlmostEqual(var_sd[0][0] ** 2, var_hat[0][0])

        _, var_fc = model.predict(np.array([[10.0, 10.0, 10.0], [5.0, 5.0, 5.0]]), cov_return_type="full")
        self.assertEqual(var_fc.shape, (2, 2))

    def test_normalization(self):
        super(TestGPGPyTorch, self).test_normalization()

    def test_sampling_shape(self):
        X = np.arange(-5, 5, 0.1).reshape((-1, 1))
        X_test = np.arange(-5.05, 5.05, 0.1).reshape((-1, 1))
        for shape in (None, (-1, 1)):

            if shape is None:
                y = np.sin(X).flatten()
            else:
                y = np.sin(X).reshape(shape)

            rng = np.random.RandomState(1)
            for gp in (
                get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=False),
                get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=True),
            ):
                gp._train(X, y)
                func = gp.sample_functions(X_test=X_test, n_funcs=1)
                self.assertEqual(func.shape, (101, 1), msg=shape)
                func = gp.sample_functions(X_test=X_test, n_funcs=2)
                self.assertEqual(func.shape, (101, 2))
