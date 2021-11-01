import unittest.mock

import numpy as np
import sklearn

from smac.configspace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter
from smac.epm.gaussian_process_gpytorch import GaussianProcessGPyTorch, ExactGPModel
from botorch.models.kernels.categorical import CategoricalKernel
import torch
import pyro

from gpytorch.kernels import ProductKernel, MaternKernel, ScaleKernel
from gpytorch.priors import LogNormalPrior, HorseshoePrior, UniformPrior
from gpytorch.constraints.constraints import Interval
from gpytorch.utils.errors import NotPSDError

from test import requires_extra

from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood

torch.manual_seed(0)
pyro.set_rng_seed(0)


def get_cont_data(rs):
    X = rs.rand(20, 10)
    Y = rs.rand(10, 1)
    n_dims = 10
    return X, Y, n_dims


def get_cat_data(rs):
    X_cont = rs.rand(20, 5)
    X_cat = rs.randint(low=0, high=3, size=(20, 5))
    X = np.concatenate([X_cat, X_cont], axis=1)
    Y = rs.rand(10, 1)
    cat_dims = [0, 1, 2, 3, 4]
    cont_dims = [5, 6, 7, 8, 9]
    return X, Y, cat_dims, cont_dims


@requires_extra('gpytorch')
def get_gp(n_dimensions, rs, noise=None, normalize_y=True) -> GaussianProcessGPyTorch:
    exp_kernel = MaternKernel(2.5,
                              lengthscale_constraint=Interval(
                                  torch.tensor(np.exp(-6.754111155189306).repeat(n_dimensions)),
                                  torch.tensor(np.exp(0.0858637988771976).repeat(n_dimensions)),
                                  transform=None,
                                  initial_value=1.0
                              ),
                              ard_num_dims=n_dimensions,
                              active_dims=torch.arange(n_dimensions),
                              lengthscale_prior=UniformPrior(np.exp(-6.754111155189306), np.exp(0.858637988771976))
                              ).double()

    kernel = ScaleKernel(exp_kernel,
                         outputscale_constraint=Interval(
                             np.exp(-10.),
                             np.exp(2.),
                             transform=None,
                             initial_value=2.0
                         ),
                         outputscale_prior=LogNormalPrior(0.0, 1.0)).double()
    if noise is None:
        likelihood = None
    else:
        noise_prior = HorseshoePrior(0.1)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=Interval(np.exp(-25), np.exp(2), transform=None)
        ).double()
        likelihood.initialize(noise=noise)

    bounds = [(0., 1.) for _ in range(n_dimensions)]
    types = np.zeros(n_dimensions)

    configspace = ConfigurationSpace()
    for i in range(n_dimensions):
        configspace.add_hyperparameter(UniformFloatHyperparameter('x%d' % i, 0, 1))

    model = GaussianProcessGPyTorch(
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


@requires_extra('gpytorch')
def get_mixed_gp(cat_dims, cont_dims, rs, normalize_y=True):
    cat_dims = np.array(cat_dims, dtype=np.int)
    cont_dims = np.array(cont_dims, dtype=np.int)
    n_dimensions = len(cat_dims) + len(cont_dims)
    exp_kernel = MaternKernel(2.5,
                              lengthscale_constraint=Interval(
                                  torch.tensor(np.exp(-6.754111155189306).repeat(cont_dims.shape[-1])),
                                  torch.tensor(np.exp(0.0858637988771976).repeat(cont_dims.shape[-1])),
                                  transform=None,
                                  initial_value=1.0
                              ),
                              ard_num_dims=cont_dims.shape[-1],
                              active_dims=tuple(cont_dims),
                              lengthscale_prior=UniformPrior(np.exp(-6.754111155189306), np.exp(0.858637988771976))
                              ).double()

    ham_kernel = CategoricalKernel(
        lengthscale_constraint=Interval(
            torch.tensor(np.exp(-6.754111155189306).repeat(cat_dims.shape[-1])),
            torch.tensor(np.exp(0.0858637988771976).repeat(cat_dims.shape[-1])),
            transform=None,
            initial_value=1.0
        ),
        ard_num_dims=cat_dims.shape[-1],
        active_dims=tuple(cat_dims),
        lengthscale_prior=UniformPrior(np.exp(-6.754111155189306), np.exp(0.858637988771976))
    ).double()

    kernel = ProductKernel(exp_kernel, ham_kernel)

    kernel = ScaleKernel(kernel,
                         outputscale_constraint=Interval(
                             np.exp(-10.),
                             np.exp(2.),
                             transform=None,
                             initial_value=2.0
                         ),
                         outputscale_prior=LogNormalPrior(0.0, 1.0))

    bounds = [0] * n_dimensions
    types = np.zeros(n_dimensions)
    for c in cont_dims:
        bounds[c] = (0., 1.)
    for c in cat_dims:
        types[c] = 3
        bounds[c] = (3, np.nan)

    cs = ConfigurationSpace()
    for c in cont_dims:
        cs.add_hyperparameter(UniformFloatHyperparameter('X%d' % c, 0, 1))
    for c in cat_dims:
        cs.add_hyperparameter(CategoricalHyperparameter('X%d' % c, [0, 1, 2, 3]))

    model = GaussianProcessGPyTorch(
        configspace=cs,
        bounds=bounds,
        types=types,
        kernel=kernel,
        seed=rs.randint(low=1, high=10000),
        normalize_y=normalize_y,
    )
    return model


@requires_extra('gpytorch')
class TestGPGPyTorch(unittest.TestCase):
    def test_likelihood(self):
        rs = np.random.RandomState(1)
        X, Y, n_dims = get_cont_data(rs)
        model = get_gp(n_dims, rs)
        self.assertIsInstance(model.likelihood, GaussianLikelihood)
        self.assertEqual(torch.tensor([0.]), model.likelihood.raw_noise.data)
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

        fixture = np.array([0.0, 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        hypers = model.hypers

        np.testing.assert_array_almost_equal(hypers, fixture)


        model._train(X[:10], Y[:10], do_optimize=True)
        hypers = model.hypers
        self.assertFalse(np.any(hypers == fixture))


    @unittest.mock.patch('gpytorch.mlls.exact_marginal_log_likelihood.ExactMarginalLogLikelihood.forward')
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

        fixture = np.array([0.0, 2., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

        hypers = model.hypers

        np.testing.assert_array_almost_equal(
            hypers, fixture
        )

    def test_predict_with_actual_values(self):
        X = np.array([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.]], dtype=np.float64)
        y = np.array([
            [.1],
            [.2],
            [9],
            [9.2],
            [100.],
            [100.2],
            [109.],
            [109.2]], dtype=np.float64)
        rs = np.random.RandomState(1)
        model = get_gp(3, rs)
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

        mu_hat, var_hat = model.predict(X)
        for y_i, y_hat_i, mu_hat_i in zip(
            y.reshape((1, -1)).flatten(), mu_hat.reshape((1, -1)).flatten(), var_hat.reshape((1, -1)).flatten(),
        ):
            self.assertAlmostEqual(y_hat_i, y_i, delta=2)
            self.assertAlmostEqual(mu_hat_i, 0, delta=2)

        # Regression test that performance does not drastically decrease in the near future
        mu_hat, var_hat = model.predict(np.array([[10., 10., 10.]]))

        self.assertAlmostEqual(mu_hat[0][0], 54.612500000000004)
        # There's a slight difference between my local installation and travis
        self.assertLess(abs(var_hat[0][0] - 1017.1374468449195), 2)


        # test other covariance results
        _, var_fc = model.predict(X, cov_return_type='full_cov')
        self.assertEqual(var_fc.shape, (8, 8))
        _, var_sd = model.predict(X, cov_return_type='diagonal_std')
        self.assertEqual(var_sd.shape, (8, 1))
        _, var_no = model.predict(np.array([[10., 10., 10.]]), cov_return_type=None)
        self.assertIsNone(var_no)
        # check values
        _, var_fc = model.predict(np.array([[10., 10., 10.]]), cov_return_type='full_cov')
        self.assertAlmostEqual(var_fc[0][0], var_hat[0][0])
        _, var_sd = model.predict(np.array([[10., 10., 10.]]), cov_return_type='diagonal_std')
        self.assertAlmostEqual(var_sd[0][0] ** 2, var_hat[0][0])

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
