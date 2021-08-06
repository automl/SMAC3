import unittest.mock

import numpy as np
import sklearn.datasets
import sklearn.model_selection

from smac.configspace import ConfigurationSpace, UniformFloatHyperparameter
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior

from test import requires_extra

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def get_gp(n_dimensions, rs, noise=1e-3, normalize_y=True, average_samples=False, n_iter=50):
    from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel

    cov_amp = ConstantKernel(
        2.0,
        constant_value_bounds=(1e-10, 2),
        prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rs),
    )
    exp_kernel = Matern(
        np.ones([n_dimensions]),
        [(np.exp(-10), np.exp(2)) for _ in range(n_dimensions)],
        nu=2.5,
        prior=None,
    )
    noise_kernel = WhiteKernel(
        noise_level=noise,
        noise_level_bounds=(1e-10, 2),
        prior=HorseshoePrior(scale=0.1, rng=rs),
    )
    kernel = cov_amp * exp_kernel + noise_kernel

    n_mcmc_walkers = 3 * len(kernel.theta)
    if n_mcmc_walkers % 2 == 1:
        n_mcmc_walkers += 1

    bounds = [(0., 1.) for _ in range(n_dimensions)]
    types = np.zeros(n_dimensions)

    configspace = ConfigurationSpace()
    for i in range(n_dimensions):
        configspace.add_hyperparameter(UniformFloatHyperparameter('x%d' % i, 0, 1))

    model = GaussianProcessMCMC(
        configspace=configspace,
        types=types,
        bounds=bounds,
        kernel=kernel,
        n_mcmc_walkers=n_mcmc_walkers,
        chain_length=n_iter,
        burnin_steps=n_iter,
        normalize_y=normalize_y,
        seed=rs.randint(low=1, high=10000),
        mcmc_sampler='emcee',
        average_samples=average_samples,
    )
    return model


@requires_extra('gpmcmc')
class TestGPMCMC(unittest.TestCase):
    def test_predict_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)
        model = get_gp(10, rs)

        X = rs.rand(10)
        self.assertRaisesRegex(ValueError, "Expected 2d array, got 1d array!",
                               model.predict, X)
        X = rs.rand(10, 10, 10)
        self.assertRaisesRegex(ValueError, "Expected 2d array, got 3d array!",
                               model.predict, X)

        X = rs.rand(10, 5)
        self.assertRaisesRegex(ValueError, "Rows in X should have 10 entries "
                                           "but have 5!",
                               model.predict, X)

    def test_gp_train(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)

        fixture = np.array([0.693147, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -6.907755])

        model = get_gp(10, rs)
        np.testing.assert_array_almost_equal(model.kernel.theta, fixture)
        model.train(X[:10], Y[:10])
        self.assertEqual(len(model.models), 36)

        for base_model in model.models:
            theta = base_model.gp.kernel.theta
            theta_ = base_model.gp.kernel_.theta
            # Test that the kernels of the base GP are actually changed!
            np.testing.assert_array_almost_equal(theta, theta_)
            self.assertFalse(np.any(theta == fixture))
            self.assertFalse(np.any(theta_ == fixture))

    def test_gp_train_posterior_mean(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)

        fixture = np.array([0.693147, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -6.907755])

        model = get_gp(10, rs, average_samples=True)
        np.testing.assert_array_almost_equal(model.kernel.theta, fixture)
        model.train(X[:10], Y[:10])

        for base_model in model.models:
            theta = base_model.gp.kernel.theta
            theta_ = base_model.gp.kernel_.theta
            # Test that the kernels of the base GP are actually changed!
            np.testing.assert_array_almost_equal(theta, theta_)
            self.assertFalse(np.any(theta == fixture))
            self.assertFalse(np.any(theta_ == fixture))

        self.assertEqual(len(model.models), 1)

    def test_predict(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)
        model = get_gp(10, rs)
        model.train(X[:10], Y[:10])
        m_hat, v_hat = model.predict(X[10:])
        self.assertEqual(m_hat.shape, (10, 1))
        self.assertEqual(v_hat.shape, (10, 1))

    @unittest.mock.patch.object(GaussianProcessMCMC, 'predict')
    def test_predict_marginalized_over_instances_no_features(self, rf_mock):
        """The GP should fall back to the regular predict() method."""

        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)
        model = get_gp(10, rs)
        model.train(X[:10], Y[:10])
        model.predict(X[10:])
        self.assertEqual(rf_mock.call_count, 1)

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
        model = get_gp(3, rs, noise=1e-10, n_iter=200)
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

        y_hat, var_hat = model.predict(X)
        for y_i, y_hat_i, var_hat_i in zip(
            y.reshape((1, -1)).flatten(), y_hat.reshape((1, -1)).flatten(), var_hat.reshape((1, -1)).flatten(),
        ):
            # Chain length too short to get excellent predictions, apparently there's a lot of predictive variance
            self.assertAlmostEqual(y_i, y_hat_i, delta=1)
            self.assertAlmostEqual(var_hat_i, 0, delta=500)

        # Regression test that performance does not drastically decrease in the near future
        y_hat, var_hat = model.predict(np.array([[10, 10, 10]]))
        self.assertAlmostEqual(y_hat[0][0], 54.613410745846785, delta=0.1)
        # Massive variance due to internally used law of total variances, also a massive difference locally and on
        # travis-ci
        self.assertLessEqual(abs(var_hat[0][0]) - 3700, 200, msg=str(var_hat))

    def test_gp_on_sklearn_data(self):
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        # Normalize such that the bounds in get_gp hold
        X = X / X.max(axis=0)
        rs = np.random.RandomState(1)
        model = get_gp(X.shape[1], rs, noise=1e-10, normalize_y=True)
        cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

        maes = [6.841565457149357281, 7.4943401900804902144]

        for i, (train_split, test_split) in enumerate(cv.split(X, y)):
            X_train = X[train_split]
            y_train = y[train_split]
            X_test = X[test_split]
            y_test = y[test_split]
            model.train(X_train, y_train)
            y_hat, mu_hat = model.predict(X_test)
            mae = np.mean(np.abs(y_hat - y_test), dtype=np.float128)
            self.assertAlmostEqual(mae, maes[i])

    def test_normalization(self):
        X = np.arange(-5, 5, 0.1).reshape((-1, 1))
        X_test = np.arange(-5.05, 5.05, 0.1).reshape((-1, 1))
        y = np.sin(X)
        rng = np.random.RandomState(1)
        gp = get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=False)
        gp._train(X, y, do_optimize=False)
        self.assertFalse(gp.models[0].normalize_y)
        self.assertFalse(hasattr(gp.models[0], 'mean_y_'))
        mu_hat, var_hat = gp.predict(X_test)
        gp_norm = get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=True)
        gp_norm._train(X, y, do_optimize=False)
        self.assertTrue(gp_norm.models[0].normalize_y)
        self.assertTrue(hasattr(gp_norm.models[0], 'mean_y_'))
        mu_hat_prime, var_hat_prime = gp_norm.predict(X_test)
        np.testing.assert_array_almost_equal(mu_hat, mu_hat_prime, decimal=4)
        np.testing.assert_array_almost_equal(var_hat, var_hat_prime, decimal=4)
