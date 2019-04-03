import unittest
import unittest.mock

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import skopt.learning.gaussian_process

from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.epm.gp_default_priors import DefaultPrior


def get_gp(n_dimensions, rs, noise=1e-3, normalize_y=True):
    cov_amp = skopt.learning.gaussian_process.kernels.ConstantKernel(2.0, constant_value_bounds=(1e-10, 2))
    exp_kernel = skopt.learning.gaussian_process.kernels.Matern(
        np.ones([n_dimensions]),
        [(np.exp(-10), np.exp(2)) for _ in range(n_dimensions)],
        nu=2.5,
    )
    noise_kernel = skopt.learning.gaussian_process.kernels.WhiteKernel(
        noise_level=noise,
        noise_level_bounds=(1e-10, 2),
    )
    kernel = cov_amp * exp_kernel + noise_kernel

    prior = DefaultPrior(len(kernel.theta), rng=rs)

    n_mcmc_walkers = 3 * len(kernel.theta)
    if n_mcmc_walkers % 2 == 1:
        n_mcmc_walkers += 1

    bounds = [(0., 1.) for _ in range(n_dimensions)]
    types = np.zeros(n_dimensions)

    model = GaussianProcessMCMC(
        types=types,
        bounds=bounds,
        kernel=kernel,
        prior=prior,
        n_mcmc_walkers=n_mcmc_walkers,
        chain_length=20,
        burnin_steps=100,
        normalize_y=normalize_y,
        seed=rs.randint(low=1, high=10000),
    )
    return model


class TestGPMCMC(unittest.TestCase):
    def test_predict_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)
        model = get_gp(10, rs)

        X = rs.rand(10)
        self.assertRaisesRegexp(ValueError, "Expected 2d array, got 1d array!",
                                model.predict, X)
        X = rs.rand(10, 10, 10)
        self.assertRaisesRegexp(ValueError, "Expected 2d array, got 3d array!",
                                model.predict, X)

        X = rs.rand(10, 5)
        self.assertRaisesRegexp(ValueError, "Rows in X should have 10 entries "
                                            "but have 5!",
                                model.predict, X)

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
        model = get_gp(3, rs, noise=1e-10)
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

        y_hat, var_hat = model.predict(X)
        for y_i, y_hat_i, var_hat_i in zip(
            y.reshape((1, -1)).flatten(), y_hat.reshape((1, -1)).flatten(), var_hat.reshape((1, -1)).flatten(),
        ):
            # Chain length too short to get excellent predictions
            self.assertAlmostEqual(y_i, y_hat_i, delta=0.1)
            self.assertAlmostEqual(var_hat_i, 0, delta=0.1)

        # Regression test that performance does not drastically decrease in the near future
        y_hat, var_hat = model.predict(np.array([[10, 10, 10]]))
        self.assertAlmostEqual(y_hat[0][0], 54.61249999999999)
        # Massive variance due to internally used law of total variances, also a massive difference locally and on
        # travis-cis
        self.assertLessEqual(abs(var_hat[0][0] - 3500), 50)

    def test_gp_on_sklearn_data(self):
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        # Normalize such that the bounds in get_gp hold
        X = X / X.max(axis=0)
        rs = np.random.RandomState(1)
        model = get_gp(X.shape[1], rs, noise=1e-10, normalize_y=True)
        cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

        maes = [10.179732235886636338, 7.7493806362044323605]

        for i, (train_split, test_split) in enumerate(cv.split(X, y)):
            X_train = X[train_split]
            y_train = y[train_split]
            X_test = X[test_split]
            y_test = y[test_split]
            model.train(X_train, y_train)
            print(model.hypers)
            y_hat, mu_hat = model.predict(X_test)
            print(y_hat, np.mean(np.abs(np.mean(y_train) - y_test), dtype=np.float128))
            mae = np.mean(np.abs(y_hat - y_test), dtype=np.float128)
            self.assertAlmostEqual(mae, maes[i])

    def test_normalization(self):
        X = np.arange(-5, 5, 0.1).reshape((-1, 1))
        X_test = np.arange(-5.05, 5.05, 0.1).reshape((-1, 1))
        y = np.sin(X)
        rng = np.random.RandomState(1)
        gp = get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=False)
        gp._train(X, y, do_optimize=False)
        mu_hat, var_hat = gp.predict(X_test)
        gp_norm = get_gp(n_dimensions=1, rs=rng, noise=1e-10, normalize_y=True)
        gp_norm._train(X, y, do_optimize=False)
        mu_hat_prime, var_hat_prime = gp_norm.predict(X_test)
        np.testing.assert_array_almost_equal(mu_hat, mu_hat_prime, decimal=4)
        np.testing.assert_array_almost_equal(var_hat, var_hat_prime, decimal=4)
