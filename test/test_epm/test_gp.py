import unittest
import unittest.mock

import scipy.optimize
import numpy as np
import sklearn.datasets
import sklearn.model_selection

from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gp_default_priors import DefaultPrior
from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel


def get_gp(n_dimensions, rs, noise=1e-3):
    cov_amp = ConstantKernel(2.0, constant_value_bounds=(1e-10, 2))
    exp_kernel = Matern(
        np.ones([n_dimensions]),
        [(np.exp(-10), np.exp(2)) for _ in range(n_dimensions)],
        nu=2.5,
    )
    noise_kernel = WhiteKernel(
        noise_level=noise,
        noise_level_bounds=(1e-10, 2),
    )
    kernel = cov_amp * exp_kernel + noise_kernel

    prior = DefaultPrior(len(kernel.theta) + 1, rng=rs)

    bounds = [(0., 1.) for _ in range(n_dimensions)]
    types = np.zeros(n_dimensions)

    model = GaussianProcess(
        bounds=bounds,
        types=types,
        kernel=kernel,
        prior=prior,
        seed=rs.randint(low=1, high=10000),
        normalize_y=False,
    )
    return model


class TestGP(unittest.TestCase):
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

    @unittest.mock.patch.object(GaussianProcess, 'predict')
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
        model = get_gp(3, rs)
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

        y_hat, mu_hat = model.predict(X)
        for y_i, y_hat_i, mu_hat_i in zip(
            y.reshape((1, -1)).flatten(), y_hat.reshape((1, -1)).flatten(), mu_hat.reshape((1, -1)).flatten(),
        ):
            self.assertAlmostEqual(y_i, y_hat_i, delta=2)
            self.assertAlmostEqual(mu_hat_i, 0, delta=2)

        # Regression test that performance does not drastically decrease in the near future
        y_hat, mu_hat = model.predict(np.array([[10, 10, 10]]))
        self.assertAlmostEqual(y_hat[0][0], 1.2579769765743784e-05)
        self.assertAlmostEqual(mu_hat[0][0], 8.701826756269622)

    def test_gp_on_sklearn_data(self):
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        # Normalize such that the bounds in get_gp (10) hold
        X = X / X.max(axis=0)
        rs = np.random.RandomState(1)
        model = get_gp(X.shape[1], rs)
        cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

        maes = [8.6835211971259218715, 9.087941533717404484]

        for i, (train_split, test_split) in enumerate(cv.split(X, y)):
            X_train = X[train_split]
            y_train = y[train_split]
            X_test = X[test_split]
            y_test = y[test_split]
            model.train(X_train, y_train)
            y_hat, mu_hat = model.predict(X_test)
            mae = np.mean(np.abs(y_hat - y_test), dtype=np.float128)
            self.assertAlmostEqual(mae, maes[i])

    def test_nll(self):
        rs = np.random.RandomState(1)
        gp = get_gp(1, rs)
        gp.train(np.array([[0], [1]]), np.array([0, 1]))
        n_above_1 = 0
        for i in range(1000):
            theta = np.array([rs.uniform(1e-10, 10), rs.uniform(-10, 2), rs.uniform(-10, 1)])  # Values from the default prior
            error = scipy.optimize.check_grad(lambda x: gp._nll(x)[0], lambda x: gp._nll(x)[1], theta, epsilon=1e-5)
            # print(error, theta, gp._nll(theta)[1], scipy.optimize.approx_fprime(theta, lambda x: gp._nll(x)[0], 1e-5))
            if error > 0.1:
                n_above_1 += 1
        self.assertLessEqual(n_above_1, 10)
