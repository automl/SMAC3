import unittest
import unittest.mock

import george
import numpy as np
import sklearn.datasets
import sklearn.model_selection

from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.epm.gp_default_priors import DefaultPrior


def get_gp(n_dimensions, rs, noise=-8):
    cov_amp = 2

    initial_ls = np.ones([n_dimensions])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=n_dimensions)
    kernel = cov_amp * exp_kernel

    prior = DefaultPrior(len(kernel) + 1, rng=rs)

    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    bounds = [(0., 1.) for _ in range(n_dimensions)]
    types = np.zeros(n_dimensions)

    model = GaussianProcessMCMC(
        types=types,
        bounds=bounds,
        kernel=kernel,
        prior=prior,
        n_hypers=n_hypers,
        chain_length=20,
        burnin_steps=100,
        normalize_input=True,
        normalize_output=True,
        rng=rs, noise=noise,
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
        model = get_gp(3, rs, noise=-8)
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
        # Massive variance due to internally used law of total variances
        self.assertAlmostEqual(var_hat[0][0], 27691.09369406573)

    def test_gp_on_sklearn_data(self):
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        # Normalize such that the bounds in get_gp hold
        X = X / X.max(axis=0)
        rs = np.random.RandomState(1)
        model = get_gp(X.shape[1], rs)
        cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

        maes = [9.1921780939101723415, 8.929528339491875724]

        for i, (train_split, test_split) in enumerate(cv.split(X, y)):
            X_train = X[train_split]
            y_train = y[train_split]
            X_test = X[test_split]
            y_test = y[test_split]
            model.train(X_train, y_train)
            y_hat, mu_hat = model.predict(X_test)
            mae = np.mean(np.abs(y_hat - y_test), dtype=np.float128)
            self.assertAlmostEqual(mae, maes[i])
