import unittest
import unittest.mock

import george
import numpy as np
import sklearn.datasets
import sklearn.model_selection

from smac.epm.rf_with_instances_hpo import RandomForestWithInstancesHPO


def get_rf(n_dimensions, rs):
    bounds = [(0., 1.) for _ in range(n_dimensions)]
    types = np.zeros(n_dimensions)

    model = RandomForestWithInstancesHPO(
        types=types, bounds=bounds, log_y=False, bootstrap=False, n_iters=5, n_splits=5,
    )
    return model


class TestRandomForestWithInstancesHPO(unittest.TestCase):
    def test_predict_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)
        model = get_rf(10, rs)

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
        model = get_rf(10, rs)
        model.train(X[:10], Y[:10])
        m_hat, v_hat = model.predict(X[10:])
        self.assertEqual(m_hat.shape, (10, 1))
        self.assertEqual(v_hat.shape, (10, 1))

    @unittest.mock.patch.object(RandomForestWithInstancesHPO, 'predict')
    def test_predict_marginalized_over_instances_no_features(self, rf_mock):
        """The GP should fall back to the regular predict() method."""

        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)
        model = get_rf(10, rs)
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
        model = get_rf(3, rs)
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

        y_hat, var_hat = model.predict(X)
        for y_i, y_hat_i, var_hat_i in zip(
            y.reshape((1, -1)).flatten(), y_hat.reshape((1, -1)).flatten(), var_hat.reshape((1, -1)).flatten(),
        ):
            # Chain length too short to get excellent predictions
            self.assertAlmostEqual(y_i, y_hat_i, delta=0.2)
            self.assertAlmostEqual(var_hat_i, 0, delta=0.5)

        # RF predicts the rightmost point
        y_hat, var_hat = model.predict(np.array([[10., 10., 10.]]))
        self.assertAlmostEqual(y_hat[0][0], 109.2)
        # No variance because the data point is outside of the observed data
        print(var_hat)
        self.assertAlmostEqual(var_hat[0][0], 0)

    def test_rf_on_sklearn_data(self):
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        # Normalize such that the bounds in get_gp hold
        X = X / X.max(axis=0)
        rs = np.random.RandomState(1)
        model = get_rf(X.shape[1], rs)
        cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

        maes = [9.129075955836567871, 9.367454539503001523]

        for i, (train_split, test_split) in enumerate(cv.split(X, y)):
            X_train = X[train_split]
            y_train = y[train_split]
            X_test = X[test_split]
            y_test = y[test_split]
            model.train(X_train, y_train)
            y_hat, mu_hat = model.predict(X_test)
            mae = np.mean(np.abs(y_hat - y_test), dtype=np.float128)
            self.assertAlmostEqual(mae, maes[i])
