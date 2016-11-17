import unittest
import sys

import numpy as np

from smac.epm.rf_with_instances import RandomForestWithInstances

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock


class TestRFWithInstances(unittest.TestCase):
    def test_predict_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)

        model = RandomForestWithInstances(np.zeros((10,), dtype=np.uint))
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
        model = RandomForestWithInstances(np.zeros((10,), dtype=np.uint))
        model.train(X[:10], Y[:10])
        m_hat, v_hat = model.predict(X[10:])
        self.assertEqual(m_hat.shape, (10, 1))
        self.assertEqual(v_hat.shape, (10, 1))

    def test_predict_marginalized_over_instances_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)

        model = RandomForestWithInstances(np.zeros((10,), dtype=np.uint),
                                          instance_features=rs.rand(10, 2))
        X = rs.rand(10)
        self.assertRaisesRegexp(ValueError, "Expected 2d array, got 1d array!",
                                model.predict_marginalized_over_instances, X)
        X = rs.rand(10, 10, 10)
        self.assertRaisesRegexp(ValueError, "Expected 2d array, got 3d array!",
                                model.predict_marginalized_over_instances, X)

        X = rs.rand(10, 5)
        self.assertRaisesRegexp(ValueError, "Rows in X should have 8 entries "
                                            "but have 5!",
                                model.predict_marginalized_over_instances, X)

    @mock.patch.object(RandomForestWithInstances, 'predict')
    def test_predict_marginalized_over_instances_no_features(self, rf_mock):
        """The RF should fall back to the regular predict() method."""

        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)
        model = RandomForestWithInstances(np.zeros((10,), dtype=np.uint))
        model.train(X[:10], Y[:10])
        model.predict(X[10:])
        self.assertEqual(rf_mock.call_count, 1)

    def test_predict_marginalized_over_instances(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        F = rs.rand(10, 5)
        Y = rs.rand(len(X) * len(F), 1)
        X_ = rs.rand(200, 15)

        model = RandomForestWithInstances(np.zeros((15,), dtype=np.uint),
                                          instance_features=F)
        model.train(X_, Y)
        means, vars = model.predict_marginalized_over_instances(X)
        self.assertEqual(means.shape, (20, 1))
        self.assertEqual(vars.shape, (20, 1))

    @mock.patch.object(RandomForestWithInstances, 'predict')
    def test_predict_marginalized_over_instances_mocked(self, rf_mock):
        """Use mock to count the number of calls to predict()"""
        class SideEffect(object):
            def __call__(self, X):
                # Numpy array of number 0 to X.shape[0]
                rval = np.array(list(range(X.shape[0]))).reshape((-1, 1))
                # Return mean and variance
                return rval, rval

        rf_mock.side_effect = SideEffect()

        rs = np.random.RandomState(1)
        F = rs.rand(10, 5)

        model = RandomForestWithInstances(np.zeros((15,), dtype=np.uint),
                                          instance_features=F)
        means, vars = model.predict_marginalized_over_instances(rs.rand(11, 10))
        self.assertEqual(rf_mock.call_count, 11)
        self.assertEqual(means.shape, (11, 1))
        self.assertEqual(vars.shape, (11, 1))
        for i in range(11):
            self.assertEqual(means[i], 4.5)
            self.assertEqual(vars[i], 4.5)