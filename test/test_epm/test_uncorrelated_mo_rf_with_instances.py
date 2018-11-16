import sys
import unittest

import numpy as np

from smac.epm.uncorrelated_mo_rf_with_instances import \
    UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.epm.rf_with_instances import RandomForestWithInstances

if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock


class TestUncorrelatedMultiObjectiveWrapper(unittest.TestCase):
    def test_train_and_predict_with_rf(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 2)
        model = UncorrelatedMultiObjectiveRandomForestWithInstances(
            ['cost', 'ln(runtime)'],
            types=np.zeros((10, ), dtype=np.uint),
            bounds=np.array([
                (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan),
                (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan)
            ], dtype=object),
            rf_kwargs={'seed': 1},
            pca_components=5
        )
        self.assertEqual(model.estimators[0].seed, 1)
        self.assertEqual(model.estimators[1].seed, 1)
        self.assertEqual(model.pca_components, 5)
        model.train(X[:10], Y)
        m, v = model.predict(X[10:])
        self.assertEqual(m.shape, (10, 2))
        self.assertEqual(v.shape, (10, 2))

    # We need to track how often the base model was called!
    @mock.patch.object(RandomForestWithInstances, 'predict')
    def test_predict_mocked(self, rf_mock):
        class SideEffect(object):
            def __init__(self):
                self.counter = 0

            def __call__(self, X):
                self.counter += 1
                # Mean and variance
                rval = np.array([self.counter] * X.shape[0])
                return rval, rval

        rf_mock.side_effect = SideEffect()

        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 3)
        model = UncorrelatedMultiObjectiveRandomForestWithInstances(
            ['cost', 'ln(runtime)', 'foo'],
            types=np.zeros((10,), dtype=np.uint),
            bounds=np.array([
                (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan),
                (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan)
            ], dtype=object),
        )

        model.train(X[:10], Y[:10])
        m_hat, v_hat = model.predict(X[10:])
        self.assertEqual(m_hat.shape, (10, 3))
        self.assertEqual(v_hat.shape, (10, 3))
        self.assertEqual(rf_mock.call_count, 3)
        for i in range(10):
            for j in range(3):
                self.assertEqual(m_hat[i][j], j + 1)
                self.assertEqual(v_hat[i][j], j + 1)
