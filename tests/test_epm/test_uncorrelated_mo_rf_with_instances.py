import unittest
from unittest import mock

import numpy as np

import smac.configspace
from smac.epm.uncorrelated_mo_rf_with_instances import \
    UncorrelatedMultiObjectiveRandomForestWithInstances
from smac.epm.rf_with_instances import RandomForestWithInstances

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestUncorrelatedMultiObjectiveWrapper(unittest.TestCase):

    def _get_cs(self, n_dimensions):
        configspace = smac.configspace.ConfigurationSpace()
        for i in range(n_dimensions):
            configspace.add_hyperparameter(smac.configspace.UniformFloatHyperparameter('x%d' % i, 0, 1))
        return configspace

    def test_train_and_predict_with_rf(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 2)
        model = UncorrelatedMultiObjectiveRandomForestWithInstances(
            configspace=self._get_cs(10),
            target_names=['cost', 'ln(runtime)'],
            types=np.zeros((10, ), dtype=np.uint),
            bounds=[
                (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan),
                (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan)
            ],
            seed=1,
            rf_kwargs={'seed': 1},
            pca_components=5,
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
            target_names=['cost', 'ln(runtime)', 'foo'],
            configspace=self._get_cs(10),
            types=np.zeros((10,), dtype=np.uint),
            bounds=[
                (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan),
                (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan), (0, np.nan)
            ],
            seed=1,
            rf_kwargs={'seed': 1},
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
