import unittest
import sys

import numpy as np
import pyrfr.regression

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter

from smac.epm.rf_with_instances import RandomForestWithInstances, RandomForestClassifierWithInstances
from smac.epm.rf_with_instances import RandomForestClassifierWithInstances
import smac
from smac.utils.util_funcs import get_types


if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock


class TestRFClassifierWithInstances(unittest.TestCase):
    def test_predict_marginalized_over_instances(self):
        # 10 instances
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        F = rs.rand(10, 5)

        Y = rs.random_integers(low=0,high=1, size=(len(X) * len(F), 1))
        X_ = rs.rand(len(X) * len(F), 15)

        model = RandomForestClassifierWithInstances(instance_features=F)
        model.train(X_, Y)
        products_classprob_over_instances = model.predict_marginalized_over_instances(X)

        # check that the probabilities are between 0 and 1
        for prob in np.nditer(products_classprob_over_instances):
            self.assertLessEqual(prob, 1)
            self.assertGreaterEqual(prob, 0)
        # check the shape
        self.assertEqual(products_classprob_over_instances.shape, (20, 2))
        # 1 or no instance
        # We train and predict on the same set
        X = np.array([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.]], dtype=np.float64)
        Y = np.array([
            [0],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [1]], dtype=np.float64)
        F = np.array([[42, 43, 44]])
        X_ = np.array([
            np.hstack(([0., 0., 0.], F[0])),
            np.hstack(([0., 0., 1.], F[0])),
            np.hstack(([0., 1., 0.], F[0])),
            np.hstack(([0., 1., 1.], F[0])),
          np.hstack(([1., 0., 0.], F[0])),
          np.hstack(([1., 0., 1.], F[0])),
          np.hstack(([1., 1., 0.], F[0])),
          np.hstack(([1., 1., 1.], F[0]))], dtype=np.float64)
        model = RandomForestClassifierWithInstances(instance_features=F)
        model.train(X_, Y)
        products_classprob_over_one_instance = model.predict_marginalized_over_instances(X)
        model = RandomForestClassifierWithInstances(instance_features=None)
        class_labels_over_one_instance = np.around(products_classprob_over_one_instance)
        
        np.testing.assert_equal(class_labels_over_one_instance[:,1], Y[:,0])

        model.train(X, Y)
        products_classprob_without_instances = model.predict_marginalized_over_instances(X)
        class_labels_without_instances = np.around(products_classprob_without_instances)
        np.testing.assert_equal(class_labels_over_one_instance, class_labels_without_instances)


class TestRFWithInstances(unittest.TestCase):
    def test_predict_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)

        model = RandomForestWithInstances(np.zeros((10,), dtype=np.uint), bounds=np.array(
            list(map(lambda x: (0, 10), range(10))), dtype=object))
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
        model = RandomForestWithInstances(np.zeros((10,), dtype=np.uint), bounds=np.array(
                list(map(lambda x: (0, 10), range(10))), dtype=object))
        model.train(X[:10], Y[:10])
        m_hat, v_hat = model.predict(X[10:])
        self.assertEqual(m_hat.shape, (10, 1))
        self.assertEqual(v_hat.shape, (10, 1))

    def test_train_with_pca(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 20)
        F = rs.rand(10, 10)
        Y = rs.rand(20, 1)
        model = RandomForestWithInstances(np.zeros((20,), dtype=np.uint),
                                          np.array(list(map(lambda x: (0, 10), range(10))), dtype=object),
                                          pca_components=2,
                                          instance_features=F)
        model.train(X, Y)
        
        self.assertEqual(model.n_params, 10)
        self.assertEqual(model.n_feats, 10)
        self.assertIsNotNone(model.pca)
        self.assertIsNotNone(model.scaler)
        
    def test_predict_marginalized_over_instances_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)

        model = RandomForestWithInstances(np.zeros((10,), dtype=np.uint),
                                          instance_features=rs.rand(10, 2),
                                          bounds=np.array(list(map(lambda x: (0, 10), range(10))), dtype=object))
        X = rs.rand(10)
        self.assertRaisesRegexp(ValueError, "Expected 2d array, got 1d array!",
                                model.predict_marginalized_over_instances, X)
        X = rs.rand(10, 10, 10)
        self.assertRaisesRegexp(ValueError, "Expected 2d array, got 3d array!",
                                model.predict_marginalized_over_instances, X)

    @mock.patch.object(RandomForestWithInstances, 'predict')
    def test_predict_marginalized_over_instances_no_features(self, rf_mock):
        """The RF should fall back to the regular predict() method."""

        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)
        model = RandomForestWithInstances(np.zeros((10,), dtype=np.uint), bounds=np.array(
            list(map(lambda x: (0, 10), range(10))), dtype=object))
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
                                          instance_features=F,
                                          bounds=np.array(list(map(lambda x: (0, 10), range(10))), dtype=object))
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
                                          instance_features=F,
                                          bounds=np.array(list(map(lambda x: (0, 10), range(10))), dtype=object))
        means, vars = model.predict_marginalized_over_instances(rs.rand(11, 10))
        self.assertEqual(rf_mock.call_count, 11)
        self.assertEqual(means.shape, (11, 1))
        self.assertEqual(vars.shape, (11, 1))
        for i in range(11):
            self.assertEqual(means[i], 4.5)
            self.assertEqual(vars[i], 4.5)

    def test_predict_with_actual_values(self):
        print()
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
        # print(X.shape, y.shape)
        model = RandomForestWithInstances(types=np.array([0, 0, 0], dtype=np.uint),
                                          bounds=np.array([(0, np.nan), (0, np.nan), (0, np.nan)], dtype=object),
                                          instance_features=None, seed=12345,
                                          ratio_features=1.0)
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))
        # for idx, x in enumerate(X):
        #     print(model.rf.all_leaf_values(x))
        #     print(x, model.predict(np.array([x]))[0], y[idx])

        y_hat, _ = model.predict(X)
        for y_i, y_hat_i in zip(y.reshape((1, -1)).flatten(), y_hat.reshape((1, -1)).flatten()):
            # print(y_i, y_hat_i)
            self.assertAlmostEqual(y_i, y_hat_i, delta=0.1)
        # print()

    def test_with_ordinal(self):
        cs = smac.configspace.ConfigurationSpace()
        a = cs.add_hyperparameter(CategoricalHyperparameter('a', [0, 1],
                                                            default=0))
        b = cs.add_hyperparameter(OrdinalHyperparameter('b', [0, 1],
                                                        default=1))
        b = cs.add_hyperparameter(UniformFloatHyperparameter('c', lower=0., upper=1.,
                                                             default=1))
        b = cs.add_hyperparameter(UniformIntegerHyperparameter('d', lower=0, upper=10,
                                                               default=1))
        cs.seed(1)

        feat_array = np.array([0,0,0]).reshape(1, -1)
        types, bounds = get_types(cs, feat_array)
        model = RandomForestWithInstances(types=types, bounds=bounds,
                                          instance_features=feat_array,
                                          seed=1, ratio_features=1.0,
                                          pca_components=9)
        self.assertEqual(bounds[0][0], 2)
        self.assertTrue(bounds[0][1] is np.nan)
        self.assertEqual(bounds[1][0], 0)
        self.assertEqual(bounds[1][1], 1)
        self.assertEqual(bounds[2][0], 0.)
        self.assertEqual(bounds[2][1], 1.)
        self.assertEqual(bounds[3][0], 0.)
        self.assertEqual(bounds[3][1], 1.)
        X = np.array([
            [0., 0., 0., 0., 0, 0, 0],
            [0., 0., 1., 0., 0, 0, 0],
            [0., 1., 0., 9., 0, 0, 0],
            [0., 1., 1., 4., 0, 0, 0]], dtype=np.float64)
        y = np.array([0, 1, 2, 3], dtype=np.float64)

        model.train(np.vstack((X, X, X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y, y, y)))
        mean, _ = model.predict(X)
        for idx, m in enumerate(mean):
            self.assertAlmostEqual(y[idx], m, 0.05)
