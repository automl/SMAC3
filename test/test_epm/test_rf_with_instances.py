import unittest
import unittest.mock

import numpy as np

from ConfigSpace import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    OrdinalHyperparameter, EqualsCondition

from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.util_funcs import get_types
import smac
import smac.configspace

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestRFWithInstances(unittest.TestCase):

    def _get_cs(self, n_dimensions):
        configspace = smac.configspace.ConfigurationSpace()
        for i in range(n_dimensions):
            configspace.add_hyperparameter(UniformFloatHyperparameter('x%d' % i, 0, 1))
        return configspace

    def test_predict_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)

        model = RandomForestWithInstances(
            configspace=self._get_cs(10),
            types=np.zeros((10,), dtype=np.uint),
            bounds=list(map(lambda x: (0, 10), range(10))),
            seed=1,
        )
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

    def test_predict(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)
        model = RandomForestWithInstances(
            configspace=self._get_cs(10),
            types=np.zeros((10,), dtype=np.uint),
            bounds=list(map(lambda x: (0, 10), range(10))),
            seed=1,
        )
        model.train(X[:10], Y[:10])
        m_hat, v_hat = model.predict(X[10:])
        self.assertEqual(m_hat.shape, (10, 1))
        self.assertEqual(v_hat.shape, (10, 1))

    def test_train_with_pca(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 20)
        F = rs.rand(10, 10)
        Y = rs.rand(20, 1)
        model = RandomForestWithInstances(
            configspace=self._get_cs(10),
            types=np.zeros((20,), dtype=np.uint),
            bounds=list(map(lambda x: (0, 10), range(10))),
            seed=1,
            pca_components=2,
            instance_features=F,
        )
        model.train(X, Y)

        self.assertEqual(model.n_params, 10)
        self.assertEqual(model.n_feats, 10)
        self.assertIsNotNone(model.pca)
        self.assertIsNotNone(model.scaler)

    def test_predict_marginalized_over_instances_wrong_X_dimensions(self):
        rs = np.random.RandomState(1)

        model = RandomForestWithInstances(
            configspace=self._get_cs(10),
            types=np.zeros((10,), dtype=np.uint),
            instance_features=rs.rand(10, 2),
            seed=1,
            bounds=list(map(lambda x: (0, 10), range(10))),
        )
        X = rs.rand(10)
        self.assertRaisesRegex(ValueError, "Expected 2d array, got 1d array!",
                               model.predict_marginalized_over_instances, X)
        X = rs.rand(10, 10, 10)
        self.assertRaisesRegex(ValueError, "Expected 2d array, got 3d array!",
                               model.predict_marginalized_over_instances, X)

    @unittest.mock.patch.object(RandomForestWithInstances, 'predict')
    def test_predict_marginalized_over_instances_no_features(self, rf_mock):
        """The RF should fall back to the regular predict() method."""

        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 1)
        model = RandomForestWithInstances(
            configspace=self._get_cs(10),
            types=np.zeros((10,), dtype=np.uint),
            bounds=list(map(lambda x: (0, 10), range(10))),
            seed=1,
        )
        model.train(X[:10], Y[:10])
        model.predict(X[10:])
        self.assertEqual(rf_mock.call_count, 1)

    def test_predict_marginalized_over_instances(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        F = rs.rand(10, 5)
        Y = rs.rand(len(X) * len(F), 1)
        X_ = rs.rand(200, 15)

        model = RandomForestWithInstances(
            configspace=self._get_cs(10),
            types=np.zeros((15,), dtype=np.uint),
            instance_features=F,
            bounds=list(map(lambda x: (0, 10), range(10))),
            seed=1,
        )
        model.train(X_, Y)
        means, vars = model.predict_marginalized_over_instances(X)
        self.assertEqual(means.shape, (20, 1))
        self.assertEqual(vars.shape, (20, 1))

    @unittest.mock.patch.object(RandomForestWithInstances, 'predict')
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

        model = RandomForestWithInstances(
            configspace=self._get_cs(10),
            types=np.zeros((15,), dtype=np.uint),
            instance_features=F,
            bounds=list(map(lambda x: (0, 10), range(10))),
            seed=1,
        )
        X = rs.rand(20, 10)
        F = rs.rand(10, 5)
        Y = rs.randint(1, size=(len(X) * len(F), 1)) * 1.
        X_ = rs.rand(200, 15)
        model.train(X_, Y)
        means, vars = model.predict_marginalized_over_instances(rs.rand(11, 10))
        # expected to be 0 as the predict is replaced by manual unloggin the trees
        self.assertEqual(rf_mock.call_count, 0)
        self.assertEqual(means.shape, (11, 1))
        self.assertEqual(vars.shape, (11, 1))
        for i in range(11):
            self.assertEqual(means[i], 0.)
            self.assertEqual(vars[i], 1.e-10)

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
        model = RandomForestWithInstances(
            configspace=self._get_cs(3),
            types=np.array([0, 0, 0], dtype=np.uint),
            bounds=[(0, np.nan), (0, np.nan), (0, np.nan)],
            instance_features=None,
            seed=12345,
            ratio_features=1.0,
        )
        model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

        y_hat, _ = model.predict(X)
        for y_i, y_hat_i in zip(y.reshape((1, -1)).flatten(), y_hat.reshape((1, -1)).flatten()):
            self.assertAlmostEqual(y_i, y_hat_i, delta=0.1)

    def test_with_ordinal(self):
        cs = smac.configspace.ConfigurationSpace()
        _ = cs.add_hyperparameter(CategoricalHyperparameter('a', [0, 1], default_value=0))
        _ = cs.add_hyperparameter(OrdinalHyperparameter('b', [0, 1], default_value=1))
        _ = cs.add_hyperparameter(UniformFloatHyperparameter('c', lower=0., upper=1., default_value=1))
        _ = cs.add_hyperparameter(UniformIntegerHyperparameter('d', lower=0, upper=10, default_value=1))
        cs.seed(1)

        feat_array = np.array([0, 0, 0]).reshape(1, -1)
        types, bounds = get_types(cs, feat_array)
        model = RandomForestWithInstances(
            configspace=cs,
            types=types,
            bounds=bounds,
            instance_features=feat_array,
            seed=1,
            ratio_features=1.0,
            pca_components=9,
        )
        self.assertEqual(bounds[0][0], 2)
        self.assertTrue(bounds[0][1] is np.nan)
        self.assertEqual(bounds[1][0], 0)
        self.assertEqual(bounds[1][1], 1)
        self.assertEqual(bounds[2][0], 0.)
        self.assertEqual(bounds[2][1], 1.)
        self.assertEqual(bounds[3][0], 0.)
        self.assertEqual(bounds[3][1], 1.)
        X = np.array([
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 9., 0., 0., 0.],
            [0., 1., 1., 4., 0., 0., 0.]], dtype=np.float64)
        y = np.array([0, 1, 2, 3], dtype=np.float64)

        X_train = np.vstack((X, X, X, X, X, X, X, X, X, X))
        y_train = np.vstack((y, y, y, y, y, y, y, y, y, y))

        model.train(X_train, y_train.reshape((-1, 1)))
        mean, _ = model.predict(X)
        for idx, m in enumerate(mean):
            self.assertAlmostEqual(y[idx], m, 0.05)

    def test_rf_on_sklearn_data(self):
        import sklearn.datasets
        X, y = sklearn.datasets.load_boston(return_X_y=True)
        rs = np.random.RandomState(1)

        types = np.zeros(X.shape[1])
        bounds = [(np.min(X[:, i]), np.max(X[:, i])) for i in range(X.shape[1])]

        cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

        for do_log in [False, True]:
            if do_log:
                targets = np.log(y)
                model = RandomForestWithInstances(
                    configspace=self._get_cs(X.shape[1]),
                    types=types,
                    bounds=bounds,
                    seed=1,
                    ratio_features=1.0,
                    pca_components=100,
                    log_y=True,
                )
                maes = [0.43169704431695493156, 0.4267519520332511912]
            else:
                targets = y
                model = RandomForestWithInstances(
                    configspace=self._get_cs(X.shape[1]),
                    types=types,
                    bounds=bounds,
                    seed=1,
                    ratio_features=1.0,
                    pca_components=100,
                )
                maes = [9.3298376833224042496, 9.348010654109179346]

            for i, (train_split, test_split) in enumerate(cv.split(X, targets)):
                X_train = X[train_split]
                y_train = targets[train_split]
                X_test = X[test_split]
                y_test = targets[test_split]
                model.train(X_train, y_train)
                y_hat, mu_hat = model.predict(X_test)
                mae = np.mean(np.abs(y_hat - y_test), dtype=np.float128)
                self.assertAlmostEqual(mae, maes[i], msg=('Do log: %s, iteration %i' % (str(do_log), i)),
                                       # We observe a difference of around 0.00017
                                       # in github actions if doing log
                                       places=3 if do_log else 7)

    def test_impute_inactive_hyperparameters(self):
        cs = smac.configspace.ConfigurationSpace()
        a = cs.add_hyperparameter(CategoricalHyperparameter('a', [0, 1]))
        b = cs.add_hyperparameter(CategoricalHyperparameter('b', [0, 1]))
        c = cs.add_hyperparameter(UniformFloatHyperparameter('c', 0, 1))
        cs.add_condition(EqualsCondition(b, a, 1))
        cs.add_condition(EqualsCondition(c, a, 0))
        cs.seed(1)

        configs = cs.sample_configuration(size=100)
        config_array = smac.configspace.convert_configurations_to_array(configs)
        for line in config_array:
            if line[0] == 0:
                self.assertTrue(np.isnan(line[1]))
            elif line[0] == 1:
                self.assertTrue(np.isnan(line[2]))

        model = RandomForestWithInstances(
            configspace=cs,
            types=np.zeros((3,), dtype=np.uint),
            bounds=list(map(lambda x: (0, 1), range(10))),
            seed=1,
        )
        config_array = model._impute_inactive(config_array)
        for line in config_array:
            if line[0] == 0:
                self.assertEqual(line[1], 2)
            elif line[0] == 1:
                self.assertEqual(line[2], -1)
