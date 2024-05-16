import numpy as np
import pytest
from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    EqualsCondition,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.model.random_forest.random_forest import RandomForest
from smac.utils.configspace import convert_configurations_to_array

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def _get_cs(n_dimensions):
    configspace = ConfigurationSpace(seed=0)
    for i in range(n_dimensions):
        configspace.add_hyperparameter(UniformFloatHyperparameter("x%d" % i, 0, 1))

    return configspace


def test_predict_wrong_X_dimensions():
    rs = np.random.RandomState(1)

    model = RandomForest(
        configspace=_get_cs(10),
    )
    X = rs.rand(10)
    with pytest.raises(ValueError, match="Expected 2d array.*"):
        model.predict(X)

    X = rs.rand(10, 10, 10)
    with pytest.raises(ValueError, match="Expected 2d array.*"):
        model.predict(X)

    X = rs.rand(10, 5)

    with pytest.raises(ValueError, match="Feature mismatch: .*"):
        model.predict(X)


def test_predict():
    rs = np.random.RandomState(1)
    X = rs.rand(20, 10)
    Y = rs.rand(10, 1)
    model = RandomForest(configspace=_get_cs(10))
    model.train(X[:10], Y[:10])
    m_hat, v_hat = model.predict(X[10:])
    assert m_hat.shape == (10, 1)
    assert v_hat.shape == (10, 1)


def test_train_with_pca():
    rs = np.random.RandomState(1)
    X = rs.rand(20, 20)
    Y = rs.rand(20, 1)

    F = {}
    for i in range(10):
        F[f"instance-{i}"] = list(rs.rand(10))

    model = RandomForest(
        configspace=_get_cs(10),
        pca_components=2,
        instance_features=F,
    )
    model.train(X, Y)

    assert model._n_features == 10
    assert model._n_hps == 10
    assert model._pca is not None
    assert model._scaler is not None


def test_predict_marginalized_over_instances_wrong_X_dimensions():
    rs = np.random.RandomState(1)
    F = {}
    for i in range(10):
        F[f"instance-{i}"] = list(rs.rand(2))

    model = RandomForest(
        configspace=_get_cs(10),
        instance_features=F,
    )
    X = rs.rand(10)
    with pytest.raises(ValueError, match="Expected 2d array.*"):
        model.predict_marginalized(X)

    X = rs.rand(10, 10, 10)
    with pytest.raises(ValueError, match="Expected 2d array.*"):
        model.predict_marginalized(X)


def test_predict_marginalized_no_features():
    """The RF should fall back to the regular predict() method."""

    rs = np.random.RandomState(1)
    X = rs.rand(20, 10)
    Y = rs.rand(10, 1)
    model = RandomForest(configspace=_get_cs(10))
    model.train(X[:10], Y[:10])
    model.predict(X[10:])


def test_predict_marginalized():
    rs = np.random.RandomState(1)
    X = rs.rand(20, 10)
    F = {}
    for i in range(10):
        F[f"instance-{i}"] = list(rs.rand(5))
    Y = rs.rand(len(X) * len(F), 1)
    X_ = rs.rand(200, 15)

    model = RandomForest(
        configspace=_get_cs(10),
        instance_features=F,
    )
    model.train(X_, Y)
    means, variances = model.predict_marginalized(X)
    assert means.shape == (20, 1)
    assert variances.shape == (20, 1)


def test_predict_marginalized_mocked():

    rs = np.random.RandomState(1)
    F = {}
    for i in range(10):
        F[f"instance-{i}"] = list(rs.rand(5))

    model = RandomForest(
        configspace=_get_cs(10),
        instance_features=F,
    )
    X = rs.rand(20, 10)
    Y = rs.randint(1, size=(len(X) * len(F), 1)) * 1.0
    X_ = rs.rand(200, 15)
    model.train(X_, Y)
    means, variances = model.predict_marginalized(rs.rand(11, 10))
    assert means.shape == (11, 1)
    assert variances.shape == (11, 1)
    for i in range(11):
        assert means[i] == 0.0
        assert variances[i] == 1.0e-10


def test_predict_with_actual_values():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([[0.1], [0.2], [9], [9.2], [100.0], [100.2], [109.0], [109.2]], dtype=np.float64)
    model = RandomForest(
        configspace=_get_cs(3),
        instance_features=None,
        seed=12345,
        ratio_features=1.0,
    )
    model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

    y_hat, _ = model.predict(X)
    for y_i, y_hat_i in zip(y.reshape((1, -1)).flatten(), y_hat.reshape((1, -1)).flatten()):
        assert pytest.approx(y_i, 0.1) == y_hat_i


def test_with_ordinal():
    cs = ConfigurationSpace(seed=0)
    _ = cs.add_hyperparameter(CategoricalHyperparameter("a", [0, 1], default_value=0))
    _ = cs.add_hyperparameter(OrdinalHyperparameter("b", [0, 1], default_value=1))
    _ = cs.add_hyperparameter(UniformFloatHyperparameter("c", lower=0.0, upper=1.0, default_value=1))
    _ = cs.add_hyperparameter(UniformIntegerHyperparameter("d", lower=0, upper=10, default_value=1))

    F = {}
    for i in range(1):
        F[f"instance-{i}"] = [0, 0, 0]

    model = RandomForest(
        configspace=cs,
        instance_features=F,
        ratio_features=1.0,
        pca_components=9,
    )

    X = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 9.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 4.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 1, 2, 3], dtype=np.float64)

    X_train = np.vstack((X, X, X, X, X, X, X, X, X, X))
    y_train = np.vstack((y, y, y, y, y, y, y, y, y, y))

    model.train(X_train, y_train.reshape((-1, 1)))
    mean, _ = model.predict(X)
    for idx, m in enumerate(mean):
        assert pytest.approx(y[idx], 0.05) == m


# def test_rf_on_sklearn_data():
#     import sklearn.datasets

#     X, y = sklearn.datasets.load_boston(return_X_y=True)
#     rs = np.random.RandomState(1)

#     cv = sklearn.model_selection.KFold(shuffle=True, random_state=rs, n_splits=2)

#     for do_log in [False, True]:
#         if do_log:
#             targets = np.log(y)
#             model = RandomForest(
#                 configspace=_get_cs(X.shape[1]),
#                 ratio_features=1.0,
#                 pca_components=100,
#                 log_y=True,
#             )
#             maes = [0.43169704431695493156, 0.4267519520332511912]
#         else:
#             targets = y
#             model = RandomForest(
#                 configspace=_get_cs(X.shape[1]),
#                 seed=1,
#                 ratio_features=1.0,
#                 pca_components=100,
#             )
#             maes = [9.3298376833224042496, 9.348010654109179346]

#         for i, (train_split, test_split) in enumerate(cv.split(X, targets)):
#             X_train = X[train_split]
#             y_train = targets[train_split]
#             X_test = X[test_split]
#             y_test = targets[test_split]
#             model.train(X_train, y_train)
#             y_hat, mu_hat = model.predict(X_test)
#             mae = np.mean(np.abs(y_hat - y_test), dtype=float)

#             assert pytest.approx(mae, 0.1) == maes[i]


def test_impute_inactive_hyperparameters():
    cs = ConfigurationSpace(seed=0)
    a = cs.add_hyperparameter(CategoricalHyperparameter("a", [0, 1, 2]))
    b = cs.add_hyperparameter(CategoricalHyperparameter("b", [0, 1]))
    c = cs.add_hyperparameter(UniformFloatHyperparameter("c", 0, 1))
    d = cs.add_hyperparameter(OrdinalHyperparameter("d", [0, 1, 2]))
    cs.add_condition(EqualsCondition(b, a, 1))
    cs.add_condition(EqualsCondition(c, a, 0))
    cs.add_condition(EqualsCondition(d, a, 2))

    configs = cs.sample_configuration(size=100)
    config_array = convert_configurations_to_array(configs)
    for line in config_array:
        if line[0] == 0:
            assert np.isnan(line[1])
            assert np.isnan(line[3])
        elif line[0] == 1:
            assert np.isnan(line[2])
            assert np.isnan(line[3])
        elif line[0] == 2:
            assert np.isnan(line[1])
            assert np.isnan(line[2])

    model = RandomForest(configspace=cs)
    config_array = model._impute_inactive(config_array)
    for line in config_array:
        if line[0] == 0:
            assert line[1] == 2
            assert line[3] == 3
        elif line[0] == 1:
            assert line[2] == -1
            assert line[3] == 3
        elif line[0] == 2:
            assert line[1] == 2
            assert line[2] == -1
