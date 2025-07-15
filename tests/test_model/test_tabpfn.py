import numpy as np
import sys
import pytest

from ConfigSpace import (
    CategoricalHyperparameter,
    ConfigurationSpace,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

try:
    from smac.model.tabPFNv2 import TabPFNModel
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="tabpfn requires Python >=3.9"
)

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


def _get_cs(n_dimensions):
    configspace = ConfigurationSpace(seed=0)
    for i in range(n_dimensions):
        configspace.add(UniformFloatHyperparameter("x%d" % i, 0, 1))

    return configspace


def test_predict_wrong_X_dimensions():
    rs = np.random.RandomState(1)

    model = TabPFNModel(
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
    model = TabPFNModel(configspace=_get_cs(10))
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

    model = TabPFNModel(
        configspace=_get_cs(10),
        pca_components=2,
        instance_features=F,
    )
    model.train(X, Y)

    assert model._n_features == 10
    assert model._n_hps == 10
    assert model._pca is not None
    assert model._scaler is not None


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
    model = TabPFNModel(
        configspace=_get_cs(3),
        instance_features=None,
        seed=12345,
    )
    model.train(np.vstack((X, X, X, X, X, X, X, X)), np.vstack((y, y, y, y, y, y, y, y)))

    y_hat, _ = model.predict(X)
    for y_i, y_hat_i in zip(y.reshape((1, -1)).flatten(), y_hat.reshape((1, -1)).flatten()):
        assert pytest.approx(y_i, rel=0.5) == y_hat_i


def test_with_ordinal():
    cs = ConfigurationSpace(seed=0)
    cs.add(CategoricalHyperparameter("a", [0, 1], default_value=0))
    cs.add(OrdinalHyperparameter("b", [0, 1], default_value=1))
    cs.add(UniformFloatHyperparameter("c", lower=0.0, upper=1.0, default_value=1))
    cs.add(UniformIntegerHyperparameter("d", lower=0, upper=10, default_value=1))

    F = {}
    for i in range(1):
        F[f"instance-{i}"] = [0, 0, 0]

    model = TabPFNModel(
        configspace=cs,
        instance_features=F,
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
        assert pytest.approx(y[idx], abs=0.05) == m

def test_predict_before_train_raises():
    model = TabPFNModel(configspace=_get_cs(3))
    X = np.random.rand(2, 3)
    with pytest.raises(AssertionError):
        model.predict(X)