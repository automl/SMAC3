import unittest
from unittest import mock

import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

from smac.model.multi_objective_model import MultiObjectiveModel
from smac.model.random_forest.random_forest import RandomForest

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def _get_cs(n_dimensions):
    configspace = ConfigurationSpace()
    for i in range(n_dimensions):
        configspace.add_hyperparameter(UniformFloatHyperparameter("x%d" % i, 0, 1))

    return configspace


def test_train_and_predict_with_rf():
    rs = np.random.RandomState(1)
    X = rs.rand(20, 10)
    Y = rs.rand(10, 2)

    model = MultiObjectiveModel(
        models=RandomForest(_get_cs(10)),
        objectives=["cost", "ln(runtime)"],
    )

    model.train(X[:10], Y)
    m, v = model.predict(X[10:])
    assert m.shape == (10, 2)
    assert v.shape == (10, 2)

    m, v = model.predict_marginalized(X[10:])
    assert m.shape == (10, 2)
    assert v.shape == (10, 2)
