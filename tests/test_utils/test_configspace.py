import unittest

import numpy as np
from ConfigSpace import ConfigurationSpace, EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.utils.configspace import get_types

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


def test_get_types():
    cs = ConfigurationSpace(seed=0)
    cs.add_hyperparameter(CategoricalHyperparameter("a", ["a", "b"]))
    cs.add_hyperparameter(UniformFloatHyperparameter("b", 1, 5))
    cs.add_hyperparameter(UniformIntegerHyperparameter("c", 3, 7))
    cs.add_hyperparameter(Constant("d", -5))
    cs.add_hyperparameter(OrdinalHyperparameter("e", ["cold", "hot"]))
    cs.add_hyperparameter(CategoricalHyperparameter("f", ["x", "y"]))
    types, bounds = get_types(cs, None)

    assert types == [2, 0, 0, 0, 0, 2]
    assert bounds[0][0] == 2
    assert not np.isfinite(bounds[0][1])
    assert bounds[1] == (0, 1)
    assert bounds[2] == (0, 1)
    assert bounds[3][0] == 0
    assert not np.isfinite(bounds[3][1])
    assert bounds[4] == (0, 1)
    assert bounds[5][0] == 2
    assert not np.isfinite(bounds[5][1])


def test_get_types_with_inactive():
    cs = ConfigurationSpace()
    a = cs.add_hyperparameter(CategoricalHyperparameter("a", ["a", "b"]))
    b = cs.add_hyperparameter(UniformFloatHyperparameter("b", 1, 5))
    c = cs.add_hyperparameter(UniformIntegerHyperparameter("c", 3, 7))
    d = cs.add_hyperparameter(Constant("d", -5))
    e = cs.add_hyperparameter(OrdinalHyperparameter("e", ["cold", "hot"]))
    f = cs.add_hyperparameter(CategoricalHyperparameter("f", ["x", "y"]))
    cs.add_condition(EqualsCondition(b, a, "a"))
    cs.add_condition(EqualsCondition(c, a, "a"))
    cs.add_condition(EqualsCondition(d, a, "a"))
    cs.add_condition(EqualsCondition(e, a, "a"))
    cs.add_condition(EqualsCondition(f, a, "a"))
    types, bounds = get_types(cs, None)

    assert types == [2, 0, 0, 2, 0, 3]
    assert bounds[0][0] == 2
    assert not np.isfinite(bounds[0][1])
    assert bounds[1] == (-1, 1)
    assert bounds[2] == (-1, 1)
    assert bounds[3][0] == 2
    assert not np.isfinite(bounds[3][1])
    assert bounds[4] == (0, 2)
    assert bounds[5][0] == 3
    assert not np.isfinite(bounds[5][1])


def test_get_types_with_instance_feautres():
    instance_features = {"i1": [4, 6, 10], "i2": [-5, 100, -10]}

    cs = ConfigurationSpace(seed=0)
    cs.add_hyperparameter(CategoricalHyperparameter("a", ["a", "b"]))
    cs.add_hyperparameter(UniformFloatHyperparameter("b", 1, 5))
    cs.add_hyperparameter(UniformIntegerHyperparameter("c", 3, 7))
    cs.add_hyperparameter(Constant("d", -5))
    cs.add_hyperparameter(OrdinalHyperparameter("e", ["cold", "hot"]))
    cs.add_hyperparameter(CategoricalHyperparameter("f", ["x", "y"]))

    types, bounds = get_types(cs, instance_features)

    assert types == [2, 0, 0, 0, 0, 2, 0, 0, 0]
    assert bounds[0][0] == 2
    assert not np.isfinite(bounds[0][1])
    assert bounds[1] == (0, 1)
    assert bounds[2] == (0, 1)
    assert bounds[3][0] == 0
    assert not np.isfinite(bounds[3][1])
    assert bounds[4] == (0, 1)
    assert bounds[5][0] == 2
    assert not np.isfinite(bounds[5][1])

    # Test instance features here
    # The bounds of the features are not included
    assert len(bounds) == 6
