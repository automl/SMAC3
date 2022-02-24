import unittest

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from ConfigSpace import ConfigurationSpace, EqualsCondition
import numpy as np

from smac.epm.util_funcs import get_types

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestUtilFuncs(unittest.TestCase):
    def test_get_types(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter('a', ['a', 'b']))
        cs.add_hyperparameter(UniformFloatHyperparameter('b', 1, 5))
        cs.add_hyperparameter(UniformIntegerHyperparameter('c', 3, 7))
        cs.add_hyperparameter(Constant('d', -5))
        cs.add_hyperparameter(OrdinalHyperparameter('e', ['cold', 'hot']))
        cs.add_hyperparameter(CategoricalHyperparameter('f', ['x', 'y']))
        types, bounds = get_types(cs, None)
        np.testing.assert_array_equal(types, [2, 0, 0, 0, 0, 2])
        self.assertEqual(bounds[0][0], 2)
        self.assertFalse(np.isfinite(bounds[0][1]))
        np.testing.assert_array_equal(bounds[1], [0, 1])
        np.testing.assert_array_equal(bounds[2], [0, 1])
        self.assertEqual(bounds[3][0], 0)
        self.assertFalse(np.isfinite(bounds[3][1]))
        np.testing.assert_array_equal(bounds[4], [0, 1])
        self.assertEqual(bounds[5][0], 2)
        self.assertFalse(np.isfinite(bounds[5][1]))

    def test_get_types_with_inactive(self):
        cs = ConfigurationSpace()
        a = cs.add_hyperparameter(CategoricalHyperparameter('a', ['a', 'b']))
        b = cs.add_hyperparameter(UniformFloatHyperparameter('b', 1, 5))
        c = cs.add_hyperparameter(UniformIntegerHyperparameter('c', 3, 7))
        d = cs.add_hyperparameter(Constant('d', -5))
        e = cs.add_hyperparameter(OrdinalHyperparameter('e', ['cold', 'hot']))
        f = cs.add_hyperparameter(CategoricalHyperparameter('f', ['x', 'y']))
        cs.add_condition(EqualsCondition(b, a, 'a'))
        cs.add_condition(EqualsCondition(c, a, 'a'))
        cs.add_condition(EqualsCondition(d, a, 'a'))
        cs.add_condition(EqualsCondition(e, a, 'a'))
        cs.add_condition(EqualsCondition(f, a, 'a'))
        types, bounds = get_types(cs, None)
        np.testing.assert_array_equal(types, [2, 0, 0, 2, 0, 3])
        self.assertEqual(bounds[0][0], 2)
        self.assertFalse(np.isfinite(bounds[0][1]))
        np.testing.assert_array_equal(bounds[1], [-1, 1])
        np.testing.assert_array_equal(bounds[2], [-1, 1])
        self.assertEqual(bounds[3][0], 2)
        self.assertFalse(np.isfinite(bounds[3][1]))
        np.testing.assert_array_equal(bounds[4], [0, 2])
        self.assertEqual(bounds[5][0], 3)
        self.assertFalse(np.isfinite(bounds[5][1]))
