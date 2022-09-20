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

from smac.model.utils import check_subspace_points, get_types

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class TestUtilFuncs(unittest.TestCase):
    def test_get_types(self):
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CategoricalHyperparameter("a", ["a", "b"]))
        cs.add_hyperparameter(UniformFloatHyperparameter("b", 1, 5))
        cs.add_hyperparameter(UniformIntegerHyperparameter("c", 3, 7))
        cs.add_hyperparameter(Constant("d", -5))
        cs.add_hyperparameter(OrdinalHyperparameter("e", ["cold", "hot"]))
        cs.add_hyperparameter(CategoricalHyperparameter("f", ["x", "y"]))
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

    def test_check_subspace_points(self):
        # 1D array
        np.testing.assert_equal([True], check_subspace_points(np.array([0.5, 0.5])))
        bounds_cont_base = np.array([0.0, 1.0])
        X_cont = np.array([[0.5, 0.8], [-0.2, 0.2], [-0.7, 0.3]])
        np.testing.assert_equal(check_subspace_points(X_cont), [True] * len(X_cont))
        cont_dims = np.arange(X_cont.shape[-1])

        with self.assertRaises(ValueError):
            # bounds_cont missing
            check_subspace_points(X_cont, cont_dims=cont_dims)

        with self.assertRaises(ValueError):
            # bounds_cont does not match
            check_subspace_points(X_cont, cont_dims=cont_dims, bounds_cont=bounds_cont_base)

        bounds_cont = np.tile(bounds_cont_base, [X_cont.shape[-1], 1])

        np.testing.assert_equal(
            check_subspace_points(X_cont, cont_dims=cont_dims, bounds_cont=bounds_cont), [True, False, False]
        )
        np.testing.assert_equal(
            check_subspace_points(X_cont, cont_dims=cont_dims, bounds_cont=bounds_cont, expand_bound=True),
            [True, True, False],
        )

        # categorical hps
        X_cat = np.array([[0, 1], [2, 1], [1, 4]])
        cat_dims = np.arange(X_cat.shape[-1])

        bounds_cat = [(0, 2), (1, 4)]

        with self.assertRaises(ValueError):
            # bounds_cont missing
            check_subspace_points(X_cat, cat_dims=cat_dims)

        with self.assertRaises(ValueError):
            # bounds_cat doe not match
            check_subspace_points(X_cat, cat_dims=cat_dims, bounds_cat=[(0, 1)])

        np.testing.assert_equal(
            check_subspace_points(X_cat, cat_dims=cat_dims, bounds_cat=bounds_cat), [True, True, False]
        )

        # cat + cont
        X_mix = np.hstack([X_cont, X_cat])
        cat_dims += len(cont_dims)
        ss_mix = check_subspace_points(
            X_mix, cont_dims=cont_dims, cat_dims=cat_dims, bounds_cont=bounds_cont, bounds_cat=bounds_cat
        )
        np.testing.assert_equal(ss_mix, [True, False, False])
