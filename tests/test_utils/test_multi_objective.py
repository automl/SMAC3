import unittest

import numpy as np

from smac.multi_objective.utils import normalize_costs

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class MultiObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.bounds_1d = [(0, 1)]
        self.bounds_2d = [(0, 1), (50, 100)]

    def test_normalize_costs(self):
        # Normalize between 0..1 given data only
        v = np.array([[5, 2], [10, 0]])
        nv = normalize_costs(v)
        self.assertEqual(list(nv.flatten()), list(np.array([[0, 1], [1, 0]]).flatten()))

        # Normalize between 0..1 given data only
        v = np.array([[5, 75], [0.5, 50], [0.75, 60], [0, 100]])
        nv = normalize_costs(v, self.bounds_2d)

        self.assertEqual(
            list(nv.flatten()),
            list(np.array([[5, 0.5], [0.5, 0], [0.75, 0.2], [0, 1]]).flatten()),
        )

        # No normalization
        v = np.array([[5, 2]])
        nv = normalize_costs(v)
        self.assertEqual(list(nv.flatten()), list(np.array([[1.0, 1.0]]).flatten()))

        # Normalization with given bounds
        v = np.array([[500, 150]])
        nv = normalize_costs(v, self.bounds_2d)
        self.assertEqual(list(nv.flatten()), list(np.array([[500, 2.0]]).flatten()))

        # Test one-dimensional list
        v = [500, 150]
        nv = normalize_costs(v, self.bounds_1d)
        self.assertEqual(list(nv.flatten()), list(np.array([[500], [150]]).flatten()))

        # Test one-dimensional array without bounds
        v = np.array([500, 150])
        nv = normalize_costs(v)
        self.assertEqual(list(nv.flatten()), list(np.array([[1.0], [0.0]]).flatten()))

        # Test one-dimensional array without bounds
        v = np.array([1000, 200, 400, 800, 600, 0])
        nv = normalize_costs(v)
        self.assertEqual(
            list(nv.flatten()),
            list(np.array([[1], [0.2], [0.4], [0.8], [0.6], [0.0]]).flatten()),
        )

        # Test one-dimensional array with one objective
        v = np.array([500])
        nv = normalize_costs(v, self.bounds_1d)
        self.assertEqual(list(nv.flatten()), list(np.array([[500.0]]).flatten()))

        # Test one-dimensional list with one objective
        v = [500]
        nv = normalize_costs(v, self.bounds_1d)
        self.assertEqual(list(nv.flatten()), list(np.array([[500.0]]).flatten()))


if __name__ == "__main__":
    t = MultiObjectiveTest()
    t.setUp()
    t.test_normalize_costs()
