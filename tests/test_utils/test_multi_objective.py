import unittest
from multiprocessing.sharedctypes import Value

import numpy as np
import pytest

from smac.utils.multi_objective import normalize_costs

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class MultiObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.bounds = [(0, 50), (50, 100)]
        self.bounds_invalid = [(0, 0), (5, 5)]

    def test_normalize_costs(self):
        # If no bounds are passed, we get the same result back
        v = [5, 2]
        nv = normalize_costs(v)
        self.assertEqual(nv, [5, 2])

        # Normalize between 0..1 given data only
        v = [25, 50]
        nv = normalize_costs(v, self.bounds)
        self.assertEqual(nv, [0.5, 0])

        # Invalid bounds
        v = [25, 50]
        nv = normalize_costs(v, self.bounds_invalid)
        self.assertEqual(nv, [1, 1])

        # Invalid input
        v = [[25], [50]]
        with pytest.raises(AssertionError):
            nv = normalize_costs(v, self.bounds)

        # Wrong shape
        v = [25, 50, 75]
        with pytest.raises(ValueError):
            nv = normalize_costs(v, self.bounds)


if __name__ == "__main__":
    t = MultiObjectiveTest()
    t.setUp()
    t.test_normalize_costs()
