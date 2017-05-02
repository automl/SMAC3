'''
Created on Dec 15, 2015

@author: Aaron Klein
'''
import unittest
import os

import numpy as np
from scipy.spatial.distance import euclidean

from smac.configspace import pcs

from smac.optimizer.local_search import LocalSearch
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter


def rosenbrock_4d(cfg):
    x1 = cfg["x1"]
    x2 = cfg["x2"]
    x3 = cfg["x3"]
    x4 = cfg["x4"]

    val = (100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2 +
           100 * (x3 - x2 ** 2) ** 2 + (x2 - 1) ** 2 +
           100 * (x4 - x3 ** 2) ** 2 + (x3 - 1) ** 2)

    return(val)

class TestLocalSearch(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.test_files_dir = os.path.join(current_dir, '..', 'test_files')
        seed = np.random.randint(1, 100000)
        self.cs = ConfigurationSpace(seed=seed)
        x1 = UniformFloatHyperparameter("x1", -5, 5, default=5)
        self.cs.add_hyperparameter(x1)
        x2 = UniformIntegerHyperparameter("x2", -5, 5, default=5)
        self.cs.add_hyperparameter(x2)
        x3 = CategoricalHyperparameter(
            "x3", [5, 2, 0, 1, -1, -2, 4, -3, 3, -5, -4], default=5)
        self.cs.add_hyperparameter(x3)
        x4 = UniformIntegerHyperparameter("x4", -5, 5, default=5)
        self.cs.add_hyperparameter(x4)

    def test_local_search(self):

        def acquisition_function(point):

            opt = np.array([1, 1, 1, 1])
            dist = [euclidean(point, opt)]
            return -np.min(dist)

        l = LocalSearch(acquisition_function, self.cs, epsilon=1e-10,
                        max_iterations=100000)

        start_point = self.cs.sample_configuration()

        acq_val_start_point = acquisition_function(start_point.get_array())

        _, acq_val_incumbent = l.maximize(start_point)

        # Local search needs to find something that is as least as good as the
        # start point
        self.assertLessEqual(acq_val_start_point, acq_val_incumbent)

    def test_local_search_2(self):
        pcs_file = os.path.join(self.test_files_dir, "test_local_search.pcs")
        seed = np.random.randint(1, 100000)

        with open(pcs_file) as fh:
            config_space = pcs.read(fh.readlines())
            config_space.seed(seed)

        def acquisition_function(point):
            return np.count_nonzero(np.array(point))

        start_point = config_space.get_default_configuration()

        l = LocalSearch(acquisition_function, config_space, epsilon=0.01,
                        max_iterations=100000)
        incumbent, acq_val_incumbent = l.maximize(start_point)

        self.assertEqual(acq_val_incumbent, len(start_point.get_array()))
        self.assertTrue(np.all(incumbent.get_array() ==
                               np.ones([acq_val_incumbent])))

if __name__ == "__main__":
    unittest.main()
