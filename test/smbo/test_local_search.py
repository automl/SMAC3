'''
Created on Dec 15, 2015

@author: Aaron Klein
'''
import unittest
import os

import numpy as np
from scipy.spatial.distance import euclidean

from smac.configspace import pcs

from smac.smbo.local_search import LocalSearch
from robo.task.branin import Branin


class TestLocalSearch(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.test_files_dir = os.path.join(current_dir, '..', 'test_files')

    def test_local_search(self):
        pcs_file = os.path.join(self.test_files_dir, "params_branin.pcs")
        seed = np.random.randint(1, 100000)
        task = Branin()

        with open(pcs_file) as fh:
            config_space = pcs.read(fh.readlines())
            config_space.seed(seed)

        def acquisition_function(point):
            dist = [euclidean(point, opt) for opt in task.opt]
            return -np.min(dist)

        l = LocalSearch(acquisition_function, config_space, epsilon=1e-10,
                        max_iterations=100000)

        start_point = config_space.sample_configuration()
        _, acq_val_incumbent = l.maximize(start_point)
        self.assertLessEqual(-acq_val_incumbent, 0.05)

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
