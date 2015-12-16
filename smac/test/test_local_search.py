'''
Created on Dec 15, 2015

@author: Aaron Klein
'''
import unittest
import numpy as np
from scipy.spatial.distance import euclidean

from smac.configspace import pcs

from smac.smbo.local_search import LocalSearch
from robo.task.branin import Branin


class TestLocalSearch(unittest.TestCase):

    def test_local_search(self):
        pcs_file = "/home/kleinaa/devel/git/smac3/test_files/params_branin.pcs"
        seed = 42
        task = Branin()

        with open(pcs_file) as fh:
            config_space = pcs.read(fh.readlines())
            config_space.seed(seed)

        def acquisition_function(point):
            dist = [euclidean(point, opt) for opt in task.opt]
            return np.min(dist)

        l = LocalSearch(acquisition_function, config_space, epsilon=0.01,
                        n_neighbours=50, max_iterations=100000)

        start_point = config_space.sample_configuration()
        incumbet, acq_val_incumbent = l.maximize(start_point)

if __name__ == "__main__":
    unittest.main()
