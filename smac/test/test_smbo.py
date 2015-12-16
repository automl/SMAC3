'''
Created on Dec 15, 2015

@author: Aaron Klein
'''
import unittest
import numpy as np

from smac.smbo.smbo import SMBO
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.task.branin import Branin


class TestSMBO(unittest.TestCase):

    def test_choose_next(self):
        pcs_file = "test_files/params_branin.pcs"
        seed = 42
        instance_features = np.array([[1]])
        smbo = SMBO(pcs_file, instance_features, seed)
        task = Branin()

        X = init_random_uniform(task.X_lower, task.X_upper, 10)
        Y = task.evaluate(X)

        x = smbo.choose_next(X, Y)

        assert len(x.shape) == 2
        assert x.shape[0] == 1
        assert x.shape[1] == X.shape[1]


if __name__ == "__main__":
    unittest.main()
