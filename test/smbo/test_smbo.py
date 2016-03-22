'''
Created on Dec 15, 2015

@author: Aaron Klein
'''
import unittest
import numpy as np

from smac.smbo.smbo import SMBO
from smac.scenario.scenario import Scenario
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.task.branin import Branin
from smac.utils import test_helpers

class TestSMBO(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario({'cs': test_helpers.get_branin_config_space()})

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

    def test_rng(self):
        smbo = SMBO(self.scenario, None)
        self.assertIsInstance(smbo.rng, np.random.RandomState)
        smbo = SMBO(self.scenario, 1)
        rng = np.random.RandomState(1)
        self.assertIsInstance(smbo.rng, np.random.RandomState)
        smbo = SMBO(self.scenario, rng)
        self.assertIs(smbo.rng, rng)
        self.assertRaisesRegex(TypeError, "Unknown type <class 'str'> for argument "
                                          'rng. Only accepts None, int or '
                                          'np.random.RandomState',
                               SMBO, self.scenario, 'BLA')

if __name__ == "__main__":
    unittest.main()
