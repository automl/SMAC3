'''
Created on Dec 15, 2015

@author: Aaron Klein
'''
import unittest
import numpy as np

from smac.smbo.smbo import SMBO
from smac.scenario.scenario import Scenario

from smac.utils import test_helpers


class TestSMBO(unittest.TestCase):

    def setUp(self):
        self.scenario = Scenario({'cs': test_helpers.get_branin_config_space()})
        
    def branin(self, x):
        y = (x[:, 1] - (5.1 / (4 * np.pi ** 2)) * x[:, 0] ** 2 + 5 * x[:, 0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10

        return y[:, np.newaxis]        

    def test_choose_next(self):
        seed = 42
        smbo = SMBO(self.scenario, seed)
        X = self.scenario.cs.sample_configuration().get_array()[None, :]

        Y = self.branin(X)

        x = smbo.choose_next(X, Y)[0].get_array()

        assert x.shape == (2,)

    def test_rng(self):
        smbo = SMBO(self.scenario, None)
        self.assertIsInstance(smbo.rng, np.random.RandomState)
        smbo = SMBO(self.scenario, 1)
        rng = np.random.RandomState(1)
        self.assertIsInstance(smbo.rng, np.random.RandomState)
        smbo = SMBO(self.scenario, rng)
        self.assertIs(smbo.rng, rng)
        #ML: I don't understand the following line and it throws an error 
        self.assertRaisesRegexp(TypeError, "Unknown type <class 'str'> for argument "
                                          'rng. Only accepts None, int or '
                                          'np.random.RandomState',
                               SMBO, self.scenario, 'BLA')

if __name__ == "__main__":
    unittest.main()
