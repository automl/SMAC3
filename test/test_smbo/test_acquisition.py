import unittest
import unittest.mock

import numpy as np

from smac.optimizer.acquisition import EI, LogEI


class ConfigurationMock(object):

    def __init__(self, values=None):
        self.values = values
        self.configuration_space = unittest.mock.MagicMock()
        self.configuration_space.get_hyperparameters.return_value = []

    def get_array(self):
        return self.values


class MockModel(object):
    def __init__(self, num_targets=1):
        self.num_targets = num_targets

    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] *
                        self.num_targets).reshape((-1, 1)), \
               np.array([np.mean(X, axis=1).reshape((1, -1))] *
                        self.num_targets).reshape((-1, 1))


class TestEI(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.ei = EI(self.model)

    def test_1xD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.3989422804014327)

    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = ([ConfigurationMock([0.0, 0.0, 0.0]),
                           ConfigurationMock([0.1, 0.1, 0.1]),
                           ConfigurationMock([1.0, 1.0, 1.0])])
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 0.0)
        self.assertAlmostEqual(acq[1][0], 0.90020601136712231)
        self.assertAlmostEqual(acq[2][0], 0.3989422804014327)

    def test_1x1(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.3989422804014327)

    def test_Nx1(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([0.0001]),
                          ConfigurationMock([1.0]),
                          ConfigurationMock([2.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 0.9999)
        self.assertAlmostEqual(acq[1][0], 0.3989422804014327)
        self.assertAlmostEqual(acq[2][0], 0.19964122837424575)

    def test_zero_variance(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[0.0]])
        acq = np.array(X)
        self.assertAlmostEqual(acq[0][0], 0.0)
        
class TestLogEI(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.ei = LogEI(self.model)
        
    def test_1xD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([1.0, 1.0, 1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.6480973967332011)
        
    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0)
        configurations = [ConfigurationMock([0.1, 0.0, 0.0]),
                          ConfigurationMock([0.1, 0.1, 0.1]),
                          ConfigurationMock([1.0, 1.0, 1.0])]
        acq = self.ei(configurations)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 1.6670107375002425)
        self.assertAlmostEqual(acq[1][0], 1.5570607606556273)
        self.assertAlmostEqual(acq[2][0], 0.6480973967332011)
