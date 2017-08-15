import unittest

import numpy as np

from smac.optimizer.acquisition import EI, LogEI, EI_WITH_CONSTRAINTS


class MockModel(object):
    def __init__(self, num_targets=1):
        self.num_targets = num_targets

    def predict_marginalized_over_instances(self, X):
        return np.array([np.mean(X, axis=1).reshape((1, -1))] *
                        self.num_targets).reshape((-1, 1)), \
               np.array([np.mean(X, axis=1).reshape((1, -1))] *
                        self.num_targets).reshape((-1, 1))
                        
class MockConstraintModel(object):

    def predict_marginalized_over_instances(self, X):
        rs = np.random.RandomState(1)
        self.current_probs = rs.rand(X.shape[0], 2)
        return self.current_probs
        
class TestEI_WITH_CONSTRAINTS(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.constraint_model = MockConstraintModel()
        self.ei_with_constraints = EI_WITH_CONSTRAINTS(model=self.model, constraint_model=self.constraint_model)
        self.ei = EI(self.model)
    
    def test_1xD(self):
        X = np.array([[1.0, 1.0, 1.0]])
        self.ei.update(model=self.model, eta=1.0)
        acq_ei = self.ei(X)
        
        self.ei_with_constraints.update(model=self.model, constraint_model=self.constraint_model, eta=1.0)
        acq_ei_with_constraints = self.ei_with_constraints(X)
            
        self.assertEqual(acq_ei_with_constraints.shape, (1, 1))
        self.assertAlmostEqual(acq_ei_with_constraints[0][0], acq_ei[0][0] * self.constraint_model.current_probs[0][0])
        
    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[0.0, 0.0, 0.0],
                      [0.1, 0.1, 0.1],
                      [1.0, 1.0, 1.0]])
        acq_ei = self.ei(X)
        self.ei_with_constraints.update(model=self.model, constraint_model=self.constraint_model, eta=1.0)
        acq_ei_with_constraints = self.ei_with_constraints(X)
        self.assertEqual(acq_ei_with_constraints.shape, (3, 1))
        self.assertAlmostEqual(acq_ei_with_constraints[0][0], 0.0)
        self.assertAlmostEqual(acq_ei_with_constraints[1][0], acq_ei[1][0] * self.constraint_model.current_probs[1][0])
        self.assertAlmostEqual(acq_ei_with_constraints[2][0],  acq_ei[2][0] * self.constraint_model.current_probs[2][0])
        
    def test_1x1(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[1.0]])
        acq_ei = self.ei(X)
        self.ei_with_constraints.update(model=self.model, constraint_model=self.constraint_model, eta=1.0)
        acq_ei_with_constraints = self.ei_with_constraints(X)
        self.assertEqual(acq_ei_with_constraints.shape, (1, 1))
        self.assertAlmostEqual(acq_ei_with_constraints[0][0], acq_ei[0][0]* self.constraint_model.current_probs[0][0])

    def test_Nx1(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[0.0001],
                      [1.0],
                      [2.0]])
        acq_ei = self.ei(X)
        self.ei_with_constraints.update(model=self.model, constraint_model=self.constraint_model, eta=1.0)
        acq_ei_with_constraints = self.ei_with_constraints(X)
        self.assertEqual(acq_ei_with_constraints.shape, (3, 1))
        self.assertAlmostEqual(acq_ei_with_constraints[0][0], acq_ei[0][0]* self.constraint_model.current_probs[0][0])
        self.assertAlmostEqual(acq_ei_with_constraints[1][0], acq_ei[1][0]* self.constraint_model.current_probs[1][0])
        self.assertAlmostEqual(acq_ei_with_constraints[2][0], acq_ei[2][0]* self.constraint_model.current_probs[2][0])

    def test_zero_variance(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[0.0]])
        acq_ei = self.ei(X)
        self.ei_with_constraints.update(model=self.model, constraint_model=self.constraint_model, eta=1.0)
        acq_ei_with_constraints = self.ei_with_constraints(X)
        self.assertAlmostEqual(acq_ei_with_constraints[0][0], 0.0)


class TestEI(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.ei = EI(self.model)

    def test_1xD(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[1.0, 1.0, 1.0]])
        acq = self.ei(X)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.3989422804014327)

    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[0.0, 0.0, 0.0],
                      [0.1, 0.1, 0.1],
                      [1.0, 1.0, 1.0]])
        acq = self.ei(X)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 0.0)
        self.assertAlmostEqual(acq[1][0], 0.90020601136712231)
        self.assertAlmostEqual(acq[2][0], 0.3989422804014327)

    def test_1x1(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[1.0]])
        acq = self.ei(X)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.3989422804014327)

    def test_Nx1(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[0.0001],
                      [1.0],
                      [2.0]])
        acq = self.ei(X)
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
        X = np.array([[1.0, 1.0, 1.0]])
        acq = self.ei(X)
        self.assertEqual(acq.shape, (1, 1))
        self.assertAlmostEqual(acq[0][0], 0.056696236230553559)
        
    def test_NxD(self):
        self.ei.update(model=self.model, eta=1.0)
        X = np.array([[0.0, 0.0, 0.0],
                      [0.1, 0.1, 0.1],
                      [1.0, 1.0, 1.0]])
        acq = self.ei(X)
        self.assertEqual(acq.shape, (3, 1))
        self.assertAlmostEqual(acq[0][0], 0.0)
        self.assertAlmostEqual(acq[1][0], 0.069719643222631633)
        self.assertAlmostEqual(acq[2][0], 0.056696236230553559)
