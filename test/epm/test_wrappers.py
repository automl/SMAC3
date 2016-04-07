import unittest

import numpy as np

from smac.epm.wrappers import UncorrelatedMultiObjectiveWrapper
from smac.epm.rf_with_instances import RandomForestWithInstances


class TestUncorrelatedMultiObjectiveWrapper(unittest.TestCase):
    @unittest.skip('No...')
    def test_train_and_predict_with_rf(self):
        rs = np.random.RandomState(1)
        X = rs.rand(20, 10)
        Y = rs.rand(10, 2)
        model = RandomForestWithInstances(np.zeros((10, )))
        umow = UncorrelatedMultiObjectiveWrapper(model, ['cost', 'ln(runtime)'])
        umow.fit(X[:10], Y)
        print(umow.predict(X[10:]))
        raise ValueError()