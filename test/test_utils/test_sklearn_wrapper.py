'''
Created on July 27, 2017

@author: Jan N. van Rijn
'''
import unittest

import sys

import numpy as np

from sklearn.externals.six.moves import cStringIO as StringIO
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_array_equal

from smac.utils.sklearn_wrapper import ModelBasedOptimization

# Neither of the following two estimators inherit from BaseEstimator,
# to test hyperparameter search on user-defined classifiers.
class MockClassifier(object):
    """Dummy classifier to test the parameter search algorithms"""
    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, Y):
        assert_true(len(X) == len(Y))
        self.classes_ = np.unique(Y)
        return self

    def predict(self, T):
        return T.shape[0]

    def transform(self, X):
        return X + self.foo_param

    def inverse_transform(self, X):
        return X - self.foo_param

    predict_proba = predict
    predict_log_proba = predict
    decision_function = predict

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.
        else:
            score = 0.
        return score

    def get_params(self, deep=False):
        return {'foo_param': self.foo_param}

    def set_params(self, **params):
        self.foo_param = params['foo_param']
        return self


class Sklearn_wrapper_test(unittest.TestCase):

    def test_mbo(self):
        X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
        y = np.array([1, 1, 2, 2])

        # Test that the best estimator contains the right value for foo_param
        clf = MockClassifier()
        grid_search = ModelBasedOptimization(clf, {'foo_param': [1, 2, 3]}, verbose=3, n_iter=2)
        # make sure it selects the smallest parameter in case of ties
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        grid_search.fit(X, y)
        sys.stdout = old_stdout
        assert_equal(grid_search.best_estimator_.foo_param, 2)

        assert_array_equal(grid_search.cv_results_["param_foo_param"].data, [1, 2, 3])

        # Smoke test the score etc:
        grid_search.score(X, y)
        grid_search.predict_proba(X)
        grid_search.decision_function(X)
        grid_search.transform(X)

        # Test exception handling on scoring
        grid_search.scoring = 'sklearn'
        assert_raises(ValueError, grid_search.fit, X, y)

