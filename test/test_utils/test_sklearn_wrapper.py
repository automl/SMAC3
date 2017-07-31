'''
Created on July 27, 2017

@author: Jan N. van Rijn
'''
import unittest

import numpy as np

from functools import partial

from sklearn import datasets
from sklearn.dummy import DummyClassifier
from sklearn.model_selection._split import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.testing import assert_raises

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.utils.sklearn_wrapper import ModelBasedOptimization


class SklearnWrapperTest(unittest.TestCase):

    @staticmethod
    def _config_space_to_param_grid(config_space):
        ''' Converts config space into the scikit learn param grid format. Could be useful utility function, but
        does not contain the right functionality yet.
        '''
        param_grid = {}

        for hyperparameter in config_space.get_hyperparameters():
            if isinstance(hyperparameter, CategoricalHyperparameter):
                param_grid[hyperparameter.name] = hyperparameter.choices
            elif isinstance(hyperparameter, UniformIntegerHyperparameter):
                if hyperparameter.log:
                    raise ValueError('Parameter logscale unimplemented: %s' + str(hyperparameter.__class__.__name__))
                param_grid[hyperparameter.name] = list(range(hyperparameter.lower, hyperparameter.upper+1))
            else:
                raise ValueError('Parameter type unimplemented: %s' + str(hyperparameter.__class__.__name__))

        return param_grid

    def _compare_with_smac(self, X, y, classifier, config_space, random_seed, n_iter):
        """
        Checks whether a vanilla SMAC run and the wrapped SMAC run yield the same:
         - incumbent
         - evaluation on the incumbent
        """
        # important for testing: cv is None and y is classification
        mbo_wrapper = ModelBasedOptimization(classifier, self._config_space_to_param_grid(config_space),
                                             random_state=random_seed, verbose=3, n_iter=n_iter)
        # make a partial for running SMAC under same conditions
        obj_function = partial(mbo_wrapper._obj_function, base_estimator=classifier, cv_iter=list(StratifiedKFold(3).split(X, y)), X=X, y=y)
        mbo_wrapper.fit(X, y)

        # Smoke test the score etc:
        mbo_wrapper.score(X, y)
        mbo_wrapper.predict_proba(X)

        # Test exception handling on scoring
        mbo_wrapper.scoring = 'sklearn'
        assert_raises(ValueError, mbo_wrapper.fit, X, y)

        mbo_params = mbo_wrapper.best_estimator_.get_params()

        # SMAC scenario oject
        scenario = Scenario({"run_obj": "quality",  "runcount-limit": n_iter, "cs": config_space, "deterministic": "true", "memory_limit": 3072})

        # To optimize, we pass the function to the SMAC-object
        smac = SMAC(scenario=scenario, rng=random_seed,
                    tae_runner=obj_function)
        smac_incumbent = smac.optimize()
        smac_inc_value = np.mean(np.array(smac.get_tae_runner().run(smac_incumbent, 1)[3]['test_scores']))

        # note that multiple configurations could have led to the same
        # SMAC always takes the most recently found # TODO: check assumption
        # so we find the last idx in mbo results
        mbo_idx = n_iter - 1 - np.argmax(mbo_wrapper.cv_results_['mean_test_score'][::-1])

        self.assertAlmostEqual(mbo_wrapper.best_score_, smac_inc_value, 2) # apparently, scikit-learn search returns only 2 digits
        for param_name, param_value in smac_incumbent.get_dictionary().items():
            self.assertIn(param_name, mbo_params)
            self.assertEqual(mbo_wrapper.cv_results_['param_%s'%param_name][mbo_idx], param_value, msg="param: %s" %param_name)

    def test_mbo_wrapper_dummy(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        classifier = DummyClassifier()
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformIntegerHyperparameter("random_state", 1, 50, default=1))

        self._compare_with_smac(X, y, classifier, cs, random_seed=42, n_iter=5)

    def test_mbo_wrapper_decision_tree(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        classifier = DecisionTreeClassifier()
        cs = ConfigurationSpace()
        cs.add_hyperparameter(UniformIntegerHyperparameter('max_leaf_nodes', 4, 128, default=4))
        cs.add_hyperparameter(UniformIntegerHyperparameter('min_samples_leaf', 1, 128, default=1))
        cs.add_hyperparameter(UniformIntegerHyperparameter('max_depth', 1, 128, default=1))
        cs.add_hyperparameter(CategoricalHyperparameter('criterion', ['gini', 'entropy'], default='gini'))

        self._compare_with_smac(X, y, classifier, cs, random_seed=42, n_iter=5)


