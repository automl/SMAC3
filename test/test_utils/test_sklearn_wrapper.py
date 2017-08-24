"""
author = "Jan N. van Rijn"
license = "3-clause BSD"
"""
import unittest

import numpy as np

from functools import partial

from sklearn import datasets
from sklearn.dummy import DummyClassifier
from sklearn.model_selection._split import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.testing import assert_raises
from sklearn.metrics.scorer import _check_multimetric_scoring

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter

from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from smac.utils.sklearn_wrapper import ModelBasedOptimization
from sklearn.metrics.scorer import check_scoring


class SklearnWrapperTest(unittest.TestCase):

    def setUp(self):
        self.cs_tree = ConfigurationSpace()
        self.cs_tree.add_hyperparameter(UniformIntegerHyperparameter('max_leaf_nodes', 4, 32, default=4))
        self.cs_tree.add_hyperparameter(UniformIntegerHyperparameter('min_samples_leaf', 1, 128, default=1))
        self.cs_tree.add_hyperparameter(UniformIntegerHyperparameter('max_depth', 1, 32, default=1))
        self.cs_tree.add_hyperparameter(CategoricalHyperparameter('criterion', ['gini', 'entropy'], default='gini'))

        self.cs_dummy = ConfigurationSpace()
        self.cs_dummy.add_hyperparameter(UniformIntegerHyperparameter("random_state", 1, 50, default=1))

    @staticmethod
    def _config_space_to_param_grid(config_space):
        """ Converts config space into the scikit learn param grid format. Could be useful utility function, but
        does not contain the right functionality yet.
        """
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

        # === COMPARE INCUMBENTS ===
        # note that multiple configurations could have led to the same
        # SMAC always takes the most recently found (in case of 1 instance problems)
        # so we find all the maximized indices in mbo results
        scores = mbo_wrapper.cv_results_['mean_test_score']
        mbo_max_indices = [i for i, j in enumerate(scores) if j == max(scores)]

        print(mbo_wrapper.best_score_)
        print(smac_inc_value)

        self.assertAlmostEqual(mbo_wrapper.best_score_, smac_inc_value, 2) # apparently, scikit-learn search returns only 2 digits
        foundEqual = False
        for mbo_idx in mbo_max_indices:
            for param_name, param_value in smac_incumbent.get_dictionary().items():
                self.assertIn(param_name, mbo_params)
                if mbo_wrapper.cv_results_['param_%s'%param_name][mbo_idx] != param_value:
                    break
            foundEqual = True
        self.assertTrue(foundEqual)

        # === COMPARE TRAJECTORIES ===
        for index, runkey in enumerate(smac.get_runhistory().data.keys()):
            smac_config = smac.get_runhistory().ids_config[runkey[0]].get_dictionary()
            wrapper_config = mbo_wrapper.cv_results_['params'][index]
            self.assertEqual(smac_config, wrapper_config, msg='Iteration Unequal: %d' %index)

    def test_mbo_wrapper_dummy(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        classifier = DummyClassifier()

        self._compare_with_smac(X, y, classifier, self.cs_dummy, random_seed=42, n_iter=5)

    def test_mbo_wrapper_decision_tree(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        classifier = DecisionTreeClassifier()

        self._compare_with_smac(X, y, classifier, self.cs_tree, random_seed=42, n_iter=5)

    def test_obj_function(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        cv = list(StratifiedKFold(3).split(X, y))
        classifier = DecisionTreeClassifier()
        random_state = 1
        wrapper = ModelBasedOptimization(estimator=classifier,
                                         param_distributions=self._config_space_to_param_grid(self.cs_tree),
                                         n_iter=5,
                                         random_state=random_state, verbose=3)
        # spoof fit function
        scorers, self.multimetric_ = _check_multimetric_scoring(wrapper.estimator, scoring=wrapper.scoring)
        wrapper.scorers = scorers

        score, res_per_fold = wrapper._obj_function(configuration=self.cs_tree.sample_configuration(1),
                                                    seed=random_state, instance=None, cv_iter=cv,
                                                    base_estimator=classifier, X=X, y=y)

        self.assertGreaterEqual(min(res_per_fold['score_time']), 0.0)
        self.assertGreaterEqual(min(res_per_fold['test_scores']), 0.0)
        self.assertGreaterEqual(min(res_per_fold['test_sample_counts']), 0.0)
        self.assertGreaterEqual(min(res_per_fold['train_scores']), 0.0)
        self.assertGreaterEqual(min(res_per_fold['fit_time']), 0.0)

        self.assertLessEqual(max(res_per_fold['test_scores']), 1.0)
        self.assertLessEqual(max(res_per_fold['test_sample_counts']), len(y))
        self.assertLessEqual(max(res_per_fold['train_scores']), 1.0)

    def test_dummy_param_distributions_to_config_space(self):
        param_dist = dict({'random_state': list(range(1, 50 + 1))})

        self.assertEqual(ModelBasedOptimization._param_distributions_to_config_space(param_dist), self.cs_dummy)

    def test_tree_param_distributions_to_config_space(self):
        param_dist = dict({'max_leaf_nodes': list(range(4, 32 + 1)),
                           'min_samples_leaf': list(range(1, 128 + 1)),
                           'max_depth': list(range(1, 32 + 1)),
                           'criterion': ['gini', 'entropy']})

        self.assertEqual(ModelBasedOptimization._param_distributions_to_config_space(param_dist), self.cs_tree)
