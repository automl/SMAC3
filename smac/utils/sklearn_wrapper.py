"""
author = "Jan N. van Rijn"
license = "3-clause BSD"
"""

import numpy as np
import warnings
import six

from collections import defaultdict
from functools import partial
from typing import Union, Callable, Dict, List

from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter

from scipy.stats import rankdata

from sklearn.base import is_classifier, clone, BaseEstimator
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.model_selection._split import check_cv, BaseCrossValidator
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._validation import _fit_and_score, _aggregate_score_dicts
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.validation import indexable

from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario


class ModelBasedOptimization(BaseSearchCV):
    """
    Scikit-Learn wrapper for SMAC
    """

    def __init__(self, estimator: BaseEstimator, param_distributions: Dict[str, List], n_iter: int = 200,
                 # scoring: Union[str, Callable, None] = None, NOT SUPPORTED FOR NOW
                 fit_params: Union[Dict, None] = None, iid: bool = True,
                 refit: bool = True, cv: Union[BaseCrossValidator, int] = None, verbose: int = 0,
                 random_state: Union[int, np.random.RandomState] = None, error_score: Union[str, int] = 'raise',
                 return_train_score: bool = True):
        """Scikit-learn wrapper for Sequential Model Based Optimization (SMAC). This allows to use SMAC together
        with various tools relying on the scikit-learn API, such as OpenML.org.

        Works similar to the RandomizedSearchCV class. For detailed and up-to-date
        parameter descriptions, please see:
        http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

        Main difference: requires dict of lists for param_distributions (whereas the base-class
        is more liberal and also allows for scipy distributions).

        Parameters
        ---------
        estimator : estimator object.
            A scikit-learn classifier. Should inherit from
            sklearn.base.BaseEstimator. Either estimator needs
            to provide a ``score`` function, or ``scoring`` must be passed.

        param_distributions : dict
            Dictionary with parameters names (string) as keys and
            lists of parameters to try. Distributions must provide a ``rvs``
            method for sampling (such as those from scipy.stats.distributions).
            If a list is given, it is sampled uniformly.

        n_iter : int, default=10
            Number of parameter settings that are sampled. n_iter trades
            off runtime vs quality of the solution.

        scoring : string, callable or None, default=None
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
            If ``None``, the ``score`` method of the estimator is used.

        fit_params : dict, optional
            Deprecated feature of Scikit-learn

        iid : boolean, default=True
            If True, the data is assumed to be identically distributed across
            the folds, and the loss minimized is the total loss per sample,
            and not the mean loss across the folds.

        refit : bool
            Deprecated feature of Scikit-learn

        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default 3-fold cross validation,
              - integer, to specify the number of folds in a `(Stratified)KFold`,
              - An object to be used as a cross-validation generator.
              - An iterable yielding train, test splits.

            For integer/None inputs, if the estimator is a classifier and ``y`` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used.

            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.

        refit : boolean, default=True
            Refit the best estimator with the entire dataset.
            If "False", it is impossible to make predictions using
            this RandomizedSearchCV instance after fitting.

        verbose : integer
            Controls the verbosity: the higher, the more messages.

        random_state : int or RandomState
            Pseudo random number generator state used for random uniform sampling
            from lists of possible values instead of scipy.stats distributions.

        error_score : 'raise' (default) or numeric
            Value to assign to the score if an error occurs in estimator fitting.
            If set to 'raise', the error is raised. If a numeric value is given,
            FitFailedWarning is raised. This parameter does not affect the refit
            step, which will always raise the error.

        return_train_score : boolean, default=True
            If ``'False'``, the ``cv_results_`` attribute will not include training
            scores.
            """

        # Current implementation does not like distributions yet.
        for param, values in param_distributions.items():
            if not isinstance(values, list):
                raise ValueError('Not implemented (yet): Wrapper does not work with distributions yet. Please use lists. ')
        # For now, this is not supported 
        scoring = None

        self.random_state = random_state
        self.param_distributions = param_distributions
        self.n_iter = n_iter

        # sets important hyperparameters of search module
        super(ModelBasedOptimization, self).__init__(
              estimator=estimator, scoring=scoring, fit_params=fit_params,
              n_jobs=1, iid=iid, refit=refit, cv=cv, verbose=verbose,
              pre_dispatch=False, error_score=error_score,
              return_train_score=return_train_score)

    def fit(self, X: Union[List[List], np.array], y: Union[List, np.array, None] = None,
            groups: Union[List, np.array, None] = None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """

        # the following part is directly copied from Scikit-learns RandomizedSearchCV class
        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        #################################### START DIFFERENCE WITH BASESEARCH_CV ####################################
        cv_iter = list(cv.split(X, y, groups))
        self.config_space = self._param_distributions_to_config_space(self.param_distributions)
        self.scorers = scorers
        scenario = Scenario({"run_obj": "quality", "runcount-limit": self.n_iter, "cs": self.config_space, "deterministic": "true", "memory_limit": 3072})

        obj_function = partial(self._obj_function, base_estimator=base_estimator, cv_iter=cv_iter, X=X, y=y)

        smac = SMAC(scenario=scenario, rng=self.random_state, tae_runner=obj_function)
        smac.optimize()

        history = smac.get_runhistory()

        out = {'train_scores': [], 'test_scores': [], 'test_sample_counts': [], 'fit_time': [], 'score_time': [], 'parameters': []}
        for RunKey, RunValue in history.data.items():
            config_id = RunKey[0]
            configuration = history.ids_config[config_id]
            for info_key, info_value_list in RunValue.additional_info.items():
                for _, info_value in enumerate(info_value_list):
                    out[info_key].append(info_value)
            for _ in range(len(cv_iter)):
                out['parameters'].append(configuration.get_dictionary())

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            train_scores = {'score': out['train_scores']}
        test_scores = {'score': out['test_scores']}
        test_sample_counts = out['test_sample_counts']
        fit_time = out['fit_time']
        score_time = out['score_time']
        parameters = out['parameters']

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)
        #################################### END DIFFERENCE WITH BASESEARCH_CV ####################################

        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates, ),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)
        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if self.iid else None)
            if self.return_train_score:
                _store('train_%s' % scorer_name, train_scores[scorer_name],
                       splits=True)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = candidate_params[self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _obj_function(self, configuration: Configuration, seed: int, instance: str, base_estimator: BaseEstimator,
                      cv_iter, X: Union[List[List], np.array], y: Union[List, np.array]):
        results_per_fold = {'train_scores': [], 'test_scores': [], 'test_sample_counts': [], 'fit_time': [], 'score_time': []}
        for train, test in cv_iter:
            # fit and score returns a list, containing
            # - train_scores, test_scores, test_sample_counts, fit_time, score_time, parameters (iff self.return_train_score)
            # - test_scores, test_sample_counts, fit_time, score_time, parameters (otherwise)
            results = _fit_and_score(clone(base_estimator), X, y, self.scorers,
                                     train, test, self.verbose, configuration.get_dictionary(),
                                     fit_params=self.fit_params,
                                     return_train_score=self.return_train_score,
                                     return_n_test_samples=True,
                                     return_times=True, return_parameters=True,
                                     error_score=self.error_score)

            if self.return_train_score:
                results_per_fold['train_scores'].append(results[0]['score'])
                results_per_fold['test_scores'].append(results[1]['score'])
                results_per_fold['test_sample_counts'].append(results[2])
                results_per_fold['fit_time'].append(results[3])
                results_per_fold['score_time'].append(results[4])
            else:
                results_per_fold['test_scores'].append(results[0]['score'])
                results_per_fold['test_sample_counts'].append(results[1])
                results_per_fold['fit_time'].append(results[2])
                results_per_fold['score_time'].append(results[3])
                if 'train_scores' in results_per_fold:
                    del results_per_fold['train_scores']

        score = np.mean(np.array(results_per_fold['test_scores']))
        return -1 * score, results_per_fold

    @staticmethod
    def _param_distributions_to_config_space(param_distributions: Dict[str, List]):
        cs = ConfigurationSpace()

        for param, distribution in param_distributions.items():
            if not isinstance(distribution, list):
                raise ValueError('Currently, only param_distributions of type list are allowed. ')
            if all(isinstance(x, int) for x in distribution):
                minimum = min(distribution)
                maximum = max(distribution)
                hyperparameter = UniformIntegerHyperparameter(param, minimum, maximum, minimum)
            elif all(isinstance(x, float) for x in distribution):
                minimum = min(distribution)
                maximum = max(distribution)
                hyperparameter = UniformFloatHyperparameter(param, minimum, maximum, minimum)
            else:
                hyperparameter = CategoricalHyperparameter(param, distribution, default=distribution[0])
            cs.add_hyperparameter(hyperparameter)

        return cs
