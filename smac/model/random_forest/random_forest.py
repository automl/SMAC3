from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import ConfigurationSpace
from pyrfr import regression
from pyrfr.regression import binary_rss_forest as BinaryForest
from pyrfr.regression import default_data_container as DataContainer

from smac.constants import N_TREES, VERY_SMALL_NUMBER
from smac.model.random_forest import AbstractRandomForest

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

import numpy as np
from sklearn.ensemble._forest import ForestRegressor
from sklearn.tree import DecisionTreeRegressor


class EPMRandomForest(ForestRegressor):
    def __init__(
        self,
        n_estimators: int = 100,
        *,
        log_y=False,
        cross_trees_variance=False,
        criterion="squared_error",
        splitter="random",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs=None,
        random_state=None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ) -> None:
        """A decision tree regressor.

        Read more in the :ref:`User Guide <tree>`.

        Parameters
        ----------
        n_estimators : int, default=100
        The number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

        criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, \
                default="squared_error"
            The function to measure the quality of a split. Supported criteria
            are "squared_error" for the mean squared error, which is equal to
            variance reduction as feature selection criterion and minimizes the L2
            loss using the mean of each terminal node, "friedman_mse", which uses
            mean squared error with Friedman's improvement score for potential
            splits, "absolute_error" for the mean absolute error, which minimizes
            the L1 loss using the median of each terminal node, and "poisson" which
            uses reduction in Poisson deviance to find splits.
            Training using "absolute_error" is significantly slower
            than when using "squared_error".

            .. versionadded:: 0.18
            Mean Absolute Error (MAE) criterion.

            .. versionadded:: 1.0
            Poisson criterion.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.

            .. versionchanged:: 0.18
            Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.

            .. versionchanged:: 0.18
            Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : {"sqrt", "log2", None}, int or float, default=1.0
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
            `max(1, int(max_features * n_features_in_))` features are considered at each
            split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None or 1.0, then `max_features=n_features`.

            .. note::
                The default of 1.0 is equivalent to bagged trees and more
                randomness can be achieved by setting smaller values, e.g. 0.3.

            .. versionchanged:: 1.1
                The default of `max_features` changed from `"auto"` to 1.0.

            .. deprecated:: 1.1
                The `"auto"` option was deprecated in 1.1 and will be removed
                in 1.3.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        max_leaf_nodes : int, default=None
            Grow trees with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.

            The weighted impurity decrease equation is the following::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.

            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.

            .. versionadded:: 0.19

        bootstrap : bool, default=True
            Whether bootstrap samples are used when building trees. If False, the
            whole dataset is used to build each tree.

        oob_score : bool, default=False
            Whether to use out-of-bag samples to estimate the generalization score.
            Only available if bootstrap=True.

        n_jobs : int, default=None
            The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
            :meth:`decision_path` and :meth:`apply` are all parallelized over the
            trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context. ``-1`` means using all processors. See :term:`Glossary
            <n_jobs>` for more details.

        random_state : int, RandomState instance or None, default=None
            Controls both the randomness of the bootstrapping of the samples used
            when building trees (if ``bootstrap=True``) and the sampling of the
            features to consider when looking for the best split at each node
            (if ``max_features < n_features``).
            See :term:`Glossary <random_state>` for details.

        verbose : int, default=0
            Controls the verbosity when fitting and predicting.

        warm_start : bool, default=False
            When set to ``True``, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit a whole
            new forest. See :term:`Glossary <warm_start>` and
            :ref:`gradient_boosting_warm_start` for details.

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22

        max_samples : int or float, default=None
            If bootstrap is True, the number of samples to draw from X
            to train each base estimator.

            - If None (default), then draw `X.shape[0]` samples.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples. Thus,
            `max_samples` should be in the interval `(0.0, 1.0]`.

            .. versionadded:: 0.22

        criterion : {"squared_error", "friedman_mse", "absolute_error", \
                "poisson"}, default="squared_error"
            The function to measure the quality of a split. Supported criteria
            are "squared_error" for the mean squared error, which is equal to
            variance reduction as feature selection criterion and minimizes the L2
            loss using the mean of each terminal node, "friedman_mse", which uses
            mean squared error with Friedman's improvement score for potential
            splits, "absolute_error" for the mean absolute error, which minimizes
            the L1 loss using the median of each terminal node, and "poisson" which
            uses reduction in Poisson deviance to find splits.

            .. versionadded:: 0.18
            Mean Absolute Error (MAE) criterion.

            .. versionadded:: 0.24
                Poisson deviance criterion.

        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to choose
            the best random split.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.

            .. versionchanged:: 0.18
            Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.

            .. versionchanged:: 0.18
            Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
            `max(1, int(max_features * n_features_in_))` features are considered at each
            split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            .. deprecated:: 1.1
                The `"auto"` option was deprecated in 1.1 and will be removed
                in 1.3.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the estimator. The features are always
            randomly permuted at each split, even if ``splitter`` is set to
            ``"best"``. When ``max_features < n_features``, the algorithm will
            select ``max_features`` at random at each split before finding the best
            split among them. But the best found split may vary across different
            runs, even if ``max_features=n_features``. That is the case, if the
            improvement of the criterion is identical for several splits and one
            split has to be selected at random. To obtain a deterministic behaviour
            during fitting, ``random_state`` has to be fixed to an integer.
            See :term:`Glossary <random_state>` for details.

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.

            The weighted impurity decrease equation is the following::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.

            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.

            .. versionadded:: 0.19

        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.

            .. versionadded:: 0.22
        """
        super().__init__(
            DecisionTreeRegressor(),
            n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst
        self.splitter = splitter
        self.log_y = log_y
        self.cross_trees_variance = cross_trees_variance

    def fit(self, X, y, sample_weight=None):
        assert sample_weight is None, "Sample weights are not supported"
        super().fit(X=X, y=y, sample_weight=sample_weight)

        self.trainX = X
        self.trainY = y
        if self.log_y:
            for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
                curX = X[samples_idx]
                curY = y[samples_idx]
                preds = tree.apply(curX)
                for k in np.unique(preds):
                    tree.tree_.value[k, 0, 0] = np.log(np.exp(curY[preds == k]).mean())

    def predict(self, X):
        preds = []
        for tree, samples_idx in zip(self.estimators_, self.estimators_samples_):
            preds.append(tree.predict(X))
        preds = np.array(preds).T

        means = preds.mean(axis=1)
        vars = preds.var(axis=1)

        return means.reshape(-1, 1), vars.reshape(-1, 1)


class RandomForest(AbstractRandomForest):
    """Random forest that takes instance features into account.

    Parameters
    ----------
    n_trees : int, defaults to `N_TREES`
        The number of trees in the random forest.
    n_points_per_tree : int, defaults to -1
        Number of points per tree. If the value is smaller than 0, the number of samples will be used.
    ratio_features : float, defaults to 5.0 / 6.0
        The ratio of features that are considered for splitting.
    min_samples_split : int, defaults to 3
        The minimum number of data points to perform a split.
    min_samples_leaf : int, defaults to 3
        The minimum number of data points in a leaf.
    max_depth : int, defaults to 2**20
        The maximum depth of a single tree.
    eps_purity : float, defaults to 1e-8
        The minimum difference between two target values to be considered.
    max_nodes : int, defaults to 2**20
        The maximum total number of nodes in a tree.
    bootstrapping : bool, defaults to True
        Enables bootstrapping.
    log_y: bool, defaults to False
        The y values (passed to this random forest) are expected to be log(y) transformed.
        This will be considered during predicting.
    instance_features : dict[str, list[int | float]] | None, defaults to None
        Features (list of int or floats) of the instances (str). The features are incorporated into the X data,
        on which the model is trained on.
    pca_components : float, defaults to 7
        Number of components to keep when using PCA to reduce dimensionality of instance features.
    seed : int
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        n_points_per_tree: int = -1,
        ratio_features: float = 5.0 / 6.0,
        eps_purity: float = 1e-8,
        log_y: bool = False,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
        n_estimators: int = N_TREES,  # TODO: HP: 100 is default in sklearn, here it's 10
        cross_trees_variance=False,
        criterion="squared_error",
        splitter="random",  # Should not be changed
        max_depth: int = 2**20,  # TODO: HP: None is default in sklearn, here it's 2**20
        min_samples_split: int = 3,  # TODO: HP: 2 is default in sklearn, here it's 3
        min_samples_leaf: int = 3,  # TODO: HP: 1 is default in sklearn, here it's 3
        min_weight_fraction_leaf=0.0,
        # max_features=1.0,  # Set by ratio_features
        max_leaf_nodes: int = 2**20,  # TODO: HP: None is default in sklearn, here it's 2**20
        min_impurity_decrease=0.0,
        bootstrap: bool = True,  # TODO HP: False is default in sklearn, here it's True
        oob_score: bool = False,
        n_jobs=None,
        # random_state=None,  # Set by seed
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ) -> None:
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        max_features = 0 if ratio_features > 1.0 else max(1, int(len(self._types) * ratio_features))

        self._rf: EPMRandomForest | None = None
        self._rng = np.random.default_rng(seed=seed)

        self._log_y = log_y
        self._rf_opts = {
            "n_estimators": n_estimators,
            "cross_trees_variance": cross_trees_variance,
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "bootstrap": bootstrap,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "warm_start": warm_start,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "monotonic_cst": monotonic_cst,
            "random_state": seed,
            "max_features": max_features,
            "log_y": log_y,
        }

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(self._rf_opts)

        return meta

    def _train(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        X = self._impute_inactive(X)
        y = y.flatten()

        self._rf = EPMRandomForest(**self._rf_opts)

        self._rf.fit(X, y)

        return self

    def _init_data_container(self, X: np.ndarray, y: np.ndarray) -> DataContainer:
        """Fills a pyrfr default data container s.t. the forest knows categoricals and bounds for continous data.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameter + #features]
            Input data points.
        Y : np.ndarray [#samples, #objectives]
            The corresponding target values.

        Returns
        -------
        data : DataContainer
            The filled data container that pyrfr can interpret.
        """
        # Retrieve the types and the bounds from the ConfigSpace
        data = regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self._bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)

        return data

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != len(self._types):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._types), X.shape[1]))

        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")

        assert self._rf is not None
        X = self._impute_inactive(X)

        if self._log_y:
            all_preds = []
            third_dimension = 0

            # Gather data in a list of 2d arrays and get statistics about the required size of the 3d array
            for row_X in X:
                preds_per_tree = self._rf.all_leaf_values(row_X)
                all_preds.append(preds_per_tree)
                max_num_leaf_data = max(map(len, preds_per_tree))
                third_dimension = max(max_num_leaf_data, third_dimension)

            # Transform list of 2d arrays into a 3d array
            preds_as_array = np.zeros((X.shape[0], self._rf_opts.num_trees, third_dimension)) * np.nan
            for i, preds_per_tree in enumerate(all_preds):
                for j, pred in enumerate(preds_per_tree):
                    preds_as_array[i, j, : len(pred)] = pred

            # Do all necessary computation with vectorized functions
            preds_as_array = np.log(np.nanmean(np.exp(preds_as_array), axis=2) + VERY_SMALL_NUMBER)

            # Compute the mean and the variance across the different trees
            means = preds_as_array.mean(axis=1)
            vars_ = preds_as_array.var(axis=1)
        else:
            means, vars_ = [], []
            for row_X in X:
                mean_, var = self._rf.predict(row_X)
                means.append(mean_)
                vars_.append(var)

        # means, vars = self._rf.predict(X)
        return means.reshape((-1, 1)), vars.reshape((-1, 1))

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts mean and variance marginalized over all instances.

        Under the hood: Collects the predictions for each tree in the forest
        for multiple configurations over a set of instances. Each configuration
        vector is combined with all the instance feature vectors. Based on the
        response values over all these feature vectors the mean is computed.
        In the case of log transformation the response values are decompressed
        before averaging.

        Note
        ----
        The method is random forest specific and follows the SMAC2 implementation. It requires
        no distribution assumption to marginalize the uncertainty estimates.

        Parameters
        ----------
        X : np.ndarray [#samples, #hyperparameter + #features]
            Input data points.

        Returns
        -------
        means : np.ndarray [#samples, 1]
            The predictive mean.
        vars : np.ndarray [#samples, 1]
            The predictive variance.
        """
        if self._n_features == 0:
            mean_, var = self.predict(X)
            assert var is not None

            var[var < self._var_threshold] = self._var_threshold
            var[np.isnan(var)] = self._var_threshold

            return mean_, var

        assert self._instance_features is not None

        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != len(self._bounds):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._bounds), X.shape[1]))

        assert self._rf is not None
        X = self._impute_inactive(X)

        X_feat = list(self._instance_features.values())
        dat_ = self._rf.predict_marginalized_over_instances_batch(X, X_feat, self._log_y)
        dat_ = np.array(dat_)

        # 3. compute statistics across trees
        mean_ = dat_.mean(axis=1)
        var = dat_.var(axis=1)

        if var is None:
            raise RuntimeError("The variance must not be none.")

        var[var < self._var_threshold] = self._var_threshold

        if len(mean_.shape) == 1:
            mean_ = mean_.reshape((-1, 1))
        if len(var.shape) == 1:
            var = var.reshape((-1, 1))

        return mean_, var
