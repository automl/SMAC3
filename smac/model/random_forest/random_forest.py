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
        n_trees: int = N_TREES,
        n_points_per_tree: int = -1,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        eps_purity: float = 1e-8,
        max_nodes: int = 2**20,
        bootstrapping: bool = True,
        log_y: bool = False,
        instance_features: dict[str, list[int | float]] | None = None,
        pca_components: int | None = 7,
        seed: int = 0,
    ) -> None:
        super().__init__(
            configspace=configspace,
            instance_features=instance_features,
            pca_components=pca_components,
            seed=seed,
        )

        max_features = 0 if ratio_features > 1.0 else max(1, int(len(self._types) * ratio_features))

        self._rf_opts = regression.forest_opts()
        self._rf_opts.num_trees = n_trees
        self._rf_opts.do_bootstrapping = bootstrapping
        self._rf_opts.tree_opts.max_features = max_features
        self._rf_opts.tree_opts.min_samples_to_split = min_samples_split
        self._rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
        self._rf_opts.tree_opts.max_depth = max_depth
        self._rf_opts.tree_opts.epsilon_purity = eps_purity
        self._rf_opts.tree_opts.max_num_nodes = max_nodes
        self._rf_opts.compute_law_of_total_variance = False
        self._rf: BinaryForest | None = None
        self._log_y = log_y

        # Case to `int` incase we get an `np.integer` type
        self._rng = regression.default_random_engine(int(seed))

        self._n_trees = n_trees
        self._n_points_per_tree = n_points_per_tree
        self._ratio_features = ratio_features
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_depth = max_depth
        self._eps_purity = eps_purity
        self._max_nodes = max_nodes
        self._bootstrapping = bootstrapping

        # This list well be read out by save_iteration() in the solver
        # self._hypers = [
        #    n_trees,
        #    max_nodes,
        #    bootstrapping,
        #    n_points_per_tree,
        #    ratio_features,
        #    min_samples_split,
        #    min_samples_leaf,
        #    max_depth,
        #    eps_purity,
        #    self._seed,
        # ]

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "n_trees": self._n_trees,
                "n_points_per_tree": self._n_points_per_tree,
                "ratio_features": self._ratio_features,
                "min_samples_split": self._min_samples_split,
                "min_samples_leaf": self._min_samples_leaf,
                "max_depth": self._max_depth,
                "eps_purity": self._eps_purity,
                "max_nodes": self._max_nodes,
                "bootstrapping": self._bootstrapping,
                "pca_components": self._pca_components,
            }
        )

        return meta

    def _train(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        X = self._impute_inactive(X)
        y = y.flatten()

        # self.X = X
        # self.y = y.flatten()

        if self._n_points_per_tree <= 0:
            self._rf_opts.num_data_points_per_tree = X.shape[0]
        else:
            self._rf_opts.num_data_points_per_tree = self._n_points_per_tree

        self._rf = regression.binary_rss_forest()
        self._rf.options = self._rf_opts

        data = self._init_data_container(X, y)
        self._rf.fit(data, rng=self._rng)

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
            preds_as_array = np.zeros((X.shape[0], self._rf_opts.num_trees, third_dimension)) * np.NaN
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
                mean_, var = self._rf.predict_mean_var(row_X)
                means.append(mean_)
                vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicts mean and variance marginalized over all instances.

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
