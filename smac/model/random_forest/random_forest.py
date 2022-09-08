from __future__ import annotations

from typing import Any

import numpy as np
from pyrfr import regression

from ConfigSpace import ConfigurationSpace
from smac.constants import N_TREES, VERY_SMALL_NUMBER
from smac.model.random_forest import AbstractRandomForest

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class RandomForest(AbstractRandomForest):
    """Random forest that takes instance features into account.

    Parameters
    ----------
    seed : int
        The seed that is passed to the random_forest_run library.
    log_y: bool
        y values (passed to this RF) are expected to be log(y) transformed;
        this will be considered during predicting
    num_trees : int
        The number of trees in the random forest.
    do_bootstrapping : bool
        Turns on / off bootstrapping in the random forest.
    n_points_per_tree : int
        Number of points per tree. If <= 0 X.shape[0] will be used
        in _train(X, y) instead
    ratio_features : float
        The ratio of features that are considered for splitting.
    min_samples_split : int
        The minimum number of data points to perform a split.
    min_samples_leaf : int
        The minimum number of data points in a leaf.
    max_depth : int
        The maximum depth of a single tree.
    eps_purity : float
        The minimum difference between two target values to be considered
        different
    max_num_nodes : int
        The maximum total number of nodes in a tree
    instance_features : np.ndarray (I, K)
        Contains the K dimensional instance features of the I different instances
    pca_components : float
        Number of components to keep when using PCA to reduce dimensionality of instance features. Requires to
        set n_feats (> pca_dims).
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        log_y: bool = False,
        num_trees: int = N_TREES,
        do_bootstrapping: bool = True,
        n_points_per_tree: int = -1,
        ratio_features: float = 5.0 / 6.0,
        min_samples_split: int = 3,
        min_samples_leaf: int = 3,
        max_depth: int = 2**20,
        eps_purity: float = 1e-8,
        max_num_nodes: int = 2**20,
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

        self.log_y = log_y

        self._rf_opts = regression.forest_opts()
        self._rf_opts.num_trees = num_trees
        self._rf_opts.do_bootstrapping = do_bootstrapping
        max_features = 0 if ratio_features > 1.0 else max(1, int(len(self._types) * ratio_features))
        self._rf_opts.tree_opts.max_features = max_features
        self._rf_opts.tree_opts.min_samples_to_split = min_samples_split
        self._rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
        self._rf_opts.tree_opts.max_depth = max_depth
        self._rf_opts.tree_opts.epsilon_purity = eps_purity
        self._rf_opts.tree_opts.max_num_nodes = max_num_nodes
        self._rf_opts.compute_law_of_total_variance = False

        self._n_points_per_tree = n_points_per_tree
        self._rf = None  # type: regression.binary_rss_forest

        # This list well be read out by save_iteration() in the solver
        self._hypers = [
            num_trees,
            max_num_nodes,
            do_bootstrapping,
            n_points_per_tree,
            ratio_features,
            min_samples_split,
            min_samples_leaf,
            max_depth,
            eps_purity,
            self.seed,
        ]

    @property
    def rf_opts(self) -> regression.forest_opts:
        """Random forest hyperparameter"""
        return self._rf_opts

    @property
    def rf(self) -> regression.binary_rss_forest:
        """Random Forest model, only available after training"""
        return self._rf

    @property
    def hypers(self) -> list:
        """List of random forest hyperparameters"""
        return self._hypers


    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def _train(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """
        X = self._impute_inactive(X)
        self.X = X
        self.y = y.flatten()

        if self._n_points_per_tree <= 0:
            self._rf_opts.num_data_points_per_tree = self.X.shape[0]
        else:
            self._rf_opts.num_data_points_per_tree = self._n_points_per_tree
        self.rf = regression.binary_rss_forest()
        self.rf.options = self._rf_opts
        data = self._init_data_container(self.X, self.y)
        self.rf.fit(data, rng=self._rng)

        return self

    def _init_data_container(self, X: np.ndarray, y: np.ndarray) -> regression.default_data_container:
        """Fills a pyrfr default data container, s.t. the forest knows categoricals and bounds for
        continous data.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features]
            Input data points
        y : np.ndarray [n_samples, ]
            Corresponding target values

        Returns
        -------
        data : regression.default_data_container
            The filled data container that pyrfr can interpret
        """
        # retrieve the types and the bounds from the ConfigSpace
        data = regression.default_data_container(X.shape[1])

        for i, (mn, mx) in enumerate(self._bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)
        return data

    def _predict(self, X: np.ndarray, cov_return_type: str | None = "diagonal_cov") -> tuple[np.ndarray, np.ndarray]:
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]
        cov_return_type: Optional[str]
            Specifies what to return along with the mean. Refer ``predict()`` for more information.

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))
        if X.shape[1] != len(self.types):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self.types), X.shape[1]))
        if cov_return_type != "diagonal_cov":
            raise ValueError("'cov_return_type' can only take 'diagonal_cov' for this model")

        X = self._impute_inactive(X)

        if self.log_y:
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
                mean_, var = self.rf.predict_mean_var(row_X)
                means.append(mean_)
                vars_.append(var)

        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))

    def predict_marginalized_over_instances(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Note
        ----
        This method overwrites the same method of ~smac.epm.base_epm.AbstractEPM;
        the following method is random forest specific
        and follows the SMAC2 implementation;
        it requires no distribution assumption
        to marginalize the uncertainty estimates

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if self.instance_features is None or len(self.instance_features) == 0:
            mean_, var = self.predict(X)
            assert var is not None

            var[var < self._var_threshold] = self._var_threshold
            var[np.isnan(var)] = self._var_threshold
            return mean_, var

        if len(X.shape) != 2:
            raise ValueError("Expected 2d array, got %dd array!" % len(X.shape))

        if X.shape[1] != len(self._bounds):
            raise ValueError("Rows in X should have %d entries but have %d!" % (len(self._bounds), X.shape[1]))

        X = self._impute_inactive(X)

        dat_ = np.zeros((X.shape[0], self._rf_opts.num_trees))  # marginalized predictions for each tree
        for i, x in enumerate(X):

            # Marginalize over instances
            # 1. get all leaf values for each tree
            preds_trees: list[list[float]] = [[] for i in range(self._rf_opts.num_trees)]

            for feat in self.instance_features.values():
                x_ = np.concatenate([x, feat])
                preds_per_tree = self.rf.all_leaf_values(x_)
                for tree_id, preds in enumerate(preds_per_tree):
                    preds_trees[tree_id] += preds

            # 2. average in each tree
            if self.log_y:
                for tree_id in range(self._rf_opts.num_trees):
                    dat_[i, tree_id] = np.log(np.exp(np.array(preds_trees[tree_id])).mean())
            else:
                for tree_id in range(self._rf_opts.num_trees):
                    dat_[i, tree_id] = np.array(preds_trees[tree_id]).mean()

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

