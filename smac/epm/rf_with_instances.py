import numpy as np
import logging

from pyrfr import regression

from smac.configspace import CategoricalHyperparameter
from smac.epm.base_epm import AbstractEPM


__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RandomForestWithInstances(AbstractEPM):

    """Interface to the random forest that takes instance features
    into account.

    Attributes
    ----------
    rf_opts :
        Random forest hyperparameter
    n_points_per_tree : int
    rf : regression.binary_rss_forest
        Only available after training
    hypers: list
        List of random forest hyperparameters
    seed : int
    types : list
    bounds : list
    rng : np.random.RandomState
    logger : logging.logger
    """

    def __init__(self, types: np.ndarray,
                 bounds: np.ndarray,
                 num_trees: int=10,
                 do_bootstrapping: bool=True,
                 n_points_per_tree: int=-1,
                 ratio_features: float=5. / 6.,
                 min_samples_split: int=3,
                 min_samples_leaf: int=3,
                 max_depth: int=20,
                 eps_purity: int=1e-8,
                 max_num_nodes: int=2**20,
                 seed: int=42,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        types : np.ndarray (D)
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds : np.ndarray (D, 2)
            Specifies the bounds for continuous features.
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
            The maxmimum total number of nodes in a tree
        seed : int
            The seed that is passed to the random_forest_run library.
        """
        super().__init__(**kwargs)

        self.types = types
        self.bounds = bounds
        self.rng = regression.default_random_engine(seed)

        self.rf_opts = regression.forest_opts()
        self.rf_opts.num_trees = num_trees
        self.rf_opts.do_bootstrapping = do_bootstrapping
        max_features = 0 if ratio_features > 1.0 else \
            max(1, int(types.shape[0] * ratio_features))
        self.rf_opts.tree_opts.max_features = max_features
        self.rf_opts.tree_opts.min_samples_to_split = min_samples_split
        self.rf_opts.tree_opts.min_samples_in_leaf = min_samples_leaf
        self.rf_opts.tree_opts.max_depth = max_depth
        self.rf_opts.tree_opts.epsilon_purity = eps_purity
        self.rf_opts.tree_opts.max_num_nodes = max_num_nodes

        self.n_points_per_tree = n_points_per_tree
        self.rf = None  # type: regression.binary_rss_forest

        # This list well be read out by save_iteration() in the solver
        self.hypers = [num_trees, max_num_nodes, do_bootstrapping,
                       n_points_per_tree, ratio_features, min_samples_split,
                       min_samples_leaf, max_depth, eps_purity, seed]
        self.seed = seed

        self.logger = logging.getLogger(self.__module__ + "." +
                                        self.__class__.__name__)

    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Trains the random forest on X and y.

        Parameters
        ----------
        X : np.ndarray [n_samples, n_features (config + instance features)]
            Input data points.
        Y : np.ndarray [n_samples, ]
            The corresponding target values.

        Returns
        -------
        self
        """

        self.X = X
        self.y = y.flatten()

        if self.n_points_per_tree <= 0:
            self.rf_opts.num_data_points_per_tree = self.X.shape[0]
        else:
            self.rf_opts.num_data_points_per_tree = self.n_points_per_tree
        self.rf = regression.binary_rss_forest()
        self.rf.options = self.rf_opts
        data = self.__init_data_container(self.X, self.y)
        self.rf.fit(data, rng=self.rng)
        return self

    def __init_data_container(self, X: np.ndarray, y: np.ndarray):
        """Fills a pyrfr default data container, s.t. the forest knows
        categoricals and bounds for continous data
        
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

        for i, (mn, mx) in enumerate(self.bounds):
            if np.isnan(mx):
                data.set_type_of_feature(i, mn)
            else:
                data.set_bounds_of_feature(i, mn, mx)

        for row_X, row_y in zip(X, y):
            data.add_data_point(row_X, row_y)
        return data

    def _predict(self, X: np.ndarray):
        """Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples,
                                   n_features (config + instance features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError(
                'Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != self.types.shape[0]:
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (self.types.shape[0], X.shape[1]))

        means, vars_ = [], []
        for row_X in X:
            mean, var = self.rf.predict_mean_var(row_X)
            means.append(mean)
            vars_.append(var)
        means = np.array(means)
        vars_ = np.array(vars_)

        return means.reshape((-1, 1)), vars_.reshape((-1, 1))
