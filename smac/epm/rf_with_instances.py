import numpy as np
import logging

import pyrfr.regression

from smac.utils.duplicate_filter_logging import DuplicateFilter

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RandomForestWithInstances(object):

    '''
    Interface to the random forest that takes instance features
    into account.

    Parameters
    ----------
    types: np.ndarray (D)
        Specifies the number of categorical values of an input dimension. Where
        the i-th entry corresponds to the i-th input dimension. Let say we have
        2 dimension where the first dimension consists of 3 different
        categorical choices and the second dimension is continuous than we
        have to pass np.array([2, 0]). Note that we count starting from 0.
    instance_features: np.ndarray (I, K)
        Contains the K dimensional instance features
        of the I different instances
    num_trees: int
        The number of trees in the random forest.
    do_bootstrapping: bool
        Turns on / off bootstrapping in the random forest.
    ratio_features: float
        The ratio of features that are considered for splitting.
    min_samples_split: int
        The minimum number of data points to perform a split.
    min_samples_leaf: int
        The minimum number of data points in a leaf.
    max_depth: int

    eps_purity: float

    max_num_nodes: int

    seed: int
        The seed that is passed to the random_forest_run library.
    '''

    def __init__(self, types,
                 instance_features,
                 num_trees=30,
                 do_bootstrapping=True,
                 n_points_per_tree=0,
                 ratio_features=5./6.,
                 min_samples_split=3,
                 min_samples_leaf=3,
                 max_depth=20,
                 eps_purity=1e-8,
                 max_num_nodes=1000,
                 seed=42):

        self.instance_features = instance_features
        self.types = types

        self.rf = pyrfr.regression.binary_rss()
        self.rf.num_trees = num_trees
        self.rf.seed = seed
        self.rf.do_bootstrapping = do_bootstrapping
        self.rf.num_data_points_per_tree = n_points_per_tree
        max_features = 0 if ratio_features >= 1.0 else \
                       max(1, int(types.shape[0] * ratio_features))
        self.rf.max_features = max_features
        self.rf.min_samples_to_split = min_samples_split
        self.rf.min_samples_in_leaf = min_samples_leaf
        self.rf.max_depth = max_depth
        self.rf.epsilon_purity = eps_purity
        self.rf.max_num_nodes = max_num_nodes

        # This list well be read out by save_iteration() in the solver
        self.hypers = [num_trees, max_num_nodes, do_bootstrapping,
                       n_points_per_tree, ratio_features, min_samples_split,
                       min_samples_leaf, max_depth, eps_purity, seed]
        self.seed = seed

        self.logger = logging.getLogger("RF")
        # TODO: check this -- it could slow us down
        dub_filter = DuplicateFilter()
        self.logger.addFilter(dub_filter)

        # Never use a lower variance than this
        self.var_threshold = 10 ** -5

    def train(self, X, Y, **kwargs):
        '''
        Trains the random forest on X and Y.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, 1)
            The corresponding target values.
        '''

        self.X = X
        self.Y = Y

        data = pyrfr.regression.numpy_data_container(self.X,
                                                     self.Y[:, 0],
                                                     self.types)

        self.rf.fit(data)

    def predict(self, X):
        """Predict mean and variance for given X.

        Returns the predictive mean and variance marginalised over all
        instances for a single test point. Wraps the pyrfr predict method
        which only handles x (1, D) instead of X (N, D).

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        mean: np.ndarray
            Predictive mean
        var: np.ndarray
            Predictive variance
        """
        if len(X.shape) != 2:
            raise ValueError('Input to random forest must be of shape (N, D).')

        if X.shape[0] > 1:
            if self.instance_features is None or \
                    len(self.instance_features) == 0:
                pred = np.array([self.rf.predict(x) for x in X])
                mean = pred[:, 0]
                var = pred[:, 1]
            else:
                nfeats = self.instance_features.shape[0]
                mean = np.zeros(X.shape[0])
                var = np.zeros(X.shape[0])
                for i, x in enumerate(X):
                    instance_mean = np.zeros(nfeats)
                    instance_var = np.ones(nfeats)
                    x_ = np.hstack(
                        (np.tile(x, (nfeats, 1)), self.instance_features))
                    for j, x__ in enumerate(x_):
                        instance_mean[i], instance_var[i] = self.rf.predict(x__)
                    var[i] = np.mean(instance_var) + np.var(instance_mean)
                    mean[i] = np.mean(instance_mean)

            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold

            mean = np.array(mean)
            var = np.array(var)
        else:
            mean, var = self.rf.predict(X.flatten())
            if var < self.var_threshold:
                self.logger.debug(
                    "Variance is small, capping to 10^-5")
                var = self.var_threshold
            var = np.array([var])
            mean = np.array([mean, ])

        if len(mean.shape) != 2:
            mean = mean.reshape((-1, 1))
        if len(var.shape) != 2:
            var = var.reshape((-1, 1))

        return mean, var

