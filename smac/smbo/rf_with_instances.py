import numpy as np
import logging

import pyrfr.regression

from smac.utils.duplicate_filter_logging import DuplicateFilter

__author__ = "Aaron Klein"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
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
                 ratio_features=1,
                 min_samples_split=2,
                 min_samples_leaf=2,
                 max_depth=20,
                 eps_purity=1e-8,
                 max_num_nodes=1000,
                 seed=42):

        self.instance_features = instance_features
        # make sure types are uint
        if instance_features is not None:
            types = np.hstack((types, np.zeros((instance_features.shape[1]))))
        self.types = np.array(types, dtype=np.uint)

        self.rf = pyrfr.regression.binary_rss()
        self.rf.num_trees = num_trees
        self.rf.seed = seed
        self.rf.do_bootstrapping = do_bootstrapping
        self.rf.num_data_points_per_tree = n_points_per_tree
        self.rf.max_features = int(types.shape[0] * ratio_features)
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

    def predict(self, Xtest, **kwargs):
        """
        Returns the predictive mean and variance marginalised over all
        instances for a single test point.

        Parameters
        ----------
        Xtest: np.ndarray (1, D)
            Input test point

        Returns
        ----------
        np.array(1,)
            predictive mean over all instances
        np.array(1,)
            predictive variance over all instances
        """
        # first we make sure this does not break in cases
        # where we have no instance features
        if self.instance_features is None or len(self.instance_features) == 0:
            X_ = Xtest
        else:
            nfeats = self.instance_features.shape[0]
            # TODO: Use random forest data container for instances
            X_ = np.hstack(
                (np.tile(Xtest, (nfeats, 1)), self.instance_features))

        mean = np.zeros(X_.shape[0])
        var = np.zeros(X_.shape[0])

        # TODO: Would be nice if the random forest supports batch predictions
        for i, x in enumerate(X_):
            mean[i], var[i] = self.rf.predict(x)

        mean = np.mean(mean)
        var = np.sum(var)  # TODO: is this correct?

        return mean, var
