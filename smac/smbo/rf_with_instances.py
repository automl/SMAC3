'''
Created on Dec 16, 2015

@author: Aaron Klein
'''

import numpy as np

from robo.models.random_forest import RandomForest


class RandomForestWithInstances(RandomForest):
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
                 ratio_features=0.5,
                 min_samples_split=1,
                 min_samples_leaf=1,
                 max_depth=20,
                 eps_purity=1e-8,
                 max_num_nodes=1000,
                 seed=42):

        self.instance_features = instance_features
        super(RandomForestWithInstances, self).__init__(types,
                                                        num_trees,
                                                        do_bootstrapping,
                                                        n_points_per_tree,
                                                        ratio_features,
                                                        min_samples_split,
                                                        min_samples_leaf,
                                                        max_depth,
                                                        eps_purity,
                                                        max_num_nodes,
                                                        seed)

    def predict(self, Xtest, **kwargs):
        """
        Returns the predictive mean and variance marginalised over all
        instances.

        Parameters
        ----------
        Xtest: np.ndarray (N, D)
            Input test points

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
            X_ = np.repeat(Xtest, self.instance_features.shape[0], axis=0)
            I_ = np.tile(self.instance_features, (Xtest.shape[0], 1))

            X_ = np.concatenate((X_, I_), axis=1)

        mu, var = super(RandomForestWithInstances, self).predict(X_)

        return mu.mean(), var.mean()
