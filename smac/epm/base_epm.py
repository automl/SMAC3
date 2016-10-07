import logging
import numpy as np

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class AbstractEPM(object):
    '''Abstract implementation of the EPM API '''

    def __init__(self, rng):
        '''
        initialize random number generator

        Parameters
        ----------
        rng : np.random.RandomState
        '''
        self.rng = rng

    def train(self, X, Y, **kwargs):
        '''
        Trains the EPM on X and Y.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y: np.ndarray (N, 1)
            The corresponding target values.
        '''
        raise NotImplementedError()

    def predict(self, X):
        '''
        Predict values for configs in X

        Parameters
        ----------
        configs : list
            list of configurations

        instance_features : list
            list of instance features

        Returns
        -------
        predictions
        '''
        raise NotImplementedError()
    
    def predict_marginalized_over_instances(self, X):
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray of shape = [n_features (config), ]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        raise NotImplementedError()