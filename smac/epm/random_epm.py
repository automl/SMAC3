import logging
import numpy as np

import smac.runhistory.runhistory
import smac.configspace

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RandomEpm(object):
    '''implement an epm, which returns only random values'''

    def __init__(self, rng):
        '''
        initialize random number generator and logger

        Parameters
        ----------
        rng : np.random.RandomState
        '''
        self.logger = logging.getLogger("RandomEpm")
        self.rng = rng

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

        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray")
        if not isinstance(Y, np.ndarray):
            raise NotImplementedError("Y has to be of type np.ndarray")

        self.logger.debug("(Pseudo) Fit model to data")

    def predict(self, X):
        '''
        Predict values for configs

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
        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray")
        return self.rng.rand(len(X), 1), self.rng.rand(len(X), 1)
    
    def _predict(self, x):
        """Predict mean and variance for given x.

        Parameters
        ----------
        x : np.ndarray of shape = [n_features (config + instance features), ]

        Returns
        -------
        mean : float
            Predictive mean
        var : float
            Predictive variance
        """
        mean, var = self.rf.predict(x)

        return self.rng.rand(), self.rng.rand()

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

        return self.rng.rand(len(X), 1), self.rng.rand(len(X), 1)