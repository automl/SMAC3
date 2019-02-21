import logging
import numpy as np

from smac.epm.base_epm import AbstractEPM

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RandomEPM(AbstractEPM):
    """EPM which returns random values on a call to ``fit``."""

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.rng = np.random.RandomState(self.seed)

    def _train(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        """
        Pseudo training on X and Y.

        Parameters
        ----------
        X : np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        Y : np.ndarray (N, 1)
            The corresponding target values.
        """

        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray")
        if not isinstance(Y, np.ndarray):
            raise NotImplementedError("Y has to be of type np.ndarray")

        self.logger.debug("(Pseudo) Fit model to data")

    def _predict(self, X: np.ndarray):
        """
        Predict means and variances for given X.

        Parameters
        ----------
        X : np.ndarray of shape = [n_samples, n_features (config + instance features)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, n_objectives]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, n_objectives]
            Predictive variance
        """
        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray")
        return self.rng.rand(len(X), 1), self.rng.rand(len(X), 1)
