from abc import abstractmethod

import numpy as np


__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class BaseImputor(object):
    """Abstract implementation of the Imputation API."""

    def __init__(self):
        pass

    @abstractmethod
    def impute(self, censored_X: np.ndarray, censored_y: np.ndarray,
               uncensored_X: np.ndarray, uncensored_y: np.ndarray):
        """
        Imputes censored runs and returns new y values.

        Parameters
        ----------
        censored_X : np.ndarray [N, M]
            Feature array of all censored runs.
        censored_y : np.ndarray [N, 1]
            Target values for all runs censored runs.
        uncensored_X : np.ndarray [N, M]
            Feature array of all non-censored runs.
        uncensored_y : np.ndarray [N, 1]
            Target values for all non-censored runs.

        Returns
        ----------
        imputed_y: np.ndarray
            Same shape as censored_y [N, 1]
        """
