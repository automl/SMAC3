from abc import ABCMeta, abstractmethod
import logging

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class BaseImputor(object):
    """abstract Imputor class"""

    def __init__(self):
        """
        initialize imputator module
        """
        pass

    @abstractmethod
    def impute(self, censored_x, censored_y, uncensored_x, uncensored_y):
        """
        impute censored runs and returns new y values

        Parameters
        ----------
        censored_x : np.array [N, M]
            feature array of all runs that are censored
        censored_y : np.array [N, 1]
            array of target values for all runs that are censored
        uncensored_x : np.array [N, M]
            feature array of all runs that are not censored
        uncensored_y : np.array [N, 1]
            array of target values for all runs that are not censored

        Returns
        ----------
        imputed_y: np.array
            same shape as censored_y [N, 1]
        """