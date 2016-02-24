from abc import ABCMeta, abstractmethod
import logging

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class BaseImputor(metaclass=ABCMeta):
    """abstract Imputor class"""

    def __init__(self):
        """
        initialize imputator module
        """
        pass

    @abstractmethod
    def impute(self, censored_x, censored_y, uncensored_x, uncensored_y):
        """
        impute runs and returns imputed y values

        Parameters
        ----------
        censored_x : array
            Object that keeps complete run_history
        censored_y : list
        uncensored_x : array
        uncensored_y : list
        """