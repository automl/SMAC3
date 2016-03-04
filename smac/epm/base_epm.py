from abc import ABCMeta, abstractmethod
import logging

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class BaseEpm(ABCMeta):
    '''abstract EPM class'''
    def __init__(self):
        '''initialize epm module'''
        self.logger = logging.getLogger("epm")


    @abstractmethod
    def fit(self, run_history):
        '''
        fit model to run history

        Parameters
        ----------
        run_history : dict
            Object that keeps complete run_history
        '''


    @abstractmethod
    def update(self, config, value, instance_features=None):
        '''
        Update model (if possible)

        Parameters
        ----------
        config : configSpace.config
            configuration
        value : float
            costs for config
        instance_features : list
            list of instance features
        '''


    @abstractmethod
    def predict(self, configs, instance_features=None):
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

