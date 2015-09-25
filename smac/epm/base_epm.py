'''
Created on Sep 25, 2015

@author: eggenspk
'''
from abc import ABCMeta, abstractmethod
import logging

import smac.smbo.run_history

logger = logging.getLogger("epm")


class baseEpm(metaclass=ABCMeta):
    """abstract EPM class"""
    def __init__(self):
        """initialize epm module"""

    @abstractmethod
    def fit(self, run_history):
        """
        fit model to run history

        :param run_history: Object that keeps complete run_history
        """

    @abstractmethod
    def update(self, config, value, instance_feature=None):
        """
        Update model (if possible)

        :param config: one configuration (configspace.configuration)
        :param value: one return value (float)
        :param instance_feature: list of features
        """

    @abstractmethod
    def predict(self, configs, instance_features=None):
        """Predict values for configs

        :param configs: list of configurations
        :param instance_features: list of instance features
        :return: predictions
        """
