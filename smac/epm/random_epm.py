'''
Created on Sep 25, 2015

@author: eggenspk
'''
import logging
import numpy

import smac.smbo.run_history
import smac.epm.base_Epm
import smac.configspace

logger = logging.getLogger("random_epm")


class randomEpm(smac.epm.base_Epm.baseEPM):
    """implement an epm, which returns only random values"""

    def __init__(self, rng):
        """
        initialize random number generator
        :param rng: any integer
        """
        numpy.random.seed(rng)

    def fit(self, run_history):
        """
        fit model to run history

        :param run_history: Object that keeps complete run_history
        """
        # (KE) I assume a model can either read run_history or run_history
        # provides a method to transform data to a matrix
        if not isinstance(run_history, smac.smbo.run_history):
            raise NotImplementedError("Can only fit on run_history")
        logger.debug("Fit model to data")

    def update(self, config, value, instance_feature=None):
        """
        Update model (if possible)

        :param config: one configuration (configspace.configuration)
        :param value: one return value (float)
        :param instance_feature: list of features
        """
        if not isinstance(config, smac.configspace.configuration):
            raise NotImplementedError("Can only update with "
                                      "smac.configspace.configuration")
        if not isinstance(value, (float, int)):
            raise NotImplementedError("Accepts only floats as objective value")
        if instance_feature is not None and \
                not isinstance(instance_feature, list):
            raise NotImplementedError("Can only update with instance_feature"
                                      " being a list")

        dummy_run_history = smac.smbo.run_history()
        dummy_run_history.add(config, value, instance_feature)
        # (KE) If a model is updateable, it will read a run_history entry
        # or matrix
        logger.debug("Update model with one sample")

    def predict(self, configs, instance_features=None):
        """Predict values for configs

        :param configs: list of configurations
        :param instance_features: list of instance features
        :return: predictions
        """
        if not isinstance(configs, list):
            raise NotImplementedError("Can only predict for lists of configs")
        # (KE): I assume a model can either read a list of config or configspace
        # provides a method to transform data to matrix
        return numpy.random.rand(len(configs))