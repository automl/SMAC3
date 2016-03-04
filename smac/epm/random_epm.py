import logging
import numpy

import smac.runhistory.runhistory
import smac.epm.base_epm
import smac.configspace

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RandomEpm(smac.epm.base_epm.baseEPM):
    '''implement an epm, which returns only random values'''

    def __init__(self, rng):
        '''
        initialize random number generator and logger

        Parameters
        ----------
        rng : int
        '''
        self.logger = logging.getLogger("random_epm")
        numpy.random.seed(rng)

    def fit(self, run_history):
        '''
        fit model to run history

        Parameters
        ----------
        run_history : dict
            Object that keeps complete run_history
        '''

        # (KE) I assume a model can either read run_history or run_history
        # provides a method to transform data to a matrix
        if not isinstance(run_history, smac.runhistory.runhistory.RunHistory):
            raise NotImplementedError("Can only fit on run_history")
        self.logger.debug("Fit model to data")

    def update(self, config, value, instance_feature=None):
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

        if not isinstance(config, smac.configspace.configuration):
            raise NotImplementedError("Can only update with "
                                      "smac.configspace.configuration")
        if not isinstance(value, (float, int)):
            raise NotImplementedError("Accepts only floats as objective value")
        if instance_feature is not None and \
                not isinstance(instance_feature, list):
            raise NotImplementedError("Can only update with instance_feature"
                                      " being a list")

        dummy_run_history = smac.runhistory.runhistory.RunHistory()
        dummy_run_history.add(config, value, instance_feature)
        # (KE) If a model is updateable, it will read a run_history entry
        # or matrix
        self.logger.debug("Update model with one sample")

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
        if not isinstance(configs, list):
            raise NotImplementedError("Can only predict for lists of configs")
        # (KE): I assume a model can either read a list of config or configspace
        # provides a method to transform data to matrix
        return numpy.random.rand(len(configs), 1)