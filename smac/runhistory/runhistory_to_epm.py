__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"

import logging


class RunHistoryToEPM(object):
    '''
        takes a runhistory object and preprocess data in order to train EPM
    '''

    def __init__(self, config=None):
        '''
        Constructor
        '''
        self.config = dict()
        self.fitted = False
        self.logger = logging.getLogger("epm")
        if config is not None:
            self.config.update(config)

    def fit(self, runhistory):
        '''
        sets all internal information to transform a runhistory to a vector

        Attributes
        ----------
            runhistory : list of dicts
                parameter configurations
        '''
        # learn values to scale params between [0,1] from configspace

        self.fitted = True
        pass

    def transform(self, runhistory):
        '''
        returns vector representation of runhistory

        Attributes
        ----------
        runhistory : list of dicts
                parameter configurations
        '''
        if not self.fitted:
            raise ValueError("runhistory2epm is not yet fitted")
        # consider only finished runs (status == SUCCESS), put them into a list
        # impute nonactive parameters
        # create an array
        # scale x-values to be within [0,1]
        pass

    def fit_transform(self, runhistory):
        '''
        calls fit and transform, returns vector representation of runhistory

        Attributes
        ----------
        runhistory : list of dicts
                parameter configurations
        '''
        self.fit(runhistory)
        return self.transform(runhistory)