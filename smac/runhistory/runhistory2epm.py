import copy
from collections import OrderedDict
import logging

import numpy

from smac.runhistory.runhistory import SUCCESS, TIMEOUT, CRASHED, ABORT, MEMOUT

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RunHistory2EPM(object):
    '''
        takes a runhistory object and preprocess data in order to train EPM
    '''

    def __init__(self, config=None):
        '''
        Constructor
        '''
        self.config = OrderedDict({
            'success_states': [SUCCESS, ],
            'scale': [0, 1],
                                   })
        self.logger = logging.getLogger("runhistory2epm")
        if config is not None:
            self.config.update(config)

    def transform(self, runhistory, inplace=False):
        '''
        returns vector representation of runhistory

        Attributes
        ----------
        runhistory : list of dicts
                parameter configurations
        '''

        # consider only successfully finished runs, put them in a list
        if inplace:
            run_list = self.__selectRuns(copy.deepcopy(runhistory)).values()
        else:
            run_list = self.__selectRuns(runhistory).values()

        # Store a list of instance IDs
        instance_id_list = [r['instance_id'] for r in run_list]

        # transform to configspace-configs & impute nonactive
        run_list = [configSpace.dict2config(r['config']).imputeNonActive()
                    for r in run_list]

        # Scale within [0,1]
        run_list = [r.scale(lower=self.config['scale'][0],
                            upper=self.config['scale'][0]) for r in run_list]

        # transform to array
        run_list = numpy.array([r.toArray() for r in run_list])

        return run_list

    def __selectRuns(self, runhistory):
        for run in runhistory.keys():
            if run.status not in self.config['success_states']:
                del runhistory[run]