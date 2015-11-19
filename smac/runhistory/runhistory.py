__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"

import logging
import collections


class RunHistory(object):
    '''
         saves all run informations from target algorithm runs

        Attributes
        ----------
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.data = []

        self.Data = collections.namedtuple(
            'Data', ['config', 'instance_id', 'seed', 'cost',
                     'time', 'status', 'additional_info'])

    def add(self, config, cost, time,
            status, instance_id=None,
            seed=None,
            additional_info=None):
        '''
        adds a data of a new target algorithm (TA) run 

        Attributes
        ----------
            config : dict
                parameter configuratoin
            cost: float
                cost of TA run
            time: float
                runtime of TA run
            status: str
                status in {SUCCESS, TIMEOUT, CRASHED, ABORT, MEMOUT}
            instance_id: int
                id of instance (default: None)
            seed: int
                random seed used by TA (default: None)
            additional_info: dict
                additional run infos (could include further returned 
                information from TA or fields such as starttime and host_id)
        '''

        d = self.Data(config, instance_id, seed, cost, time, status, additional_info)
        self.data.append(d)
