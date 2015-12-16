import copy
from collections import OrderedDict
import logging

import numpy

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory

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

    def __init__(self, num_params, config=None):
        '''
        Constructor
        Parameters
        ----------
        num_params : int
            number of parameters in config space
        config: dict
            configuration for conversion
        '''
        self.config = OrderedDict({
            'success_states': [StatusType.SUCCESS, ],
            'impute_censored_data': False,
            'cutoff_time': -1,
            'impute_state': [StatusType.TIMEOUT],
                                   })
        self.logger = logging.getLogger("runhistory2epm")
        if config is not None:
            self.config.update(config)
        self.num_params = num_params

    def transform(self, runhistory):
        '''
        returns vector representation of runhistory

        Parameters
        ----------
        runhistory : list of dicts
                parameter configurations
        '''
        assert isinstance(runhistory, RunHistory)

        # consider only successfully finished runs, put them in a list
        run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.data),
                                      select_censored=False)
        # Store a list of instance IDs
        instance_id_list = [k.instance_id for k in run_list.keys()]

        run_list = [(run_list[r].time, r.config_id) for r in run_list]
        
        if self.config['impute_censored_data']:
            cens_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.data),
                                           select_censored=True)
            # Store a list of instance IDs
            cens_instance_id_list = [k.instance_id for k in cens_list.keys()]

            cens_list = [c.config for c in cens_list.keys()]
            raise NotImplementedError("Imputation of right censored data is not"
                                      " yet possible")


        # First build nan-matrix of size #configs x #params+1
        n_rows = len(run_list)
        n_cols = self.num_params
        X = numpy.ones([n_rows, n_cols]) * numpy.nan
        y = numpy.ones([n_rows])

        # Then populate matrix
        for row, run in enumerate(run_list):
            # Scaling is automatically done in configSpace
            X[row, :] = runhistory.ids_config[run[1]].get_array()
            #TODO: replace with instance features if available
            #run_array[row, -1] = instance_id_list[row]
            #TODO: replace by cost if we optimize quality/cost
            y[row] = run[0]
            
        return X, y 

    def __select_runs(self, rh_data, select_censored=False):
        '''
        select runs of a runhistory

        Parameters
        ----------
        rh_data : runhistory
            list of ConfigSpace.config

        select_censored : bool, optional
            return censored runs
            if True return runs with status in self.config['impute_state'] and
            runtime < self.config['cutoff_time']
            if False return with with status in self.config['success_states']

        Returns
        -------
        list of ConfigSpace.config
        '''
        new_dict = dict()
        if select_censored:
            for run in rh_data.keys():
                if rh_data[run].status in self.config['impute_state'] and \
                                rh_data[run].time < self.config['cutoff_time']:
                    # This run was censored
                    new_dict[run] = rh_data[run]
        else:
            for run in rh_data.keys():
                if rh_data[run].status in self.config['success_states']:
                    # This run was successful
                    new_dict[run] = rh_data[run]
        return new_dict