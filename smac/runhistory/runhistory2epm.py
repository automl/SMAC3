import copy
from collections import OrderedDict
import logging

import numpy

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.configspace import impute_inactive_values

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

    def __init__(self, num_params, cutoff_time, 
                 instance_features=None, success_states=None,
                 impute_censored_data=False, impute_state=None):
        '''
        Constructor
        Parameters
        ----------
        num_params : int
            number of parameters in config space
        cutoff_time: positive float
            cutoff time for this scenario, matters only if runs are censored
        instance_features: dict
            dictionary mapping instance names to feature vector
        success_states: list, optional
            list of states considered as successful (such as StatusType.SUCCESS)
        impute_censored_data: bool, optional
            should we impute data?
        impute_state: list, optional
            list of states that mark censored data (such as StatusType.TIMEOUT)
            in combination with runtime < cutoff_time
        '''
        if impute_state is None:
            impute_state = [StatusType.TIMEOUT, ]

        if success_states is None:
            success_states = [StatusType.SUCCESS, ]

        self.config = OrderedDict({
            'success_states': success_states,
            'impute_censored_data': impute_censored_data,
            'cutoff_time': cutoff_time,
            'impute_state': impute_state,
        })
        
        self.instance_features = instance_features
        if self.instance_features:
            self.n_feats = len(self.instance_features[self.instance_features.keys()[0]])
        else:
            self.n_feats = 0
        
        self.logger = logging.getLogger("runhistory2epm")
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
        X = numpy.ones([n_rows, n_cols+self.n_feats]) * numpy.nan
        Y = numpy.ones([n_rows, 1])

        # Then populate matrix
        for row, (key, run) in enumerate(run_list.items()):
            # Scaling is automatically done in configSpace
            conf = runhistory.ids_config[key.config_id]
            conf = impute_inactive_values(conf)
            if self.n_feats:
                feats = self.instance_features[key.instance_id]
                X[row, : ] = numpy.hstack((conf.get_array(), feats))
            else:
                X[row, :] = conf.get_array()
                
            #run_array[row, -1] = instance_id_list[row]
            Y[row, 0] = run.cost

        return X, Y

    def __select_runs(self, rh_data, select_censored=False):
        '''
        select runs of a runhistory

        Parameters
        ----------
        rh_data : runhistory
            dict of ConfigSpace.config

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
                if rh_data[run].status in self.config['success_states'] or \
                    (rh_data[run].status == StatusType.TIMEOUT and rh_data[run].time >= self.config['cutoff_time']):
                    # This run was successful or a not censored timeout
                    new_dict[run] = rh_data[run]
        return new_dict
