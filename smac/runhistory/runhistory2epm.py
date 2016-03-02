import copy
import logging

import numpy

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.configspace import impute_inactive_values
import smac.epm.base_imputor

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class RunType(object):
    '''
       class to define numbers for status types.
       Makes life easier in select_runs
    '''
    SUCCESS = 1
    TIMEOUT = 2
    CENSORED = 3


class RunHistory2EPM(object):
    '''
        takes a runhistory object and preprocess data in order to train EPM
    '''

    def __init__(self, scenario, num_params, success_states=None,
                 impute_censored_data=False, impute_state=None, imputor=None,
                 rs=None):
        '''
        Constructor
        Parameters
        ----------
        num_params : int
            number of parameters in config space
        cutoff_time: positive float
            cutoff time for this scenario, matters only if runs are censored
        success_states: list, optional
            list of states considered as successful (such as StatusType.SUCCESS)
        impute_censored_data: bool, optional
            should we impute data?
        impute_state: list, optional
            list of states that mark censored data (such as StatusType.TIMEOUT)
            in combination with runtime < cutoff_time
        rs : numpy.random.RandomState
            only used for reshuffling data after imputation
        '''
        self.logger = logging.getLogger("runhistory2epm")

        # General arguments
        self.scenario = scenario
        self.rs = rs
        self.num_params = num_params

        # Configuration
        self.success_states = success_states
        self.impute_censored_data = impute_censored_data
        self.impute_state = impute_state
        self.cutoff_time = self.scenario.cutoff
        self.imputor = imputor

        # Fill with some default values
        if rs is None:
            self.rs = numpy.random.RandomState()

        if self.impute_state is None:
            self.impute_state = [StatusType.TIMEOUT, ]

        if self.success_states is None:
            self.success_states = [StatusType.SUCCESS, ]

        # Sanity checks
        # TODO: Decide whether we need this
        if impute_censored_data and scenario.run_obj != "runtime":
            # So far we don't know how to handle censored quality data
            self.logger.critical("Cannot impute censored data when optimizing "
                                 "runtime")
            raise NotImplementedError("Cannot impute censored data when "
                                      "optimizing runtime")

        # Check imputor stuff
        if impute_censored_data and self.imputor is None:
            self.logger.critical("You want me to impute cencored data, but "
                                 "I don't know how. Imputor is None")
            raise ValueError("impute_censored data, but no imputor given")
        elif impute_censored_data and not \
                isinstance(self.imputor, smac.epm.base_imputor.BaseImputor):
            raise ValueError("Given imputor is not an instance of "
                             "smac.epm.base_imputor.BaseImputor, but %s" %
                             type(self.imputor))

    def _build_matrix(self, run_list, runhistory, instances=None):
        # First build nan-matrix of size #configs x #params+1
        n_rows = len(run_list)
        n_cols = self.num_params
        X = numpy.ones([n_rows, n_cols]) * numpy.nan
        Y = numpy.ones([n_rows, 1])

        # Then populate matrix
        for row, (key, run) in enumerate(run_list.items()):
            # Scaling is automatically done in configSpace
            conf = runhistory.ids_config[key.config_id]
            conf = impute_inactive_values(conf)
            X[row, :] = conf.get_array()
            # TODO: replace with instance features if available
            #run_array[row, -1] = instances[row]
            Y[row, 0] = run.cost
        return X, Y.flatten()

    def transform(self, runhistory, shuffle=True):
        '''
        returns vector representation of runhistory

        Parameters
        ----------
        runhistory : list of dicts
                parameter configurations
        shuffle : bool, optional
                shuffle array to mix successful, timeouts and imputed data,
                probably makes everything nondeterministic
        '''
        assert isinstance(runhistory, RunHistory)

        # consider only successfully finished runs
        s_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.data),
                                        select=RunType.SUCCESS)
        # Store a list of instance IDs
        s_instance_id_list = [k.instance_id for k in s_run_list.keys()]
        X, Y = self._build_matrix(run_list=s_run_list, runhistory=runhistory,
                                  instances=s_instance_id_list)

        # Also get TIMEOUT runs
        t_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.data),
                                        select=RunType.TIMEOUT)
        t_instance_id_list = [k.instance_id for k in s_run_list.keys()]

        tX, tY = self._build_matrix(run_list=t_run_list, runhistory=runhistory,
                                    instances=t_instance_id_list)

        if self.impute_censored_data:
            # Get all censored runs
            c_run_list = self.__select_runs(rh_data=copy.deepcopy(runhistory.data),
                                            select=RunType.CENSORED)
            if len(c_run_list) == 0:
                self.logger.critical("No censored data found, skip imputation")
            else:
                # Store a list of instance IDs
                c_instance_id_list = [k.instance_id for k in c_run_list.keys()]

                cen_X, cen_Y = self._build_matrix(run_list=c_run_list,
                                                  runhistory=runhistory,
                                                  instances=c_instance_id_list)

                # Also impute TIMEOUTS
                cen_X = numpy.vstack((cen_X, tX))
                cen_Y = numpy.concatenate((cen_Y, tY))
                self.logger.debug("%d TIMOUTS, %d censored, %d regular" %
                                  (tX.shape[0], cen_X.shape[0], X.shape[0]))

                if shuffle:
                    shuffle_idx = self.rs.permutation(X.shape[0])
                    cen_X = cen_X[shuffle_idx, :]
                    cen_Y = cen_Y[shuffle_idx, :]

                imp_Y = self.imputor.impute(censored_X=cen_X, censored_y=cen_Y,
                                            uncensored_X=X, uncensored_y=Y)

                # Shuffle data to mix censored and imputed data
                X = numpy.vstack((X, cen_X))
                Y = numpy.concatenate((Y, imp_Y))
        else:
            # If we do not impute,we also return TIMEOUT data
            X = numpy.vstack((X, tX))
            Y = numpy.concatenate((Y, tY))

        if shuffle:
            shuffle_idx = self.rs.permutation(X.shape[0])
            X = X[shuffle_idx, :]
            Y = Y[shuffle_idx, :]

        return X, Y

    def __select_runs(self, rh_data, select):
        '''
        select runs of a runhistory

        Parameters
        ----------
        rh_data : runhistory
            dict of ConfigSpace.config

        select : RunType.SUCCESS
            one of "success", "timeout", "censored"
            return only runs for this type
        Returns
        -------
        list of ConfigSpace.config
        '''
        new_dict = dict()

        if select == RunType.SUCCESS:
            for run in rh_data.keys():
                if rh_data[run].status in self.success_states:
                    new_dict[run] = rh_data[run]
        elif select == RunType.TIMEOUT:
            for run in rh_data.keys():
                if (rh_data[run].status == StatusType.TIMEOUT and
                            rh_data[run].time >= self.cutoff_time):
                    new_dict[run] = rh_data[run]
        elif select == RunType.CENSORED:
            for run in rh_data.keys():
                if rh_data[run].status in self.impute_state \
                        and rh_data[run].time < self.cutoff_time:
                    new_dict[run] = rh_data[run]
        else:
            err_msg = "select must be in (%s), but is %s" % \
                      (",".join(["%d" % t for t in
                                [RunType.SUCCESS, RunType.TIMEOUT,
                                 RunType.CENSORED]]), select)
            self.logger.critical(err_msg)
            raise ValueError(err_msg)

        return new_dict
