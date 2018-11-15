import abc
from collections import OrderedDict
import logging
import typing

import numpy as np

from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory, RunKey, RunValue
from smac.configspace import convert_configurations_to_array
from smac.epm.base_imputor import BaseImputor
from smac.utils import constants
from smac.scenario.scenario import Scenario

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class AbstractRunHistory2EPM(object):
    __metaclass__ = abc.ABCMeta

    """Abstract class for preprocessing data in order to train an EPM.

    Attributes
    ----------
    logger
    scenario
    rng
    num_params

    success_states
    impute_censored_data
    impute_state
    cutoff_time
    imputor
    instance_features
    n_feats
    num_params
    """

    def __init__(
        self,
        scenario: Scenario,
        num_params: int,
        success_states: typing.Optional[typing.List[StatusType]]=None,
        impute_censored_data: bool=False,
        impute_state: typing.Optional[typing.List[StatusType]]=None,
        imputor: typing.Optional[BaseImputor]=None,
        rng: typing.Optional[np.random.RandomState]=None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        scenario: Scenario Object
            Algorithm Configuration Scenario
        num_params : int
            number of parameters in config space
        success_states: list, optional
            List of states considered as successful (such as StatusType.SUCCESS)
            If None, set to [StatusType.SUCCESS, ]
        impute_censored_data: bool, optional
            Should we impute data?
        imputor: epm.base_imputor Instance
            Object to impute censored data
        impute_state: list, optional
            List of states that mark censored data (such as StatusType.TIMEOUT)
            in combination with runtime < cutoff_time
            If None, set to [StatusType.CAPPED, ]
        rng : numpy.random.RandomState
            only used for reshuffling data after imputation
        """

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # General arguments
        self.scenario = scenario
        self.rng = rng
        self.num_params = num_params

        # Configuration
        self.success_states = success_states
        self.impute_censored_data = impute_censored_data
        self.impute_state = impute_state
        self.cutoff_time = self.scenario.cutoff
        self.imputor = imputor

        # Fill with some default values
        if rng is None:
            self.rng = np.random.RandomState(seed=1)

        if self.impute_state is None:
            self.impute_state = [StatusType.CAPPED, ]

        if self.success_states is None:
            self.success_states = [StatusType.SUCCESS, ]

        self.instance_features = scenario.feature_dict
        self.n_feats = scenario.n_features

        self.num_params = num_params

        # Sanity checks
        # TODO: Decide whether we need this
        if impute_censored_data and scenario.run_obj != "runtime":
            # So far we don't know how to handle censored quality data
            self.logger.critical("Cannot impute censored data when not "
                                 "optimizing runtime")
            raise NotImplementedError("Cannot impute censored data when not "
                                      "optimizing runtime")

        # Check imputor stuff
        if impute_censored_data and self.imputor is None:
            self.logger.critical("You want me to impute cencored data, but "
                                 "I don't know how. Imputor is None")
            raise ValueError("impute_censored data, but no imputor given")
        elif impute_censored_data and not \
                isinstance(self.imputor, BaseImputor):
            raise ValueError("Given imputor is not an instance of "
                             "smac.epm.base_imputor.BaseImputor, but %s" %
                             type(self.imputor))

    @abc.abstractmethod
    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory,
                      instances: list=None,
                      par_factor: int=1):
        """Builds x,y matrixes from selected runs from runhistory

        Parameters
        ----------
        run_dict: dict(RunKey -> RunValue)
            dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            runhistory object
        instances: list
            list of instances
        par_factor: int
            penalization factor for censored runtime data

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """
        raise NotImplementedError()

    def transform(self, runhistory: RunHistory):
        """Returns vector representation of runhistory; if imputation is
        disabled, censored (TIMEOUT with time < cutoff) will be skipped

        Parameters
        ----------
        runhistory : smac.runhistory.runhistory.RunHistory
            Runhistory containing all evaluated configurations/instances

        Returns
        -------
        X: numpy.ndarray
            configuration vector x instance features
        Y: numpy.ndarray
            cost values
        """
        self.logger.debug("Transform runhistory into X,y format")

        # consider only successfully finished runs
        s_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                      if runhistory.data[run].status in self.success_states}

        # Store a list of instance IDs
        s_instance_id_list = [k.instance_id for k in s_run_dict.keys()]
        X, Y = self._build_matrix(run_dict=s_run_dict, runhistory=runhistory,
                                  instances=s_instance_id_list)

        # Also get TIMEOUT runs
        t_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                      if runhistory.data[run].status == StatusType.TIMEOUT and
                      runhistory.data[run].time >= self.cutoff_time}
        t_instance_id_list = [k.instance_id for k in s_run_dict.keys()]

        # use penalization (e.g. PAR10) for EPM training
        tX, tY = self._build_matrix(run_dict=t_run_dict, runhistory=runhistory,
                                    instances=t_instance_id_list,
                                    par_factor=self.scenario.par_factor)

        # if we don't have successful runs,
        # we have to return all timeout runs
        if not s_run_dict:
            return tX, tY

        if self.impute_censored_data:
            # Get all censored runs
            c_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                          if runhistory.data[run].status in self.impute_state and
                          runhistory.data[run].time < self.cutoff_time}
            if len(c_run_dict) == 0:
                self.logger.debug("No censored data found, skip imputation")
                # If we do not impute, we also return TIMEOUT data
                X = np.vstack((X, tX))
                Y = np.concatenate((Y, tY))
            else:
                # Store a list of instance IDs
                c_instance_id_list = [k.instance_id for k in c_run_dict.keys()]

                # better empirical results by using PAR1 instead of PAR10
                # for censored data imputation
                cen_X, cen_Y = self._build_matrix(run_dict=c_run_dict,
                                                  runhistory=runhistory,
                                                  instances=c_instance_id_list,
                                                  par_factor=1)

                # Also impute TIMEOUTS
                tX, tY = self._build_matrix(run_dict=t_run_dict,
                                            runhistory=runhistory,
                                            instances=t_instance_id_list,
                                            par_factor=1)
                cen_X = np.vstack((cen_X, tX))
                cen_Y = np.concatenate((cen_Y, tY))
                self.logger.debug("%d TIMOUTS, %d censored, %d regular" %
                                  (tX.shape[0], cen_X.shape[0], X.shape[0]))

                # return imp_Y in PAR depending on the used threshold in
                # imputor
                imp_Y = self.imputor.impute(censored_X=cen_X, censored_y=cen_Y,
                                            uncensored_X=X, uncensored_y=Y)

                # Shuffle data to mix censored and imputed data
                X = np.vstack((X, cen_X))
                Y = np.concatenate((Y, imp_Y))
        else:
            # If we do not impute, we also return TIMEOUT data
            X = np.vstack((X, tX))
            Y = np.concatenate((Y, tY))

        self.logger.debug("Converted %d observations" % (X.shape[0]))
        return X, Y

    def get_X_y(self, runhistory: RunHistory):
        """Simple interface to obtain all data in runhistory in X, y format

        Parameters
        ----------
        runhistory : smac.runhistory.runhistory.RunHistory
            runhistory of all evaluated configurations x instances

        Returns
        -------
        X: numpy.ndarray
            matrix of all configurations (+ instance features)
        y: numpy.ndarray
            vector of cost values; can include censored runs
        cen: numpy.ndarray
            vector of bools indicating whether the y-value is censored
        """
        X = []
        y = []
        cen = []
        feature_dict = self.scenario.feature_dict
        params = self.scenario.cs.get_hyperparameters()
        for k, v in runhistory.data.items():
            config = runhistory.ids_config[k.config_id]
            x = [config.get(p.name) for p in params]
            features = feature_dict.get(k.instance_id)
            if features:
                x.extend(features)
            X.append(x)
            y.append(v.cost)
            cen.append(v.status != StatusType.SUCCESS)
        return np.array(X), np.array(y), np.array(cen)


class RunHistory2EPM4Cost(AbstractRunHistory2EPM):
    """TODO"""

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory, instances: typing.List[str]=None,
                      par_factor: int=1):
        """"Builds X,y matrixes from selected runs from runhistory

        Parameters
        ----------
        run_dict: dict: RunKey -> RunValue
            dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            runhistory object
        instances: list
            list of instances
        par_factor: int
            penalization factor for censored runtime data

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """

        # First build nan-matrix of size #configs x #params+1
        n_rows = len(run_dict)
        n_cols = self.num_params
        X = np.ones([n_rows, n_cols + self.n_feats]) * np.nan
        y = np.ones([n_rows, 1])

        # Then populate matrix
        for row, (key, run) in enumerate(run_dict.items()):
            # Scaling is automatically done in configSpace
            conf = runhistory.ids_config[key.config_id]
            conf_vector = convert_configurations_to_array([conf])[0]
            if self.n_feats:
                feats = self.instance_features[key.instance_id]
                X[row, :] = np.hstack((conf_vector, feats))
            else:
                X[row, :] = conf_vector
            # run_array[row, -1] = instances[row]
            if self.scenario.run_obj == "runtime":
                if run.status != StatusType.SUCCESS:
                    y[row, 0] = run.time * par_factor
                else:
                    y[row, 0] = run.time
            else:
                y[row, 0] = run.cost

        return X, y


class RunHistory2EPM4LogCost(RunHistory2EPM4Cost):
    """TODO"""

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory, instances: typing.List[str]=None,
                      par_factor: int=1):
        """Builds X,y matrices from selected runs from runhistory; transforms
         y by using log

        Parameters
        ----------
        run_dict: dict(RunKey -> RunValue)
            Dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            Runhistory object
        instances: list
            List of instances
        par_factor: int
            Penalization factor for censored runtime data

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """
        X, y = super()._build_matrix(run_dict=run_dict, runhistory=runhistory,
                                     instances=instances, par_factor=par_factor)

        # ensure that minimal value is larger than 0
        if np.any(y <= 0):
            self.logger.warning(
                "Got cost of smaller/equal to 0. Replace by %f since we use"
                " log cost." % (constants.MINIMAL_COST_FOR_LOG))
            y[y < constants.MINIMAL_COST_FOR_LOG] =\
                constants.MINIMAL_COST_FOR_LOG
        y = np.log(y)

        return X, y

class RunHistory2EPM4ScaledCost(RunHistory2EPM4Cost):
    """TODO"""

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory, instances: typing.List[str]=None,
                      par_factor: int=1):
        """Builds X,y matrices from selected runs from runhistory; transforms
         y by linearly scaling 

        Parameters
        ----------
        run_dict: dict(RunKey -> RunValue)
            Dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            Runhistory object
        instances: list
            List of instances
        par_factor: int
            Penalization factor for censored runtime data

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """
        X, y = super()._build_matrix(run_dict=run_dict, runhistory=runhistory,
                                     instances=instances, par_factor=par_factor)

        if y.size > 0:
            perc = np.percentile(y, 5)
            min_y = 2 * np.min(y) - perc # ensure that scaled y cannot be 0
            max_y = np.max(y)
            # linear scaling
            if min_y == max_y:
                # prevent diving by zero
                min_y *= 1 - 10**-101
            y = (y - min_y) / (max_y - min_y)

        return X, y

class RunHistory2EPM4InvScaledCost(RunHistory2EPM4Cost):
    """TODO"""

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory, instances: typing.List[str]=None,
                      par_factor: int=1):
        """Builds X,y matrices from selected runs from runhistory; transforms
         y by linearly scaling and using inverse

        Parameters
        ----------
        run_dict: dict(RunKey -> RunValue)
            Dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            Runhistory object
        instances: list
            List of instances
        par_factor: int
            Penalization factor for censored runtime data

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """
        X, y = super()._build_matrix(run_dict=run_dict, runhistory=runhistory,
                                     instances=instances, par_factor=par_factor)

        if y.size > 0:
            perc = np.percentile(y, 5)
            min_y = 2 * np.min(y) - perc # ensure that scaled y cannot be 0
            max_y = np.max(y)
            # linear scaling
            if min_y == max_y:
                # prevent diving by zero
                min_y *= 1 - 10**-10
            y = (y - min_y) / (max_y - min_y)
            y = 1 - 1/y

        return X, y
    
class RunHistory2EPM4SqrtScaledCost(RunHistory2EPM4Cost):
    """TODO"""

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory, instances: typing.List[str]=None,
                      par_factor: int=1):
        """Builds X,y matrices from selected runs from runhistory; transforms
         y by linearly scaling and using sqrt

        Parameters
        ----------
        run_dict: dict(RunKey -> RunValue)
            Dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            Runhistory object
        instances: list
            List of instances
        par_factor: int
            Penalization factor for censored runtime data

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """
        X, y = super()._build_matrix(run_dict=run_dict, runhistory=runhistory,
                                     instances=instances, par_factor=par_factor)

        if y.size > 0:
            perc = np.percentile(y, 5)
            min_y = 2 * np.min(y) - perc # ensure that scaled y cannot be 0
            max_y = np.max(y)
            # linear scaling
            if min_y == max_y:
                # prevent diving by zero
                min_y *= 1 - 10**-10
            y = (y - min_y) / (max_y - min_y)
            y = np.sqrt(y)

        return X, y

class RunHistory2EPM4LogScaledCost(RunHistory2EPM4Cost):
    """TODO"""

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory, instances: typing.List[str]=None,
                      par_factor: int=1):
        """Builds X,y matrices from selected runs from runhistory; transforms
         y by linearly scaling and using log

        Parameters
        ----------
        run_dict: dict(RunKey -> RunValue)
            Dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            Runhistory object
        instances: list
            List of instances
        par_factor: int
            Penalization factor for censored runtime data

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """
        X, y = super()._build_matrix(run_dict=run_dict, runhistory=runhistory,
                                     instances=instances, par_factor=par_factor)

        if y.size > 0:
            perc = np.percentile(y, 5) 
            min_y = 2 * np.min(y) - perc # ensure that scaled y cannot be 0
            max_y = np.max(y)
            # linear scaling
            if min_y == max_y:
                # prevent diving by zero
                min_y *= 1 - 10**-10
            y = (y - min_y) / (max_y - min_y)
            y = np.log(y)

        return X, y


class RunHistory2EPM4EIPS(AbstractRunHistory2EPM):
    """TODO"""

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory, instances: typing.List[str]=None,
                      par_factor: int=1):
        """TODO"""
        # First build nan-matrix of size #configs x #params+1
        n_rows = len(run_dict)
        n_cols = self.num_params
        X = np.ones([n_rows, n_cols + self.n_feats]) * np.nan
        Y = np.ones([n_rows, 2])

        # Then populate matrix
        for row, (key, run) in enumerate(run_dict.items()):
            # Scaling is automatically done in configSpace
            conf = runhistory.ids_config[key.config_id]
            conf_vector = convert_configurations_to_array([conf])[0]
            if self.n_feats:
                feats = self.instance_features[key.instance_id]
                X[row, :] = np.hstack((conf_vector, feats))
            else:
                X[row, :] = conf_vector
            # run_array[row, -1] = instances[row]
            Y[row, 0] = run.cost
            Y[row, 1] = np.log(1 + run.time)

        return X, Y
