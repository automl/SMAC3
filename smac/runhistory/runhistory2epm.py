import abc
import logging
import typing

import numpy as np

from smac.tae import StatusType
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
        success_states: typing.List[StatusType],
        impute_censored_data: bool = False,
        impute_state: typing.Optional[typing.List[StatusType]] = None,
        consider_for_higher_budgets_state: typing.Optional[typing.List[StatusType]] = None,
        imputor: typing.Optional[BaseImputor] = None,
        scale_perc: int = 5,
        rng: typing.Optional[np.random.RandomState] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        scenario: Scenario Object
            Algorithm Configuration Scenario
        num_params : int
            number of parameters in config space
        success_states: list, optional
            List of states considered as successful (such as StatusType.SUCCESS).
            If None, raise TypeError.
        impute_censored_data: bool, optional
            Should we impute data?
        consider_for_higher_budgets_state: list, optional
            Additionally consider all runs with these states for budget < current budget
        imputor: epm.base_imputor Instance
            Object to impute censored data
        impute_state: list, optional
            List of states that mark censored data (such as StatusType.TIMEOUT)
            in combination with runtime < cutoff_time
            If None, set to empty list [].
            If None and impute_censored_data is True, raise TypeError.
        scale_perc: int
            scaled y-transformation use a percentile to estimate distance to optimum;
            only used by some subclasses of AbstractRunHistory2EPM
        rng : numpy.random.RandomState
            Only used for reshuffling data after imputation.
            If None, use np.random.RandomState(seed=1).
        """

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # General arguments
        self.scenario = scenario
        self.rng = rng
        self.num_params = num_params
        self.scale_perc = scale_perc

        # Configuration
        self.impute_censored_data = impute_censored_data
        self.cutoff_time = self.scenario.cutoff  # type: ignore[attr-defined] # noqa F821
        self.imputor = imputor

        # Fill with some default values
        if rng is None:
            self.rng = np.random.RandomState(seed=1)

        if impute_state is None and impute_censored_data:
            raise TypeError("impute_state not given")

        if impute_state is None:
            # please mypy
            self.impute_state = []  # type: typing.List[StatusType]
        else:
            self.impute_state = impute_state

        if consider_for_higher_budgets_state is None:
            # please mypy
            self.consider_for_higher_budgets_state = []  # type: typing.List[StatusType]
        else:
            self.consider_for_higher_budgets_state = consider_for_higher_budgets_state

        if success_states is None:
            raise TypeError("success_states not given")

        self.success_states = success_states

        self.instance_features = scenario.feature_dict
        self.n_feats = scenario.n_features

        self.num_params = num_params

        # Sanity checks
        if impute_censored_data and scenario.run_obj != "runtime":
            # So far we don't know how to handle censored quality data
            self.logger.critical("Cannot impute censored data when not "
                                 "optimizing runtime")
            raise NotImplementedError("Cannot impute censored data when not "
                                      "optimizing runtime")

        # Check imputor stuff
        if impute_censored_data and self.imputor is None:
            self.logger.critical("You want me to impute censored data, but "
                                 "I don't know how. Imputor is None")
            raise ValueError("impute_censored data, but no imputor given")
        elif impute_censored_data and not \
                isinstance(self.imputor, BaseImputor):
            raise ValueError("Given imputor is not an instance of "
                             "smac.epm.base_imputor.BaseImputor, but %s" %
                             type(self.imputor))

        # Learned statistics
        self.min_y = np.NaN
        self.max_y = np.NaN
        self.perc = np.NaN

    @abc.abstractmethod
    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory,
                      return_time_as_y: bool = False,
                      store_statistics: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Builds x,y matrixes from selected runs from runhistory

        Parameters
        ----------
        run_dict: dict(RunKey -> RunValue)
            dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            runhistory object
        return_time_as_y: bool
            Return the time instead of cost as y value. Necessary to access the raw y values for imputation.
        store_statistics: bool
            Whether to store statistics about the data (to be used at subsequent calls)

        Returns
        -------
        X: np.ndarray
        Y: np.ndarray
        """
        raise NotImplementedError()

    def _get_s_run_dict(
        self,
        runhistory: RunHistory,
        budget_subset: typing.Optional[typing.List] = None,
    ) -> typing.Dict[RunKey, RunValue]:
        # Get only successfully finished runs
        if budget_subset is not None:
            if len(budget_subset) != 1:
                raise ValueError("Cannot yet handle getting runs from multiple budgets")
            s_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                          if run.budget in budget_subset
                          and runhistory.data[run].status in self.success_states}
            # Additionally add these states from lower budgets
            add = {run: runhistory.data[run] for run in runhistory.data.keys()
                   if runhistory.data[run].status in self.consider_for_higher_budgets_state
                   and run.budget < budget_subset[0]}
            s_run_dict.update(add)
        else:
            s_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                          if runhistory.data[run].status in self.success_states}
        return s_run_dict

    def _get_t_run_dict(
            self,
            runhistory: RunHistory,
            budget_subset: typing.Optional[typing.List] = None,
    ) -> typing.Dict[RunKey, RunValue]:
        if budget_subset is not None:
            t_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                          if runhistory.data[run].status == StatusType.TIMEOUT
                          and runhistory.data[run].time >= self.cutoff_time
                          and run.budget in budget_subset}
        else:
            t_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                          if runhistory.data[run].status == StatusType.TIMEOUT
                          and runhistory.data[run].time >= self.cutoff_time}
        return t_run_dict

    def get_configurations(
        self,
        runhistory: RunHistory,
        budget_subset: typing.Optional[typing.List] = None,
    ) -> np.ndarray:
        """Returns vector representation of only the configurations.

        Instance features are not appended and cost values are not taken into account.

        Parameters
        ----------
        runhistory : smac.runhistory.runhistory.RunHistory
            Runhistory containing all evaluated configurations/instances
        budget_subset : list of budgets to consider

        Returns
        -------
        numpy.ndarray
        """
        s_runs = self._get_s_run_dict(runhistory, budget_subset)
        s_config_ids = set(s_run.config_id for s_run in s_runs)
        t_runs = self._get_t_run_dict(runhistory, budget_subset)
        t_config_ids = set(t_run.config_id for t_run in t_runs)
        config_ids = s_config_ids | t_config_ids
        configurations = [runhistory.ids_config[config_id] for config_id in config_ids]
        configs_array = convert_configurations_to_array(configurations)
        return configs_array

    def transform(
        self,
        runhistory: RunHistory,
        budget_subset: typing.Optional[typing.List] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Returns vector representation of runhistory; if imputation is
        disabled, censored (TIMEOUT with time < cutoff) will be skipped

        Parameters
        ----------
        runhistory : smac.runhistory.runhistory.RunHistory
            Runhistory containing all evaluated configurations/instances
        budget_subset : list of budgets to consider

        Returns
        -------
        X: numpy.ndarray
            configuration vector x instance features
        Y: numpy.ndarray
            cost values
        """
        self.logger.debug("Transform runhistory into X,y format")

        s_run_dict = self._get_s_run_dict(runhistory, budget_subset)
        X, Y = self._build_matrix(run_dict=s_run_dict, runhistory=runhistory,
                                  store_statistics=True)

        # Get real TIMEOUT runs
        t_run_dict = self._get_t_run_dict(runhistory, budget_subset)
        # use penalization (e.g. PAR10) for EPM training
        store_statistics = True if np.isnan(self.min_y) else False
        tX, tY = self._build_matrix(run_dict=t_run_dict, runhistory=runhistory,
                                    store_statistics=store_statistics)

        # if we don't have successful runs,
        # we have to return all timeout runs
        if not s_run_dict:
            return tX, tY

        if self.impute_censored_data:
            # Get all censored runs
            if budget_subset is not None:
                c_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                              if runhistory.data[run].status in self.impute_state
                              and runhistory.data[run].time < self.cutoff_time
                              and run.budget in budget_subset}
            else:
                c_run_dict = {run: runhistory.data[run] for run in runhistory.data.keys()
                              if runhistory.data[run].status in self.impute_state
                              and runhistory.data[run].time < self.cutoff_time}

            if len(c_run_dict) == 0:
                self.logger.debug("No censored data found, skip imputation")
                # If we do not impute, we also return TIMEOUT data
                X = np.vstack((X, tX))
                Y = np.concatenate((Y, tY))
            else:

                # better empirical results by using PAR1 instead of PAR10
                # for censored data imputation
                cen_X, cen_Y = self._build_matrix(run_dict=c_run_dict,
                                                  runhistory=runhistory,
                                                  return_time_as_y=True,
                                                  store_statistics=False,)

                # Also impute TIMEOUTS
                tX, tY = self._build_matrix(run_dict=t_run_dict,
                                            runhistory=runhistory,
                                            return_time_as_y=True,
                                            store_statistics=False,)
                self.logger.debug("%d TIMEOUTS, %d CAPPED, %d SUCC" %
                                  (tX.shape[0], cen_X.shape[0], X.shape[0]))
                cen_X = np.vstack((cen_X, tX))
                cen_Y = np.concatenate((cen_Y, tY))

                # return imp_Y in PAR depending on the used threshold in imputor
                assert isinstance(self.imputor, BaseImputor)  # please mypy
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

    @abc.abstractmethod
    def transform_response_values(self, values: np.ndarray, ) -> np.ndarray:
        """Transform function response values.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        raise NotImplementedError

    def get_X_y(self, runhistory: RunHistory) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple interface to obtain all data in runhistory in X, y format
        Note: This function should not be used as it does not consider all available StatusTypes

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
        self.logger.warning("This function is not tested and might not work as expected!")
        X = []
        y = []
        cen = []
        feature_dict = self.scenario.feature_dict
        params = self.scenario.cs.get_hyperparameters()  # type: ignore[attr-defined] # noqa F821
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
                      runhistory: RunHistory,
                      return_time_as_y: bool = False,
                      store_statistics: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
        """"Builds X,y matrixes from selected runs from runhistory

        Parameters
        ----------
        run_dict: dict: RunKey -> RunValue
            dictionary from RunHistory.RunKey to RunHistory.RunValue
        runhistory: RunHistory
            runhistory object
        return_time_as_y: bool
            Return the time instead of cost as y value. Necessary to access the raw y values for imputation.
        store_statistics: bool
            Whether to store statistics about the data (to be used at subsequent calls)

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
            if return_time_as_y:
                y[row, 0] = run.time
            else:
                y[row, 0] = run.cost

        if y.size > 0:
            if store_statistics:
                self.perc = np.percentile(y, self.scale_perc)
                self.min_y = np.min(y)
                self.max_y = np.max(y)
            y = self.transform_response_values(values=y)

        return X, y

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Returns the input values.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        return values


class RunHistory2EPM4LogCost(RunHistory2EPM4Cost):
    """TODO"""

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Transforms the response values by using a log transformation.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """

        # ensure that minimal value is larger than 0
        if np.any(values <= 0):
            self.logger.warning(
                "Got cost of smaller/equal to 0. Replace by %f since we use"
                " log cost." % constants.MINIMAL_COST_FOR_LOG)
            values[values < constants.MINIMAL_COST_FOR_LOG] = \
                constants.MINIMAL_COST_FOR_LOG
        values = np.log(values)
        return values


class RunHistory2EPM4ScaledCost(RunHistory2EPM4Cost):
    """TODO"""

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Transforms the response values by linearly scaling them between zero and one.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """

        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        # linear scaling
        if self.min_y == self.max_y:
            # prevent diving by zero
            min_y *= 1 - 10 ** -101
        values = (values - min_y) / (self.max_y - min_y)
        return values


class RunHistory2EPM4InvScaledCost(RunHistory2EPM4Cost):
    """TODO"""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def] # noqa F723
        super().__init__(**kwargs)
        if self.instance_features is not None:
            if len(self.instance_features) > 1:
                raise NotImplementedError('Handling more than one instance is not supported for inverse scaled cost.')

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Transform the response values by linearly scaling them between zero and one and then using inverse scaling.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """

        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below
        # linear scaling
        if min_y == self.max_y:
            # prevent diving by zero
            min_y *= 1 - 10 ** -10
        values = (values - min_y) / (self.max_y - min_y)
        values = 1 - 1 / values
        return values


class RunHistory2EPM4SqrtScaledCost(RunHistory2EPM4Cost):
    """TODO"""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]  # noqa F723
        super().__init__(**kwargs)
        if self.instance_features is not None:
            if len(self.instance_features) > 1:
                raise NotImplementedError('Handling more than one instance is not supported for sqrt scaled cost.')

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Transform the response values by linearly scaling them between zero and one and then using the square root.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """

        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        # linear scaling
        if min_y == self.max_y:
            # prevent diving by zero
            min_y *= 1 - 10 ** -10
        values = (values - min_y) / (self.max_y - min_y)
        values = np.sqrt(values)
        return values


class RunHistory2EPM4LogScaledCost(RunHistory2EPM4Cost):
    """TODO"""

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Transform the response values by linearly scaling them between zero and one and then using the log
        transformation.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """

        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below
        # linear scaling
        if min_y == self.max_y:
            # prevent diving by zero
            min_y *= 1 - 10 ** -10
        values = (values - min_y) / (self.max_y - min_y)
        values = np.log(values)
        return values


class RunHistory2EPM4EIPS(AbstractRunHistory2EPM):
    """TODO"""

    def _build_matrix(self, run_dict: typing.Mapping[RunKey, RunValue],
                      runhistory: RunHistory,
                      return_time_as_y: bool = False,
                      store_statistics: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
        """TODO"""
        if return_time_as_y:
            raise NotImplementedError()
        if store_statistics:
            # store_statistics is currently not necessary
            pass

        # First build nan-matrix of size #configs x #params+1
        n_rows = len(run_dict)
        n_cols = self.num_params
        X = np.ones([n_rows, n_cols + self.n_feats]) * np.nan
        y = np.ones([n_rows, 2])

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
            y[row, 0] = run.cost
            y[row, 1] = 1 + run.time

        y = self.transform_response_values(values=y)

        return X, y

    def transform_response_values(self, values: np.ndarray) -> typing.Tuple[np.ndarray]:
        """Transform function response values.

        Transform the runtimes by a log transformation (log(1 + runtime).

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """

        values[:, 1] = np.log(1 + values[:, 1])
        return values
