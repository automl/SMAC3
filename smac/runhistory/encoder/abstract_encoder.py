from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping

import numpy as np

from smac.utils.configspace import convert_configurations_to_array

from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.runhistory.runhistory import RunHistory, TrialKey, TrialValue
from smac.runner.abstract_runner import StatusType
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AbstractRunHistoryEncoder:
    """Abstract class for preprocessing data in order to train an EPM.

    Parameters
    ----------
    scenario: config Object
        Algorithm Configuration config
    n_params : int
        number of parameters in config space
    success_states: list, optional
        List of states considered as successful (such as StatusType.SUCCESS).
        If None, raise TypeError.
    impute_censored_data: bool, optional
        Should we impute data?
    consider_for_higher_budgets_state: list, optional
        Additionally consider all runs with these states for budget < current budget
    imputer: epm.base_imputer Instance
        Object to impute censored data
    impute_state: list, optional
        List of states that mark censored data (such as StatusType.TIMEOUT)
        in combination with runtime < cutoff_time
        If None, set to empty list [].
        If None and impute_censored_data is True, raise TypeError.
    scale_percentage: int
        scaled y-transformation use a percentile to estimate distance to optimum;
        only used by some subclasses of AbstractRunHistory2EPM
    rng : numpy.random.RandomState
        Only used for reshuffling data after imputation.
        If None, use np.random.RandomState(seed=1).
    multi_objective_algorithm: Optional[MultiObjectiveAlgorithm]
        Instance performing multi-objective optimization. Receives an objective cost vector as input
        and returns a scalar. Is executed before transforming runhistory values.

    Attributes
    ----------
    logger
    config
    rng
    n_params

    success_states
    impute_censored_data
    impute_state
    cutoff_time
    imputer
    instance_features
    n_feats
    n_params
    """

    def __init__(
        self,
        scenario: Scenario,
        success_states: list[StatusType] = [
            StatusType.SUCCESS,
            StatusType.CRASHED,
            StatusType.MEMORYOUT,
            StatusType.DONOTADVANCE,
        ],
        consider_for_higher_budgets_state: list[StatusType]
        | None = [
            StatusType.DONOTADVANCE,
            StatusType.TIMEOUT,
            StatusType.CRASHED,
            StatusType.MEMORYOUT,
        ],  # TODO: Before it was None; I don't know if we can do that here...
        scale_percentage: int = 5,
        seed: int | None = None,
    ) -> None:
        # General arguments
        self.scenario = scenario

        if seed is None:
            seed = scenario.seed

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.scale_percentage = scale_percentage
        self.n_objectives = scenario.count_objectives()
        self.algorithm_walltime_limit = self.scenario.trial_walltime_limit

        self.consider_for_higher_budgets_state: list[StatusType]
        if consider_for_higher_budgets_state is None:
            self.consider_for_higher_budgets_state = []
        else:
            self.consider_for_higher_budgets_state = consider_for_higher_budgets_state

        if success_states is None:
            raise TypeError("No success states are given.")

        self.success_states = success_states

        self.instances = scenario.instances
        self.instance_features = scenario.instance_features
        self.n_features = scenario.count_instance_features()
        self.n_params = len(scenario.configspace.get_hyperparameters())

        # Learned statistics
        self.min_y = np.array([np.NaN] * self.n_objectives)
        self.max_y = np.array([np.NaN] * self.n_objectives)
        self.perc = np.array([np.NaN] * self.n_objectives)

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    # def _set_imputer(self, imputer: AbstractImputer | None) -> None:
    #    self.imputer = imputer

    def _set_multi_objective_algorithm(self, multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None) -> None:
        self.multi_objective_algorithm = multi_objective_algorithm

    @abstractmethod
    def _build_matrix(
        self,
        run_dict: Mapping[TrialKey, TrialValue],
        runhistory: RunHistory,
        store_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds x,y matrixes from selected runs from runhistory.

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

    def _get_successful_runs(
        self,
        runhistory: RunHistory,
        budget_subset: list | None = None,
    ) -> dict[TrialKey, TrialValue]:
        # Get only successfully finished runs
        if budget_subset is not None:
            if len(budget_subset) != 1:
                raise ValueError("Can not yet handle getting runs from multiple budgets.")

            s_run_dict = {
                run: runhistory.data[run]
                for run in runhistory.data.keys()
                if run.budget in budget_subset and runhistory.data[run].status in self.success_states
            }
            # Additionally add these states from lower budgets
            add = {
                run: runhistory.data[run]
                for run in runhistory.data.keys()
                if runhistory.data[run].status in self.consider_for_higher_budgets_state
                and run.budget < budget_subset[0]
            }
            s_run_dict.update(add)
        else:
            s_run_dict = {
                run: runhistory.data[run]
                for run in runhistory.data.keys()
                if runhistory.data[run].status in self.success_states
            }

        return s_run_dict

    def _get_timeout_runs(
        self,
        runhistory: RunHistory,
        budget_subset: list | None = None,
    ) -> dict[TrialKey, TrialValue]:
        if budget_subset is not None:
            t_run_dict = {
                run: runhistory.data[run]
                for run in runhistory.data.keys()
                if runhistory.data[run].status == StatusType.TIMEOUT
                and runhistory.data[run].time >= self.algorithm_walltime_limit  # type: ignore
                and run.budget in budget_subset
            }
        else:
            t_run_dict = {
                run: runhistory.data[run]
                for run in runhistory.data.keys()
                if runhistory.data[run].status == StatusType.TIMEOUT
                and runhistory.data[run].time >= self.algorithm_walltime_limit  # type: ignore
            }

        return t_run_dict

    def get_configurations(
        self,
        runhistory: RunHistory,
        budget_subset: list | None = None,
    ) -> np.ndarray:
        """Returns vector representation of only the configurations. Instance features are not
        appended and cost values are not taken into account.

        Parameters
        ----------
        runhistory : smac.runhistory.runhistory.RunHistory
            Runhistory containing all evaluated configurations/instances
        budget_subset : list of budgets to consider

        Returns
        -------
        numpy.ndarray
        """
        s_runs = self._get_successful_runs(runhistory, budget_subset)
        s_config_ids = set(s_run.config_id for s_run in s_runs)
        t_runs = self._get_timeout_runs(runhistory, budget_subset)
        t_config_ids = set(t_run.config_id for t_run in t_runs)
        config_ids = s_config_ids | t_config_ids
        configurations = [runhistory.ids_config[config_id] for config_id in config_ids]
        configs_array = convert_configurations_to_array(configurations)
        return configs_array

    def transform(
        self,
        runhistory: RunHistory,
        budget_subset: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns vector representation of runhistory; if imputation is disabled, censored (TIMEOUT
        with time < cutoff) will be skipped.

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
        logger.debug("Transforming runhistory into X, y format...")

        s_run_dict = self._get_successful_runs(runhistory, budget_subset)
        X, Y = self._build_matrix(run_dict=s_run_dict, runhistory=runhistory, store_statistics=True)

        # Get real TIMEOUT runs
        t_run_dict = self._get_timeout_runs(runhistory, budget_subset)
        # use penalization (e.g. PAR10) for EPM training
        store_statistics = True if np.any(np.isnan(self.min_y)) else False
        tX, tY = self._build_matrix(
            run_dict=t_run_dict,
            runhistory=runhistory,
            store_statistics=store_statistics,
        )

        # If we don't have successful runs,
        # we have to return all timeout runs
        if not s_run_dict:
            return tX, tY

        # If we do not impute, we also return TIMEOUT data
        X = np.vstack((X, tX))
        Y = np.concatenate((Y, tY))

        logger.debug("Converted %d observations." % (X.shape[0]))
        return X, Y

    @abstractmethod
    def transform_response_values(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
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
