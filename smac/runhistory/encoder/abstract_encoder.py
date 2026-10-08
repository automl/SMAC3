from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping

import numpy as np

from smac import constants
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.runhistory.runhistory import RunHistory, TrialKey, TrialValue
from smac.runner.abstract_runner import StatusType
from smac.scenario import Scenario
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AbstractRunHistoryEncoder:
    """Abstract class for preparing data in order to train a surrogate model.

    Parameters
    ----------
    scenario : Scenario object.
    considered_states : list[StatusType], defaults to [StatusType.SUCCESS, StatusType.CRASHED, StatusType.MEMORYOUT]  # noqa: E501
        Trials with the passed states are considered.
    lower_budget_states : list[StatusType], defaults to []
        Additionally consider all trials with these states for budget < current budget.
    scale_percentage : int, defaults to 5
        Scaled y-transformation use a percentile to estimate distance to optimum. Only used in some sub-classes.
    seed : int | None, defaults to none
    native_multi_objective: bool, defaults to False

    Raises
    ------
    TypeError
        If no success states are given.
    """

    def __init__(
        self,
        scenario: Scenario,
        considered_states: list[StatusType] = None,
        lower_budget_states: list[StatusType] = None,
        scale_percentage: int = 5,
        seed: int | None = None,
        native_multi_objective: bool = False,
        normalize: bool = True,
    ) -> None:
        if considered_states is None:
            considered_states = [
                StatusType.SUCCESS,
                StatusType.CRASHED,
                StatusType.MEMORYOUT,
            ]

        if seed is None:
            seed = scenario.seed

        self._seed = seed
        self._rng = np.random.RandomState(seed)
        self._scale_percentage = scale_percentage
        self._n_objectives = scenario.count_objectives()
        self._algorithm_walltime_limit = scenario.trial_walltime_limit
        self._lower_budget_states = lower_budget_states if lower_budget_states is not None else []
        self._considered_states = considered_states

        self._instances = scenario.instances
        self._instance_features = scenario.instance_features
        self._n_features = scenario.count_instance_features()
        self._n_params = len(list(scenario.configspace.values()))

        if self._instances is not None and self._n_features == 0:
            logger.warning(
                "We strongly encourage to use instance features when using instances."
                "If no instance features are passed, the runhistory encoder can not distinguish between different "
                "instances and therefore returns the same data points with different values, all of which are "
                "used to train the surrogate model.\n"
                "Consider using instance indices as features.",
            )

        # Learned statistics
        self._min_y = np.array([np.nan] * self._n_objectives)
        self._max_y = np.array([np.nan] * self._n_objectives)
        self._percentile = np.array([np.nan] * self._n_objectives)
        self._multi_objective_algorithm: AbstractMultiObjectiveAlgorithm | None = None
        self._runhistory: RunHistory | None = None

        self._native_multi_objective = native_multi_objective
        self._normalize = normalize

    @property
    def meta(self) -> dict[str, Any]:
        """
        Returns the meta-data of the created object.

        Returns
        -------
        dict[str, Any]: meta-data of the created object: name, considered states, lower budget
        states, scale_percentage, seed.
        """
        return {
            "name": self.__class__.__name__,
            "considered_states": self._considered_states,
            "lower_budget_states": self._lower_budget_states,
            "scale_percentage": self._scale_percentage,
            "seed": self._seed,
        }

    @property
    def runhistory(self) -> RunHistory:
        """The RunHistory used to transform the data."""
        assert self._runhistory is not None
        return self._runhistory

    @runhistory.setter
    def runhistory(self, runhistory: RunHistory) -> None:
        """Sets the multi objective algorithm."""
        self._runhistory = runhistory

    @property
    def multi_objective_algorithm(self) -> AbstractMultiObjectiveAlgorithm | None:
        """The multi objective algorithm used to transform the data."""
        return self._multi_objective_algorithm

    @multi_objective_algorithm.setter
    def multi_objective_algorithm(self, algorithm: AbstractMultiObjectiveAlgorithm) -> None:
        """Sets the multi objective algorithm."""
        self._multi_objective_algorithm = algorithm

    @abstractmethod
    def _build_matrix(
        self,
        trials: Mapping[TrialKey, TrialValue],
        store_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds x and y matrices from selected runs of the RunHistory.

        Parameters
        ----------
        trials : Mapping[TrialKey, TrialValue]
        runhistory : RunHistory
        store_statistics: bool, defaults to false
            Whether to store statistics about the data (to be used at subsequent calls).

        Returns
        -------
        X : np.ndarray
        Y : np.ndarray
        """
        raise NotImplementedError()

    def _get_considered_trials(
        self,
        budget_subset: list | None = None,
    ) -> dict[TrialKey, TrialValue]:
        """
        Returns all trials that are considered for the model.
        Depends on the user's considered states and lower budget states.

        Parameters
        ----------
        budget_subset : list[int|float] | None, defaults to None.
        """
        trials: dict[TrialKey, TrialValue] = {}

        if budget_subset is not None:
            if len(budget_subset) != 1:
                raise ValueError("Can not yet handle getting runs from multiple budgets.")

        for trial_key, trial_value in self.runhistory.items():
            add = False
            if budget_subset is not None:
                if trial_key.budget in budget_subset and trial_value.status in self._considered_states:
                    add = True

                if (
                    trial_key.budget is not None
                    and budget_subset[0] is not None
                    and trial_key.budget < budget_subset[0]
                    and trial_value.status in self._lower_budget_states
                ):
                    add = True
            else:
                # Get only successfully finished runs
                if trial_value.status in self._considered_states:
                    add = True

            if add:
                trials[trial_key] = trial_value

        return trials

    def _get_timeout_trials(
        self,
        budget_subset: list | None = None,
    ) -> dict[TrialKey, TrialValue]:
        """Returns all trials that did have a timeout."""
        if budget_subset is not None:
            trials = {
                trial: self.runhistory[trial]
                for trial in self.runhistory
                if self.runhistory[trial].status == StatusType.TIMEOUT
                # and runhistory.data[run].time >= self._algorithm_walltime_limit  # type: ignore
                and trial.budget in budget_subset
            }
        else:
            trials = {
                trial: self.runhistory[trial]
                for trial in self.runhistory
                if self.runhistory[trial].status == StatusType.TIMEOUT
                # and runhistory.data[run].time >= self._algorithm_walltime_limit  # type: ignore
            }

        return trials

    def get_configurations(
        self,
        budget_subset: list | None = None,
    ) -> np.ndarray:
        """Returns vector representation of the configurations.

        Warning
        -------
        Instance features are not
        appended and cost values are not taken into account.

        Parameters
        ----------
        budget_subset : list[int|float] | None, defaults to none
            List of budgets to consider.

        Returns
        -------
        configs_array : np.ndarray
        """
        s_trials = self._get_considered_trials(budget_subset)
        s_config_ids = set(s_trial.config_id for s_trial in s_trials)
        t_trials = self._get_timeout_trials(budget_subset)
        t_config_ids = set(t_trial.config_id for t_trial in t_trials)
        config_ids = s_config_ids | t_config_ids
        configurations = [self.runhistory._ids_config[config_id] for config_id in config_ids]
        configs_array = convert_configurations_to_array(configurations)

        return configs_array

    def transform(
        self,
        budget_subset: list | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a vector representation of the RunHistory.

        Parameters
        ----------
        budget_subset : list | None, defaults to none
            List of budgets to consider.

        Returns
        -------
        X : np.ndarray
            Configuration vector and instance features.
        Y : np.ndarray
            Cost values.
        """
        logger.debug("Transforming RunHistory into X, y format...")

        considered_trials = self._get_considered_trials(budget_subset)
        X, Y = self._build_matrix(trials=considered_trials, store_statistics=True)

        # Get real TIMEOUT runs
        timeout_trials = self._get_timeout_trials(budget_subset)

        # Use penalization (e.g. PAR10) for EPM training
        store_statistics = True if np.any(np.isnan(self._min_y)) else False
        tX, tY = self._build_matrix(trials=timeout_trials, store_statistics=store_statistics)

        # If we don't have successful runs, we have to return all timeout runs
        if not considered_trials:
            return tX, tY

        # If we do not impute, we also return TIMEOUT data
        X = np.vstack((X, tX))
        Y = np.concatenate((Y, tY))

        logger.debug("Converted %d observations." % (X.shape[0]))
        return X, Y

    def inverse_transform_response_values(self, values: np.ndarray) -> np.ndarray | None:
        """Maps values the surrogate model works in back to cost values, or returns `None` if this encoder cannot.

        The inverse of `transform_response_values`, for a reader that wants an answer in the objective's own units:
        a difference between two model predictions means something different under every encoder. It applies the
        statistics stored by the last `transform` call, as `transform_response_values` itself does.

        Only an inverse defined by the same class as the transform is trusted. A subclass that changes the
        transform without supplying the matching inverse gets `None` rather than the inverse of its parent's.

        Parameters
        ----------
        values : np.ndarray
            Values as the surrogate model sees them.

        Returns
        -------
        np.ndarray | None
            The corresponding cost values, or `None` if the transformation cannot be inverted.
        """
        if _owner(type(self), "transform_response_values") is not _owner(type(self), "_inverse_response_values"):
            return None

        return self._inverse_response_values(np.asarray(values, dtype=float))

    def _inverse_response_values(self, values: np.ndarray) -> np.ndarray | None:
        """The inverse of `transform_response_values`, or `None` if there is none.

        See `inverse_transform_response_values`.
        """
        return None

    def _scaling_bounds(self, collapse_factor: float) -> tuple[np.ndarray, np.ndarray]:
        """The lower and upper ends the scaling encoders map costs between, from the stored statistics.

        The lower end sits below the smallest cost by as much as the percentile sits above it, and a little further,
        so that the smallest cost scales to a small positive number rather than to zero. Where every cost was the
        same, the lower end is moved by `collapse_factor` so that the scaling does not divide by zero.
        """
        min_y = self._min_y - (self._percentile - self._min_y)
        min_y -= constants.VERY_SMALL_NUMBER
        min_y[np.where(min_y == self._max_y)] *= collapse_factor

        return min_y, self._max_y

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
        transformed_values : np.ndarray
        """
        raise NotImplementedError


def _owner(cls: type, name: str) -> type | None:
    """The class in `cls`'s method resolution order that defines `name`."""
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass

    return None
