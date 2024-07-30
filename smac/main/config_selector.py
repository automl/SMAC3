from __future__ import annotations

from typing import Any, Iterator

import copy

import numpy as np
from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.callback.callback import Callback
from smac.initial_design import AbstractInitialDesign
from smac.model.abstract_model import AbstractModel
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class ConfigSelector:
    """The config selector handles the surrogate model and the acquisition function. Based on both components, the next
    configuration is selected.

    Parameters
    ----------
    retrain_after : int, defaults to 8
        How many configurations should be returned before the surrogate model is retrained.
    retries : int, defaults to 8
        How often to retry receiving a new configuration before giving up.
    min_trials: int, defaults to 1
        How many samples are required to train the surrogate model. If budgets are involved,
        the highest budgets are checked first. For example, if min_trials is three, but we find only
        two trials in the runhistory for the highest budget, we will use trials of a lower budget
        instead.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        retrain_after: int = 8,
        retries: int = 16,
        min_trials: int = 1,
    ) -> None:
        # Those are the configs sampled from the passed initial design
        # Selecting configurations from initial design
        self._initial_design_configs: list[Configuration] = []

        # Set classes globally
        self._scenario = scenario
        self._runhistory: RunHistory | None = None
        self._runhistory_encoder: AbstractRunHistoryEncoder | None = None
        self._model: AbstractModel | None = None
        self._acquisition_maximizer: AbstractAcquisitionMaximizer | None = None
        self._acquisition_function: AbstractAcquisitionFunction | None = None
        self._random_design: AbstractRandomDesign | None = None
        self._callbacks: list[Callback] = []

        # And other variables
        self._retrain_after = retrain_after
        self._previous_entries = -1
        self._predict_x_best = True
        self._min_trials = min_trials
        self._considered_budgets: list[float | int | None] = [None]

        # How often to retry receiving a new configuration
        # (counter increases if the received config was already returned before)
        self._retries = retries

        # Processed configurations should be stored here; this is important to not return the same configuration twice
        self._processed_configs: list[Configuration] = []

    def _set_components(
        self,
        initial_design: AbstractInitialDesign,
        runhistory: RunHistory,
        runhistory_encoder: AbstractRunHistoryEncoder,
        model: AbstractModel,
        acquisition_maximizer: AbstractAcquisitionMaximizer,
        acquisition_function: AbstractAcquisitionFunction,
        random_design: AbstractRandomDesign,
        callbacks: list[Callback] = None,
    ) -> None:
        self._runhistory = runhistory
        self._runhistory_encoder = runhistory_encoder
        self._model = model
        self._acquisition_maximizer = acquisition_maximizer
        self._acquisition_function = acquisition_function
        self._random_design = random_design
        self._callbacks = callbacks if callbacks is not None else []

        self._initial_design_configs = initial_design.select_configurations()
        if len(self._initial_design_configs) == 0:
            raise RuntimeError("SMAC needs initial configurations to work.")

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "retrain_after": self._retrain_after,
            "retries": self._retries,
            "min_trials": self._min_trials,
        }

    def __iter__(self) -> Iterator[Configuration]:
        """This method returns the next configuration to evaluate. It ignores already processed configurations, i.e.,
        the configurations from the runhistory, if the runhistory is not empty.
        The method (after yielding the initial design configurations) trains the surrogate model, maximizes the
        acquisition function and yields ``n`` configurations. After the ``n`` configurations, the surrogate model is
        trained again, etc. The program stops if ``retries`` was reached within each iteration. A configuration
        is ignored, if it was used already before.

        Note
        ----
        When SMAC continues a run, processed configurations from the runhistory are ignored. For example, if the
        intitial design configurations already have been processed, they are ignored here. After the run is
        continued, however, the surrogate model is trained based on the runhistory in all cases.

        Returns
        -------
        next_config : Iterator[Configuration]
            The next configuration to evaluate.
        """
        assert self._runhistory is not None
        assert self._runhistory_encoder is not None
        assert self._model is not None
        assert self._acquisition_maximizer is not None
        assert self._acquisition_function is not None
        assert self._random_design is not None

        self._processed_configs = self._runhistory.get_configs()

        # We add more retries because there could be a case in which the processed configs are sampled again
        self._retries += len(self._processed_configs)

        logger.debug("Search for the next configuration...")
        self._call_callbacks_on_start()

        # First: We return the initial configurations
        for config in self._initial_design_configs:
            if config not in self._processed_configs:
                self._processed_configs.append(config)
                self._call_callbacks_on_end(config)
                yield config
                self._call_callbacks_on_start()

        # We want to generate configurations endlessly
        while True:
            # Cost value of incumbent configuration (required for acquisition function).
            # If not given, it will be inferred from runhistory or predicted.
            # If not given and runhistory is empty, it will raise a ValueError.
            incumbent_value: float | None = None

            # Everytime we re-train the surrogate model, we also update our multi-objective algorithm
            if (mo := self._runhistory_encoder.multi_objective_algorithm) is not None:
                mo.update_on_iteration_start()

            X, Y, X_configurations = self._collect_data()
            previous_configs = self._runhistory.get_configs()

            if X.shape[0] == 0:
                # Only return a single point to avoid an overly high number of random search iterations.
                # We got rid of random search here and replaced it with a simple configuration sampling from
                # the configspace.
                logger.debug("No data available to train the model. Sample a random configuration.")

                config = self._scenario.configspace.sample_configuration(1)
                self._call_callbacks_on_end(config)
                yield config
                self._call_callbacks_on_start()

                # Important to continue here because we still don't have data available
                continue

            # Check if X/Y differs from the last run, otherwise use cached results
            if self._previous_entries != Y.shape[0]:
                self._model.train(X, Y)

                x_best_array: np.ndarray | None = None
                if incumbent_value is not None:
                    best_observation = incumbent_value
                else:
                    if self._runhistory.empty():
                        raise ValueError("Runhistory is empty and the cost value of the incumbent is unknown.")

                    x_best_array, best_observation = self._get_x_best(X_configurations)

                self._acquisition_function.update(
                    model=self._model,
                    eta=best_observation,
                    incumbent_array=x_best_array,
                    num_data=len(self._get_evaluated_configs()),
                    X=X_configurations,
                )

            # We want to cache how many entries we used because if we have the same number of entries
            # we don't need to train the next time
            self._previous_entries = Y.shape[0]

            # Now we maximize the acquisition function
            challengers = self._acquisition_maximizer.maximize(
                previous_configs,
                random_design=self._random_design,
            )

            counter = 0
            failed_counter = 0
            for config in challengers:
                if config not in self._processed_configs:
                    counter += 1
                    self._processed_configs.append(config)
                    self._call_callbacks_on_end(config)
                    yield config
                    retrain = counter == self._retrain_after
                    self._call_callbacks_on_start()

                    # We break to enforce a new iteration of the while loop (i.e. we retrain the surrogate model)
                    if retrain:
                        logger.debug(
                            f"Yielded {counter} configurations. Start new iteration and retrain surrogate model."
                        )
                        break
                else:
                    failed_counter += 1

                    # We exit the loop if we have tried to add the same configuration too often
                    if failed_counter == self._retries:
                        logger.warning(f"Could not return a new configuration after {self._retries} retries." "")
                        return

    def _call_callbacks_on_start(self) -> None:
        for callback in self._callbacks:
            callback.on_next_configurations_start(self)

    def _call_callbacks_on_end(self, config: Configuration) -> None:
        """Calls ``on_next_configurations_end`` of the registered callbacks."""
        # For safety reasons: Return a copy of the config
        if len(self._callbacks) > 0:
            config = copy.deepcopy(config)

        for callback in self._callbacks:
            callback.on_next_configurations_end(self, config)

    def _collect_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collects the data from the runhistory to train the surrogate model. In the case of budgets, the data
        collection strategy is as follows: Looking from highest to lowest budget, return those observations
        that support at least ``self._min_trials`` points.

        If no budgets are used, this is equivalent to returning all observations.
        """
        assert self._runhistory is not None
        assert self._runhistory_encoder is not None

        # If we use a float value as a budget, we want to train the model only on the highest budget
        unique_budgets: set[float] = {run_key.budget for run_key in self._runhistory if run_key.budget is not None}

        available_budgets: list[float] | list[None]
        if len(unique_budgets) > 0:
            # Sort available budgets from highest to lowest budget
            available_budgets = sorted(unique_budgets, reverse=True)
        else:
            available_budgets = [None]

        # Get #points per budget and if there are enough samples, then build a model
        for b in available_budgets:
            X, Y = self._runhistory_encoder.transform(budget_subset=[b])

            if X.shape[0] >= self._min_trials:
                self._considered_budgets = [b]

                # Possible add running configs?
                configs_array = self._runhistory_encoder.get_configurations(budget_subset=self._considered_budgets)

                return X, Y, configs_array

        return (
            np.empty(shape=[0, 0]),
            np.empty(
                shape=[
                    0,
                ]
            ),
            np.empty(shape=[0, 0]),
        )

    def _get_evaluated_configs(self) -> list[Configuration]:
        assert self._runhistory is not None
        return self._runhistory.get_configs_per_budget(budget_subset=self._considered_budgets)

    def _get_x_best(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Get value, configuration, and array representation of the *best* configuration.

        The definition of best varies depending on the argument ``predict``. If set to `True`,
        this function will return the stats of the best configuration as predicted by the model,
        otherwise it will return the stats for the best observed configuration.

        Parameters
        ----------
        predict : bool
            Whether to use the predicted or observed best.

        Returns
        -------
        float
        np.ndarry
        Configuration
        """
        if self._predict_x_best:
            model = self._model
            costs = list(
                map(
                    lambda x: (
                        model.predict_marginalized(x.reshape((1, -1)))[0][0][0],  # type: ignore
                        x,
                    ),
                    X,
                )
            )
            costs = sorted(costs, key=lambda t: t[0])
            x_best_array = costs[0][1]
            best_observation = costs[0][0]

        # else:
        #    all_configs = self._runhistory.get_configs_per_budget(budget_subset=self._considered_budgets)
        #    x_best = self._incumbent
        #    x_best_array = convert_configurations_to_array(all_configs)
        #    best_observation = self._runhistory.get_cost(x_best)
        #    best_observation_as_array = np.array(best_observation).reshape((1, 1))

        #    # It's unclear how to do this for inv scaling and potential future scaling.
        #    # This line should be changed if necessary
        #    best_observation = self._runhistory_encoder.transform_response_values(best_observation_as_array)
        #    best_observation = best_observation[0][0]

        return x_best_array, best_observation
