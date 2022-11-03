from __future__ import annotations

from typing import Iterator


import numpy as np
from ConfigSpace import Configuration
from smac.runhistory.encoder.abstract_encoder import AbstractRunHistoryEncoder

from smac.utils.logging import get_logger

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.initial_design import AbstractInitialDesign
from smac.model.abstract_model import AbstractModel
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.runhistory.runhistory import RunHistory
from smac.scenario import Scenario


__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class ConfigSelector:
    def __init__(
        self,
        scenario: Scenario,
        initial_design: AbstractInitialDesign,
        runhistory: RunHistory,
        runhistory_encoder: AbstractRunHistoryEncoder,
        model: AbstractModel,
        acquisition_maximizer: AbstractAcquisitionMaximizer,
        acquisition_function: AbstractAcquisitionFunction,
        random_design: AbstractRandomDesign,
        n: int = 8,
    ) -> None:
        # Those are the configs sampled from the passed initial design
        # Selecting configurations from initial design
        self._initial_design_configs = initial_design.select_configurations()
        if len(self._initial_design_configs) == 0:
            raise RuntimeError("SMAC needs initial configurations to work.")

        # Set classes globally
        self._scenario = scenario
        self._runhistory = runhistory
        self._runhistory_encoder = runhistory_encoder
        self._model = model
        self._acquisition_maximizer = acquisition_maximizer
        self._acquisition_function = acquisition_function
        self._random_design = random_design

        # And other variables
        self._n = n
        self._previous_entries = -1
        self._predict_x_best = True
        self._min_samples = 1
        self._considered_budgets: list[float | None] = [None]

        # How often to retry receiving a new configuration
        # (counter increases if the received config was already returned before)
        self._n_retries = n

        # Processed configurations should be stored here
        # Important: We have to read them from the runhistory!
        self._processed_configs = self._runhistory.get_configs()

    def __iter__(self) -> Iterator[Configuration]:
        """This method returns the next configuration to evaluate. It ignores already processed configs, i.e.,
        the configs from the runhistory if the runhistory is not empty.
        The method (after yielding the initial design configurations) trains the surrogate model, maximizes the
        acquisition function and yields ``n`` configurations. After the ``n`` configurations the surrogate model is
        trained again, etc. The program stops if ``self._n_retries`` was reached within each iteration. A configuration
        is rejected if it was used already before.

        Returns
        -------
        next_config : Iterator[Configuration]
            The next configuration to evaluate.
        """
        logger.debug("Search for the next configuration...")

        # First: We return the initial configurations
        for config in self._initial_design_configs:
            if config not in self._processed_configs:
                self._processed_configs.append(config)
                yield config

        # for callback in self._callbacks:
        #    callback.on_next_configurations_start(self)

        # We want to generate configurations endlessly
        while True:
            # Cost value of incumbent configuration (required for acquisition function).
            # If not given, it will be inferred from runhistory or predicted.
            # If not given and runhistory is empty, it will raise a ValueError.
            incumbent_value: float | None = None

            X, Y, X_configurations = self._collect_data()
            previous_configs = self._runhistory.get_configs()

            if X.shape[0] == 0:
                # Only return a single point to avoid an overly high number of random search iterations.
                # We got rid of random search here and replaced it with a simple configuration sampling from
                # the configspace.
                logger.debug("No data available to train the model. Sample a random configuration.")
                assert len(self._processed_configs) == 0

                config = self._scenario.configspace.sample_configuration(1)
                yield config

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
                n_points=self._n + self._n_retries,
                random_design=self._random_design,
            )

            counter = 0
            failed_counter = 0
            for config in challengers:
                if config not in self._processed_configs:
                    counter += 1
                    self._processed_configs.append(config)
                    yield config

                    # We break to enforce a new iteration of the while loop (i.e. we retrain the surrogate model)
                    if counter == self._n:
                        break
                else:
                    failed_counter += 1

                    # We exit the loop if we have tried to add the same configuration too often
                    if failed_counter == self._n_retries:
                        logger.warning(f"Could not return a new configuration after {self._n_retries} retries." "")
                        exit()

        # for callback in self._callbacks:
        #    challenger_list = list(copy.deepcopy(challengers))
        #    callback.on_next_configurations_end(self, challenger_list)

    def _collect_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collects the data from the runhistory to train the surrogate model. The data collection strategy if budgets
        are used is as follows: Looking from highest to lowest budget, return those observations
        that support at least ``self._min_samples`` points.

        If no budgets are used, this is equivalent to returning all observations.
        """
        # If we use a float value as a budget, we want to train the model only on the highest budget
        available_budgets = []
        for run_key in self._runhistory:
            available_budgets.append(run_key.budget)

        # Sort available budgets from highest to lowest budget
        available_budgets = sorted(list(set(available_budgets)), reverse=True)  # type: ignore

        # Get #points per budget and if there are enough samples, then build a model
        for b in available_budgets:
            X, Y = self._runhistory_encoder.transform(self._runhistory, budget_subset=[b])

            if X.shape[0] >= self._min_samples:
                self._considered_budgets = [b]
                
                # TODO: Add running configs
                
                configs_array = self._runhistory_encoder.get_configurations(
                    self._runhistory, budget_subset=self._considered_budgets
                )

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
