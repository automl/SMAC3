from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

import smac
import smac.acquisition as acquisition
from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class ConfigurationChooser:
    """Interface to train the EPM and generate/choose next configurations.

    Parameters
    ----------
    predict_x_best: bool
        Choose x_best for computing the acquisition function via the model instead of via the observations.
    min_samples_model: int
        Minimum number of samples to build a model
    """

    def __init__(
        self,
        predict_x_best: bool = True,
        min_samples_model: int = 1,
    ):
        self.smbo: smac.SMBO | None = None
        self._random_search: acquisition.random_search.RandomSearch | None = None

        self.initial_design_configs: list[Configuration] = []
        self.predict_x_best = predict_x_best
        self.min_samples_model = min_samples_model
        self.currently_considered_budgets = [
            0.0,
        ]

    def _set_smbo(self, smbo: smac.SMBO) -> None:
        self.smbo = smbo
        self._random_search = acquisition.random_search.RandomSearch(
            self.smbo.configspace,
            acquisition_function=self.smbo.acquisition_function,
            seed=self.smbo.seed,
        )

    def _collect_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.smbo is not None
        assert self._random_search is not None

        # if we use a float value as a budget, we want to train the model only on the highest budget
        available_budgets = []
        for run_key in self.smbo.runhistory.data.keys():
            available_budgets.append(run_key.budget)

        # Sort available budgets from highest to lowest budget
        available_budgets = sorted(list(set(available_budgets)), reverse=True)

        # Get #points per budget and if there are enough samples, then build a model
        for b in available_budgets:
            X, Y = self.smbo.runhistory_transformer.transform(
                self.smbo.runhistory,
                budget_subset=[
                    b,
                ],
            )
            if X.shape[0] >= self.min_samples_model:
                self.currently_considered_budgets = [
                    b,
                ]
                configs_array = self.smbo.runhistory_transformer.get_configurations(
                    self.smbo.runhistory, budget_subset=self.currently_considered_budgets
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
        assert self.smbo is not None
        return self.smbo.runhistory.get_all_configs_per_budget(budget_subset=self.currently_considered_budgets)

    def _get_x_best(self, predict: bool, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Get value, configuration, and array representation of the "best" configuration.

        The definition of best varies depending on the argument ``predict``. If set to ``True``,
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
        assert self.smbo is not None
        assert self._random_search is not None

        if predict:
            model = self.smbo.model
            costs = list(
                map(
                    lambda x: (
                        model.predict_marginalized_over_instances(x.reshape((1, -1)))[0][0][0],  # type: ignore
                        x,
                    ),
                    X,
                )
            )
            costs = sorted(costs, key=lambda t: t[0])
            x_best_array = costs[0][1]
            best_observation = costs[0][0]
            # won't need log(y) if EPM was already trained on log(y)
        else:
            all_configs = self.smbo.runhistory.get_all_configs_per_budget(
                budget_subset=self.currently_considered_budgets
            )
            x_best = self.smbo.incumbent
            x_best_array = convert_configurations_to_array(all_configs)
            best_observation = self.smbo.runhistory.get_cost(x_best)
            best_observation_as_array = np.array(best_observation).reshape((1, 1))
            # It's unclear how to do this for inv scaling and potential future scaling.
            # This line should be changed if necessary
            best_observation = self.smbo.runhistory_transformer.transform_response_values(best_observation_as_array)
            best_observation = best_observation[0][0]

        return x_best_array, best_observation

    def ask(self, incumbent_value: float = None) -> Iterator[Configuration]:
        """Choose next candidate solution with Bayesian optimization. The suggested configurations
        depend on the argument `acquisition_optimizer` to the `SMBO` class.

        Parameters
        ----------
        incumbent_value: float
            Cost value of incumbent configuration (required for acquisition function);
            If not given, it will be inferred from runhistory or predicted;
            if not given and runhistory is empty, it will raise a ValueError.

        Returns
        -------
        Iterator
        """
        assert self.smbo is not None
        assert self._random_search is not None

        logger.debug("Search for next configuration...")
        X, Y, X_configurations = self._collect_data()
        previous_configs = self.smbo.runhistory.get_all_configs()

        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations

            return self._random_search.maximize(previous_configs, num_points=1)

        self.smbo.model.train(X, Y)

        x_best_array: np.ndarray | None = None
        if incumbent_value is not None:
            best_observation = incumbent_value
        else:
            if self.smbo.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of " "the incumbent is unknown.")
            x_best_array, best_observation = self._get_x_best(self.predict_x_best, X_configurations)

        self.smbo.acquisition_function.update(
            model=self.smbo.model,
            eta=best_observation,
            incumbent_array=x_best_array,
            num_data=len(self._get_evaluated_configs()),
            X=X_configurations,
        )

        challengers = self.smbo.acquisition_optimizer.maximize(
            previous_configs,
            random_configuration_chooser=self.smbo.random_configuration_chooser,
        )

        return challengers
