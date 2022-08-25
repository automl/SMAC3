from __future__ import annotations

from typing import Iterator

import numpy as np


from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.utils.logging import get_logger
from smac.main.base_smbo import BaseSMBO
import numpy as np
from smac.runhistory import RunInfo, RunValue, StatusType
from smac.utils.logging import get_logger
from smac.runner.exceptions import (
    FirstRunCrashedException,
    TargetAlgorithmAbortException,
)


__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class SMBO(BaseSMBO):
    """Interface to train the EPM and generate/choose next configurations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.initial_design_configs: list[Configuration] = []
        self.predict_x_best = True
        self.min_samples_model = 1
        self.currently_considered_budgets = [
            0.0,
        ]

    def ask(self) -> Iterator[Configuration]:
        """Choose next candidate solution with Bayesian optimization. The suggested configurations
        depend on the surrogate model acquisition optimizer/function.
        """

        for callback in self._callbacks:
            callback.on_ask_start(self)

        # Cost value of incumbent configuration (required for acquisition function).
        # If not given, it will be inferred from runhistory or predicted.
        # If not given and runhistory is empty, it will raise a ValueError.
        incumbent_value: float = None

        logger.debug("Search for next configuration...")
        X, Y, X_configurations = self._collect_data()
        previous_configs = self.runhistory.get_configs()

        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations

            # Let's get rid of this random search here...
            # return self._random_search.maximize(previous_configs, num_points=1)

            return iter([self.configspace.sample_configuration(1)])

        self.model.train(X, Y)

        x_best_array: np.ndarray | None = None
        if incumbent_value is not None:
            best_observation = incumbent_value
        else:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of " "the incumbent is unknown.")

            x_best_array, best_observation = self._get_x_best(self.predict_x_best, X_configurations)

        self.acquisition_function.update(
            model=self.model,
            eta=best_observation,
            incumbent_array=x_best_array,
            num_data=len(self._get_evaluated_configs()),
            X=X_configurations,
        )

        challengers = self.acquisition_optimizer.maximize(
            previous_configs,
            random_design=self.random_design,
        )

        for callback in self._callbacks:
            callback.on_ask_end(self, challengers)

        return challengers

    def tell(self, run_info: RunInfo, run_value: RunValue, time_left: float, save: bool = True) -> None:
        # We removed `abort_on_first_run_crash` and therefore we expect the first
        # run to always succeed.
        if self.stats.finished == 0 and run_value.status == StatusType.CRASHED:
            additional_info = ""
            if "traceback" in run_value.additional_info:
                additional_info = "\n\n" + run_value.additional_info["traceback"]

            raise FirstRunCrashedException("The first run crashed. Please check your setup again." + additional_info)

        # Update SMAC stats
        self.stats.target_algorithm_walltime_used += float(run_value.time)
        self.stats.finished += 1

        logger.debug(
            f"Status: {run_value.status}, cost: {run_value.cost}, time: {run_value.time}, "
            f"Additional: {run_value.additional_info}"
        )

        self.runhistory.add(
            config=run_info.config,
            cost=run_value.cost,
            time=run_value.time,
            status=run_value.status,
            instance=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
            starttime=run_value.starttime,
            endtime=run_value.endtime,
            force_update=True,
            additional_info=run_value.additional_info,
        )
        self.stats.n_configs = len(self.runhistory.config_ids)

        if run_value.status == StatusType.ABORT:
            raise TargetAlgorithmAbortException(
                "The target algorithm was aborted. The last incumbent can be found in the trajectory file."
            )
        elif run_value.status == StatusType.STOP:
            self._stop = True
            return

        # Update the intensifier with the result of the runs
        self.incumbent, _ = self.intensifier.process_results(
            run_info=run_info,
            run_value=run_value,
            incumbent=self.incumbent,
            runhistory=self.runhistory,
            time_bound=max(self._min_time, time_left),
        )

        if save:
            self.save()

    def _collect_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # if we use a float value as a budget, we want to train the model only on the highest budget
        available_budgets = []
        for run_key in self.runhistory.data.keys():
            available_budgets.append(run_key.budget)

        # Sort available budgets from highest to lowest budget
        available_budgets = sorted(list(set(available_budgets)), reverse=True)

        # Get #points per budget and if there are enough samples, then build a model
        for b in available_budgets:
            X, Y = self.runhistory_encoder.transform(
                self.runhistory,
                budget_subset=[
                    b,
                ],
            )
            if X.shape[0] >= self.min_samples_model:
                self.currently_considered_budgets = [
                    b,
                ]
                configs_array = self.runhistory_encoder.get_configurations(
                    self.runhistory, budget_subset=self.currently_considered_budgets
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
        return self.runhistory.get_configs_per_budget(budget_subset=self.currently_considered_budgets)

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
        if predict:
            model = self.model
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
            all_configs = self.runhistory.get_configs_per_budget(budget_subset=self.currently_considered_budgets)
            x_best = self.incumbent
            x_best_array = convert_configurations_to_array(all_configs)
            best_observation = self.runhistory.get_cost(x_best)
            best_observation_as_array = np.array(best_observation).reshape((1, 1))

            # It's unclear how to do this for inv scaling and potential future scaling.
            # This line should be changed if necessary
            best_observation = self.runhistory_encoder.transform_response_values(best_observation_as_array)
            best_observation = best_observation[0][0]

        return x_best_array, best_observation
