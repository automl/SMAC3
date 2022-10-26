from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from ConfigSpace import Configuration

from smac.main.base_smbo import BaseSMBO
from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.runhistory.enumerations import TrialInfoIntent
from smac.runner.exceptions import (
    FirstRunCrashedException,
    TargetAlgorithmAbortException,
)
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class SMBO(BaseSMBO):
    """Implements ``get_next_configurations``, ``ask``, and ``tell``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self._predict_x_best = True
        self._min_samples = 1
        self._considered_budgets: list[float | None] = [None]

    def get_next_configurations(self, n: int | None = None) -> Iterator[Configuration]:  # noqa: D102
        # TODO: Let's return the initial configurations from this method too

        for callback in self._callbacks:
            callback.on_next_configurations_start(self)

        # Cost value of incumbent configuration (required for acquisition function).
        # If not given, it will be inferred from runhistory or predicted.
        # If not given and runhistory is empty, it will raise a ValueError.
        incumbent_value: float | None = None

        logger.debug("Search for next configuration...")
        X, Y, X_configurations = self._collect_data()
        previous_configs = self._runhistory.get_configs()

        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of random search iterations.
            # We got rid of random search here and replaced it with a simple configuration sampling from
            # the configspace.
            return iter([self._scenario.configspace.sample_configuration(1)])

        # TODO: Check if X/Y differs from the last run, otherwise use cached results
        self._model.train(X, Y)

        x_best_array: np.ndarray | None = None
        if incumbent_value is not None:
            best_observation = incumbent_value
        else:
            if self._runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of the incumbent is unknown.")

            x_best_array, best_observation = self._get_x_best(self._predict_x_best, X_configurations)

        self._acquisition_function.update(
            model=self._model,
            eta=best_observation,
            incumbent_array=x_best_array,
            num_data=len(self._get_evaluated_configs()),
            X=X_configurations,
        )

        challengers = self._acquisition_maximizer.maximize(
            previous_configs,
            n_points=n,
            random_design=self._random_design,
        )

        for callback in self._callbacks:
            callback.on_next_configurations_end(self)

        return challengers

    def ask(self) -> tuple[TrialInfoIntent, TrialInfo]:  # noqa: D102
        for callback in self._callbacks:
            callback.on_ask_start(self)

        if self._runhistory_encoder.multi_objective_algorithm is not None:
            self._runhistory_encoder.multi_objective_algorithm.update_on_iteration_start()

        intent, trial_info = self._intensifier.get_next_trial(
            challengers=self._initial_design_configs,
            incumbent=self._incumbent,
            get_next_configurations=self.get_next_configurations,
            runhistory=self._runhistory,
            repeat_configs=self._intensifier.repeat_configs,
            n_workers=self._runner.count_available_workers(),
        )

        if intent == TrialInfoIntent.RUN:
            # There are 2 criteria that the stats object uses to know if the budged was exhausted.
            # The budget time, which can only be known when the run finishes,
            # And the number of ta executions. Because we submit the job at this point,
            # we count this submission as a run. This prevent for using more
            # runner runs than what the config allows.
            self._stats._submitted += 1

        # Remove config from initial design challengers to not repeat it again
        self._initial_design_configs = [c for c in self._initial_design_configs if c != trial_info.config]

        for callback in self._callbacks:
            callback.on_ask_end(self, intent, trial_info)

        return intent, trial_info

    def tell(
        self,
        info: TrialInfo,
        value: TrialValue,
        time_left: float | None = None,
        save: bool = True,
    ) -> None:  # noqa: D102
        # We first check if budget/instance/seed is supported by the intensifier
        if info.seed not in (seeds := self._intensifier.get_target_function_seeds()):
            raise ValueError(f"Seed {info.seed} is not supported by the intensifier. Consider using one of {seeds}.")
        elif info.budget not in (budgets := self._intensifier.get_target_function_budgets()):
            raise ValueError(
                f"Budget {info.budget} is not supported by the intensifier. Consider using one of {budgets}."
            )
        elif info.instance not in (instances := self._intensifier.get_target_function_instances()):
            raise ValueError(
                f"Instance {info.instance} is not supported by the intensifier. Consider using one of {instances}."
            )

        if info.config.origin is None:
            info.config.origin = "Custom"

        for callback in self._callbacks:
            response = callback.on_tell_start(self, info, value)
            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called.
            if response is False:
                logger.info("An callback returned False. Abort is requested.")
                self._stop = True

        # We expect the first run to always succeed.
        if self._stats.finished == 0 and value.status == StatusType.CRASHED:
            additional_info = ""
            if "traceback" in value.additional_info:
                additional_info = "\n\n" + value.additional_info["traceback"]

            raise FirstRunCrashedException("The first run crashed. Please check your setup again." + additional_info)

        # Update SMAC stats
        self._stats._target_function_walltime_used += float(value.time)
        self._stats._finished += 1

        logger.debug(
            f"Status: {value.status}, cost: {value.cost}, time: {value.time}, " f"Additional: {value.additional_info}"
        )

        self._runhistory.add(
            config=info.config,
            cost=value.cost,
            time=value.time,
            status=value.status,
            instance=info.instance,
            seed=info.seed,
            budget=info.budget,
            starttime=value.starttime,
            endtime=value.endtime,
            force_update=True,
            additional_info=value.additional_info,
        )
        self._stats._n_configs = len(self._runhistory._config_ids)

        if value.status == StatusType.ABORT:
            raise TargetAlgorithmAbortException(
                "The target function was aborted. The last incumbent can be found in the trajectory file."
            )
        elif value.status == StatusType.STOP:
            logger.debug("Value holds the status stop. Abort is requested.")
            self._stop = True

        if time_left is None:
            time_left = np.inf

        # Update the intensifier with the result of the runs
        self._incumbent, _ = self._intensifier.process_results(
            trial_info=info,
            trial_value=value,
            incumbent=self._incumbent,
            runhistory=self._runhistory,
            time_bound=max(self._min_time, time_left),
        )

        # Gracefully end optimization if termination cost is reached
        if self._scenario.termination_cost_threshold != np.inf:

            cost = self.runhistory.average_cost(info.config)

            if not isinstance(cost, list):
                cost = [cost]

            if not isinstance(self._scenario.termination_cost_threshold, list):
                cost_threshold = [self._scenario.termination_cost_threshold]
            else:
                cost_threshold = self._scenario.termination_cost_threshold

            if len(cost) != len(cost_threshold):
                raise RuntimeError("You must specify a termination cost threshold for each objective.")

            if all(cost[i] < cost_threshold[i] for i in range(len(cost))):
                logger.info("Cost threshold was reached. Abort is requested.")
                self._stop = True

        for callback in self._callbacks:
            response = callback.on_tell_end(self, info, value)
            # If a callback returns False, the optimization loop should be interrupted
            # the other callbacks are still being called.
            if response is False:
                logger.info("An callback returned False. Abort is requested.")
                self._stop = True

        if save:
            self.save()

    def _collect_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Collects the data from the runhistory to train the surrogate model. The data collection strategy if budgets
        are used is as follows: Looking from highest to lowest budget, return those observations
        that support at least ``self._min_samples`` points.

        If no budgets are used, this is equivalent to returning all observations.
        """
        # if we use a float value as a budget, we want to train the model only on the highest budget
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

    def _get_x_best(self, predict: bool, X: np.ndarray) -> tuple[np.ndarray, float]:
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
        if predict:
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
            # won't need log(y) if EPM was already trained on log(y)
        else:
            all_configs = self._runhistory.get_configs_per_budget(budget_subset=self._considered_budgets)
            x_best = self._incumbent
            x_best_array = convert_configurations_to_array(all_configs)
            best_observation = self._runhistory.get_cost(x_best)
            best_observation_as_array = np.array(best_observation).reshape((1, 1))

            # It's unclear how to do this for inv scaling and potential future scaling.
            # This line should be changed if necessary
            best_observation = self._runhistory_encoder.transform_response_values(best_observation_as_array)
            best_observation = best_observation[0][0]

        return x_best_array, best_observation
